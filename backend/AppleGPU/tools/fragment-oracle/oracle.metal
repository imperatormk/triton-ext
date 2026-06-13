#include <metal_stdlib>
using namespace metal;

// Fragment-ABI hardware-layout oracle.
//
// One 8x8 simdgroup tile. The fragment ABI stores it as, per lane T (0..31),
// two registers R in {0,1}. The PHYSICAL per-lane layout (verified on M1, and
// encoded in AppleMmaLayoutConversions.cpp) is:
//   phys_row = L1 | (L2<<1) | (L4<<2)
//   phys_col = (L0<<1) | (L3<<2) | R
// where Li is bit i of the lane index T.
//
// This kernel:
//   - loads a known 8x8 pattern into per-lane regs via that exact layout,
//   - performs col-replicate broadcast, row-replicate broadcast, and
//     expand_dims, each as an air.simd_shuffle network,
//   - writes every (lane,reg) result out so the host can compare BIT-EXACT
//     against a CPU reference of the same op on the same pattern.
//
// Per dest lane T, reg R:
//   col-replicate (out[r][c]=in[r][0]): src lane = T & ~0b01001, src reg = 0
//   row-replicate (out[r][c]=in[0][c]): src lane = T & ~0b10110, src reg = R
//   expand_dims (slice broadcast along the new axis): identical lane network
//     to the matching replicate; here we exercise the col-replicate form,
//     i.e. a length-8 slice indexed by row, placed across all columns.

constant uint COL_CLR = 0x9;  // bits L0,L3  -> clear for col-replicate
constant uint ROW_CLR = 0x16; // bits L1,L2,L4 -> clear for row-replicate

kernel void frag_oracle(device const float *inPat   [[buffer(0)]], // 8x8 row-major
                        device float       *loadOut [[buffer(1)]], // [lane*2+reg]
                        device float       *colOut  [[buffer(2)]],
                        device float       *rowOut  [[buffer(3)]],
                        device float       *expOut  [[buffer(4)]],
                        constant uint      &mutate  [[buffer(5)]], // 0=correct; else negative control
                        uint lane [[thread_index_in_simdgroup]])
{
    // Negative-control mutation: if mutate!=0, deliberately swap the two clear
    // masks so the oracle MUST report FAIL (proves it catches a wrong shuffle).
    uint colClr = mutate ? ROW_CLR : COL_CLR;
    uint rowClr = mutate ? COL_CLR : ROW_CLR;
    // Load: compute (row,col) for each reg from this lane, fetch from pattern.
    uint L0 = (lane >> 0) & 1, L1 = (lane >> 1) & 1, L2 = (lane >> 2) & 1;
    uint L3 = (lane >> 3) & 1, L4 = (lane >> 4) & 1;
    uint row = L1 | (L2 << 1) | (L4 << 2);

    float reg[2];
    for (uint R = 0; R < 2; ++R) {
        uint col = (L0 << 1) | (L3 << 2) | R;
        reg[R] = inPat[row * 8 + col];
        loadOut[lane * 2 + R] = reg[R];
    }

    // col-replicate: out[r][c] = in[r][0].
    // src lane = lane & ~COL_CLR, src reg = 0. simd_shuffle the reg-0 value.
    {
        uint src = lane & ~colClr;
        float v = simd_shuffle(reg[0], src);
        colOut[lane * 2 + 0] = v;
        colOut[lane * 2 + 1] = v;
    }

    // row-replicate: out[r][c] = in[0][c].
    // src lane = lane & ~ROW_CLR, src reg = R. Shuffle each reg independently.
    {
        uint src = lane & ~rowClr;
        rowOut[lane * 2 + 0] = simd_shuffle(reg[0], src);
        rowOut[lane * 2 + 1] = simd_shuffle(reg[1], src);
    }

    // expand_dims: a length-8 slice s[row] (per-row scalar, here = inPat[row*8])
    // expanded across all columns of the tile. Same lane network as
    // col-replicate, but the source value is the slice element for this row.
    {
        uint src = lane & ~colClr;
        float sliceVal = simd_shuffle(reg[0], src); // reg[0] at src holds in[row][0]
        expOut[lane * 2 + 0] = sliceVal;
        expOut[lane * 2 + 1] = sliceVal;
    }
}
