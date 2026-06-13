#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <stdio.h>

// Host oracle: drives frag_oracle.metallib, reads back per-(lane,reg), and
// compares BIT-EXACT against a CPU reference of load / col-replicate /
// row-replicate / expand_dims on the same 8x8 pattern. Independent of any fla
// numeric threshold: a single wrong lane => FAIL.

// Physical layout (must match oracle.metal / AppleMmaLayoutConversions.cpp).
static void slotOf(uint lane, uint R, uint *row, uint *col) {
    uint L0=(lane>>0)&1,L1=(lane>>1)&1,L2=(lane>>2)&1,L3=(lane>>3)&1,L4=(lane>>4)&1;
    *row = L1 | (L2<<1) | (L4<<2);
    *col = (L0<<1) | (L3<<2) | R;
}

static int cmpSlots(const char *name, const float *got, const float *ref) {
    int bad = 0;
    for (uint lane = 0; lane < 32; ++lane)
        for (uint R = 0; R < 2; ++R) {
            int i = lane*2+R;
            if (got[i] != ref[i]) { // bit-exact (both came from same int pattern)
                if (bad < 8)
                    printf("    MISMATCH lane=%2u reg=%u got=%.1f ref=%.1f\n",
                           lane, R, got[i], ref[i]);
                bad++;
            }
        }
    printf("  [%s] %s (%d/64 lanes wrong)\n", name, bad==0?"PASS":"FAIL", bad);
    return bad==0;
}

int main(int argc, char **argv) { @autoreleasepool {
    const char *libpath = argc > 1 ? argv[1]
        : "/Users/zimski/projects/oss/triton-ext/backend/AppleGPU/tools/fragment-oracle/oracle.metallib";
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> q = [dev newCommandQueue];
    NSError *err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:libpath]] error:&err];
    if (!lib) { NSLog(@"lib err %@", err); return 1; }
    id<MTLFunction> fn = [lib newFunctionWithName:@"frag_oracle"];
    id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) { NSLog(@"pso err %@", err); return 1; }

    // 8x8 pattern: distinct integers (exactly representable in f32) so equality
    // is bit-exact. pat[r][c] = r*8 + c + 1.
    id<MTLBuffer> bIn   = [dev newBufferWithLength:64*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bLoad = [dev newBufferWithLength:64*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bCol  = [dev newBufferWithLength:64*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bRow  = [dev newBufferWithLength:64*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bExp  = [dev newBufferWithLength:64*4 options:MTLResourceStorageModeShared];
    float *pat = bIn.contents;
    for (uint r=0;r<8;r++) for (uint c=0;c<8;c++) pat[r*8+c] = (float)(r*8+c+1);

    uint mutate = 0;
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bIn offset:0 atIndex:0]; [enc setBuffer:bLoad offset:0 atIndex:1];
    [enc setBuffer:bCol offset:0 atIndex:2]; [enc setBuffer:bRow offset:0 atIndex:3];
    [enc setBuffer:bExp offset:0 atIndex:4]; [enc setBytes:&mutate length:4 atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
    if (cb.error) { NSLog(@"dispatch err %@", cb.error); return 1; }

    // CPU references, indexed [lane*2+reg].
    float rLoad[64], rCol[64], rRow[64], rExp[64];
    for (uint lane=0;lane<32;lane++) for (uint R=0;R<2;R++) {
        uint row,col; slotOf(lane,R,&row,&col);
        rLoad[lane*2+R] = pat[row*8+col];           // identity load
        rCol [lane*2+R] = pat[row*8+0];             // col-replicate: in[r][0]
        rRow [lane*2+R] = pat[0*8+col];             // row-replicate: in[0][c]
        rExp [lane*2+R] = pat[row*8+0];             // expand_dims slice s[row]
    }

    printf("== Fragment-ABI hardware-layout oracle (8x8 simdgroup tile) ==\n");
    int ok = 1;
    ok &= cmpSlots("load",          bLoad.contents, rLoad);
    ok &= cmpSlots("col-replicate", bCol.contents,  rCol);
    ok &= cmpSlots("row-replicate", bRow.contents,  rRow);
    ok &= cmpSlots("expand_dims",   bExp.contents,  rExp);

    // Negative control: with mutate=1 the shuffle masks are deliberately
    // swapped, so col/row-replicate MUST fail. If they pass, the oracle is not
    // actually discriminating -> the whole check is untrustworthy.
    mutate = 1;
    cb = [q commandBuffer]; enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bIn offset:0 atIndex:0]; [enc setBuffer:bLoad offset:0 atIndex:1];
    [enc setBuffer:bCol offset:0 atIndex:2]; [enc setBuffer:bRow offset:0 atIndex:3];
    [enc setBuffer:bExp offset:0 atIndex:4]; [enc setBytes:&mutate length:4 atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
    int colWrong=0,rowWrong=0;
    float *gc=bCol.contents,*gr=bRow.contents;
    for (int i=0;i<64;i++){ if(gc[i]!=rCol[i])colWrong++; if(gr[i]!=rRow[i])rowWrong++; }
    int negOk = (colWrong>0 && rowWrong>0);
    printf("  [neg-control] %s (swapped masks: col %d/64 row %d/64 wrong -> expected nonzero)\n",
           negOk?"PASS":"FAIL", colWrong, rowWrong);
    ok &= negOk;

    printf("== ORACLE %s ==\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 2;
}}
