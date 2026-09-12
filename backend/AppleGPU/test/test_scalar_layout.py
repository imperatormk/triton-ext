"""The packed scalar buffer is laid out twice, and the two must agree.

`_compute_scalar_layout` in driver.py places the bytes; `planKernelAbi` in
agpu/include/agpu/emit/KernelAbi.h emits the offsets the kernel reads them
back from. A drift between them is a silently wrong scalar argument.
"""

from __future__ import annotations

import random
import re
from pathlib import Path

import pytest

from triton_apple_backend.driver import _compute_scalar_layout
from triton_apple_backend.tables import SCALAR_PACK_INFO

ELEM_TYPE_H = (Path(__file__).resolve().parents[1] / "agpu" / "include" /
               "agpu" / "plan" / "ElemType.h")

BITS = {
    "i1": 1,
    "u1": 1,
    "i8": 8,
    "u8": 8,
    "i16": 16,
    "u16": 16,
    "i32": 32,
    "u32": 32,
    "i64": 64,
    "u64": 64,
    "fp16": 16,
    "bf16": 16,
    "fp32": 32,
    "fp64": 64,
}


def cpp_layout(types):
    """planKernelAbi's loop: natural alignment, byteWidthOf for the size."""
    offset, offsets = 0, []
    for ty in types:
        size = (BITS[ty] + 7) // 8
        if size > 0:
            offset = (offset + size - 1) // size * size
        offsets.append(offset)
        offset += size
    return offset, offsets


def test_byte_width_still_rounds_the_bit_count():
    src = ELEM_TYPE_H.read_text()
    assert re.search(r"\(int64_t\)\(\(e\.bits \+ 7u\) / 8u\)", src), \
        "byteWidthOf changed; BITS here no longer models it"


def test_every_packable_type_has_a_bit_width():
    assert set(SCALAR_PACK_INFO) == set(BITS)


@pytest.mark.parametrize("ty", sorted(BITS))
def test_the_pack_size_matches_the_emitter(ty):
    _, size, _ = SCALAR_PACK_INFO[ty]
    assert size == (BITS[ty] + 7) // 8


def test_both_layouts_agree_on_random_signatures():
    rng = random.Random(0)
    types = sorted(BITS)
    for _ in range(2000):
        seq = [rng.choice(types) for _ in range(rng.randint(1, 8))]
        assert _compute_scalar_layout(seq) == cpp_layout(seq), seq
