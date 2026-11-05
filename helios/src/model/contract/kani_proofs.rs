/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

use super::*;

#[kani::proof]
#[kani::unwind(16)]
fn state_view_containment_matches_wide_arithmetic() {
    let same_buffer: bool = kani::any();
    let outer = StateView {
        buffer: "state".into(),
        offset: kani::any(),
        stride: kani::any(),
        nbytes: kani::any(),
    };
    let inner = StateView {
        buffer: if same_buffer { "state" } else { "other" }.into(),
        offset: kani::any(),
        stride: kani::any(),
        nbytes: kani::any(),
    };
    let outer_end = outer.offset as u128 + outer.nbytes as u128;
    let inner_end = inner.offset as u128 + inner.nbytes as u128;
    let expected = same_buffer
        && outer.stride == inner.stride
        && outer.offset <= inner.offset
        && outer_end <= usize::MAX as u128
        && inner_end <= usize::MAX as u128
        && inner_end <= outer_end;
    assert_eq!(outer.contains(&inner), expected);
}

#[kani::proof]
#[kani::unwind(16)]
fn state_view_storage_validation_matches_wide_arithmetic() {
    use crate::execution::Dtype;
    let (dtype, element_bytes) = match kani::any::<u8>() % 8 {
        0 => (Dtype::Bool, 1u128),
        1 => (Dtype::Uint8, 1),
        2 => (Dtype::Float8E4m3fn, 1),
        3 => (Dtype::Float16, 2),
        4 => (Dtype::Bfloat16, 2),
        5 => (Dtype::Int32, 4),
        6 => (Dtype::Float32, 4),
        _ => (Dtype::Int64, 8),
    };
    let rows: usize = kani::any();
    let elements: usize = kani::any();
    let nbytes: usize = kani::any();
    let has_shape: bool = kani::any();
    let has_stride: bool = kani::any();
    let mut shape = vec![rows];
    let mut strides = vec![elements];
    if !has_shape {
        shape.clear();
    }
    if !has_stride {
        strides.clear();
    }
    let view = StateView {
        buffer: String::new(),
        offset: kani::any(),
        stride: kani::any(),
        nbytes: kani::any(),
    };
    let buffer = BufferDescriptor {
        name: String::new(),
        address: 4096,
        nbytes,
        dtype,
        shape,
        strides,
    };
    let expected = view.nbytes != 0
        && has_shape
        && has_stride
        && elements as u128 * element_bytes == view.stride as u128
        && view.offset as u128 + view.nbytes as u128 <= view.stride as u128
        && rows as u128 * view.stride as u128 <= nbytes as u128;
    assert_eq!(view.fits_storage(&buffer), expected);
}
