// Element-wise binary kernels (packed bf16) for LLM-critical ops.
// Serial execution: two bf16 values share one u32 word (no write races).
//
// Entry points: add_bf16, sub_bf16, mul_bf16, div_bf16

@compute @workgroup_size(1)
fn add_bf16(@builtin(workgroup_id) wg_id: vec3<u32>) {
    if (wg_id.x != 0u) {
        return;
    }
    for (var i = 0u; i < kernel_params.elem_count; i = i + 1u) {
        store_bf16_out(i, load_bf16_in0(i) + load_bf16_in1(i));
    }
}

@compute @workgroup_size(1)
fn sub_bf16(@builtin(workgroup_id) wg_id: vec3<u32>) {
    if (wg_id.x != 0u) {
        return;
    }
    for (var i = 0u; i < kernel_params.elem_count; i = i + 1u) {
        store_bf16_out(i, load_bf16_in0(i) - load_bf16_in1(i));
    }
}

@compute @workgroup_size(1)
fn mul_bf16(@builtin(workgroup_id) wg_id: vec3<u32>) {
    if (wg_id.x != 0u) {
        return;
    }
    for (var i = 0u; i < kernel_params.elem_count; i = i + 1u) {
        store_bf16_out(i, load_bf16_in0(i) * load_bf16_in1(i));
    }
}

@compute @workgroup_size(1)
fn div_bf16(@builtin(workgroup_id) wg_id: vec3<u32>) {
    if (wg_id.x != 0u) {
        return;
    }
    for (var i = 0u; i < kernel_params.elem_count; i = i + 1u) {
        store_bf16_out(i, load_bf16_in0(i) / load_bf16_in1(i));
    }
}
