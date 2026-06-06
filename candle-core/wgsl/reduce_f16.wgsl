// Reduction kernels: sum, mean, argmax.
//
// One workgroup produces one output element. Dispatch `dst_elem_count` workgroups
// of size `REDUCE_WG_SIZE` (32, tuned for Intel integrated GPUs).
//
// Entry points: reduce_sum_f16, reduce_mean_f16, reduce_max_f16, reduce_min_f16,
//               reduce_argmax_f16, reduce_argmin_f16

@compute @workgroup_size(REDUCE_WG_SIZE)
fn reduce_sum_f16(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let dst_id = wg_id.x;
    if (dst_id >= reduce_params.dst_elem_count) {
        return;
    }

    let chunk = reduce_params.reduce_chunk_size;
    let tid = local_id.x;

    var acc = f16(0.0);
    var chunk_off = tid;
    while (chunk_off < chunk) {
        acc += reduce_load_src(dst_id, chunk_off);
        chunk_off += REDUCE_WG_SIZE;
    }
    wg_sum[tid] = acc;
    workgroup_reduce_sum(tid);

    if (tid == 0u) {
        reduce_out[dst_id] = wg_sum[0];
    }
}

@compute @workgroup_size(REDUCE_WG_SIZE)
fn reduce_mean_f16(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let dst_id = wg_id.x;
    if (dst_id >= reduce_params.dst_elem_count) {
        return;
    }

    let chunk = reduce_params.reduce_chunk_size;
    let tid = local_id.x;

    var acc = f16(0.0);
    var chunk_off = tid;
    while (chunk_off < chunk) {
        acc += reduce_load_src(dst_id, chunk_off);
        chunk_off += REDUCE_WG_SIZE;
    }
    wg_sum[tid] = acc;
    workgroup_reduce_sum(tid);

    if (tid == 0u) {
        let count = f16(chunk);
        reduce_out[dst_id] = wg_sum[0] / count;
    }
}

@compute @workgroup_size(REDUCE_WG_SIZE)
fn reduce_max_f16(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let dst_id = wg_id.x;
    if (dst_id >= reduce_params.dst_elem_count) {
        return;
    }

    let chunk = reduce_params.reduce_chunk_size;
    let tid = local_id.x;

    var best = f16(-65504.0);
    var chunk_off = tid;
    while (chunk_off < chunk) {
        best = max(best, reduce_load_src(dst_id, chunk_off));
        chunk_off += REDUCE_WG_SIZE;
    }
    wg_max_val[tid] = best;
    workgroup_reduce_max(tid);

    if (tid == 0u) {
        reduce_out[dst_id] = wg_max_val[0];
    }
}

@compute @workgroup_size(REDUCE_WG_SIZE)
fn reduce_min_f16(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let dst_id = wg_id.x;
    if (dst_id >= reduce_params.dst_elem_count) {
        return;
    }

    let chunk = reduce_params.reduce_chunk_size;
    let tid = local_id.x;

    var best = f16(65504.0);
    var chunk_off = tid;
    while (chunk_off < chunk) {
        best = min(best, reduce_load_src(dst_id, chunk_off));
        chunk_off += REDUCE_WG_SIZE;
    }
    wg_max_val[tid] = best;
    workgroup_reduce_min(tid);

    if (tid == 0u) {
        reduce_out[dst_id] = wg_max_val[0];
    }
}

// Argmax along the reduced dimension; stores the index as f16.
@compute @workgroup_size(REDUCE_WG_SIZE)
fn reduce_argmax_f16(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let dst_id = wg_id.x;
    if (dst_id >= reduce_params.dst_elem_count) {
        return;
    }

    let chunk = reduce_params.reduce_chunk_size;
    let tid = local_id.x;

    var best_val = f16(-65504.0); // -f32::MAX approx
    var best_idx = 0u;
    var found = false;

    var chunk_off = tid;
    while (chunk_off < chunk) {
        let v = reduce_load_src(dst_id, chunk_off);
        if (!found || v > best_val) {
            best_val = v;
            best_idx = chunk_off;
            found = true;
        }
        chunk_off += REDUCE_WG_SIZE;
    }

    wg_max_val[tid] = best_val;
    wg_max_idx[tid] = best_idx;
    workgroup_reduce_max(tid);

    if (tid == 0u) {
        reduce_out[dst_id] = f16(wg_max_idx[0]);
    }
}

// Argmin along the reduced dimension; stores the index as f16.
@compute @workgroup_size(REDUCE_WG_SIZE)
fn reduce_argmin_f16(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let dst_id = wg_id.x;
    if (dst_id >= reduce_params.dst_elem_count) {
        return;
    }

    let chunk = reduce_params.reduce_chunk_size;
    let tid = local_id.x;

    var best_val = f16(65504.0);
    var best_idx = 0u;
    var found = false;

    var chunk_off = tid;
    while (chunk_off < chunk) {
        let v = reduce_load_src(dst_id, chunk_off);
        if (!found || v < best_val) {
            best_val = v;
            best_idx = chunk_off;
            found = true;
        }
        chunk_off += REDUCE_WG_SIZE;
    }

    wg_max_val[tid] = best_val;
    wg_max_idx[tid] = best_idx;
    workgroup_reduce_argmin(tid);

    if (tid == 0u) {
        reduce_out[dst_id] = f16(wg_max_idx[0]);
    }
}
