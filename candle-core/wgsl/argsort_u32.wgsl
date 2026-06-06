// Bitonic argsort along the last dimension (u32).

const MAX_NCOLS_PAD: u32 = 1024u;
const WG_SIZE: u32 = 256u;

struct ArgSortParams {
    ncols: u32,
    ncols_pad: u32,
    asc: u32,
    _pad: array<u32, 69>,
}

@group(0) @binding(0)
var<storage, read_write> dst_buf: array<u32>;

@group(0) @binding(1)
var<storage, read> src_buf: array<u32>;

@group(0) @binding(2)
var<storage, read> src_pad: array<u32>;

@group(0) @binding(3)
var<storage, read> argsort_params: ArgSortParams;

var<workgroup> dst_row: array<u32, 1024>;

fn swap_indices(a: u32, b: u32) {
    let tmp = dst_row[a];
    dst_row[a] = dst_row[b];
    dst_row[b] = tmp;
}

fn cmp_gt(row_base: u32, ia: u32, ib: u32) -> bool {
    return src_buf[row_base + ia] > src_buf[row_base + ib];
}

fn cmp_lt(row_base: u32, ia: u32, ib: u32) -> bool {
    return src_buf[row_base + ia] < src_buf[row_base + ib];
}

fn bitonic_sort_row(row: u32, local_id: u32) {
    let ncols = argsort_params.ncols;
    let ncols_pad = argsort_params.ncols_pad;
    let asc = argsort_params.asc;
    let row_base = row * ncols;

    for (var col = local_id; col < ncols_pad; col = col + WG_SIZE) {
        dst_row[col] = col;
    }
    workgroupBarrier();

    var k = 2u;
    while (k <= ncols_pad) {
        var j = k / 2u;
        while (j > 0u) {
            for (var col = local_id; col < ncols_pad; col = col + WG_SIZE) {
                let ixj = col ^ j;
                if (ixj > col) {
                    if ((col & k) == 0u) {
                        if (dst_row[col] >= ncols ||
                            (dst_row[ixj] < ncols &&
                                ((asc == 1u && cmp_gt(row_base, dst_row[col], dst_row[ixj])) ||
                                    (asc == 0u && cmp_lt(row_base, dst_row[col], dst_row[ixj]))))) {
                            swap_indices(col, ixj);
                        }
                    } else {
                        if (dst_row[ixj] >= ncols ||
                            (dst_row[col] < ncols &&
                                ((asc == 1u && cmp_lt(row_base, dst_row[col], dst_row[ixj])) ||
                                    (asc == 0u && cmp_gt(row_base, dst_row[col], dst_row[ixj]))))) {
                            swap_indices(col, ixj);
                        }
                    }
                }
            }
            workgroupBarrier();
            j = j / 2u;
        }
        k = k * 2u;
    }

    for (var col = local_id; col < ncols; col = col + WG_SIZE) {
        dst_buf[row_base + col] = dst_row[col];
    }
}

@compute @workgroup_size(WG_SIZE)
fn asort_asc_u32(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    bitonic_sort_row(wg_id.x, local_id.x);
}

@compute @workgroup_size(WG_SIZE)
fn asort_desc_u32(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    bitonic_sort_row(wg_id.x, local_id.x);
}
