// Tiled bf16×bf16 matrix multiply with f32 output (no atomic packed writes).
//
// Entry points: matmul_tiled_bf16acc, matmul_tiled_vec_bf16acc

const TILE: u32 = MATMUL_WG_SIZE;
const VEC: u32 = 4u;

var<workgroup> tile_a: array<f32, 256>;
var<workgroup> tile_b: array<f32, 256>;

fn tile_dot_vec4_bf16(ty: u32, tx: u32, k_base: u32) -> f32 {
    let a_vec = vec4<f32>(
        tile_a[ty * TILE + k_base],
        tile_a[ty * TILE + k_base + 1u],
        tile_a[ty * TILE + k_base + 2u],
        tile_a[ty * TILE + k_base + 3u],
    );
    let b_vec = vec4<f32>(
        tile_b[k_base * TILE + tx],
        tile_b[(k_base + 1u) * TILE + tx],
        tile_b[(k_base + 2u) * TILE + tx],
        tile_b[(k_base + 3u) * TILE + tx],
    );
    return dot(a_vec, b_vec);
}

@compute @workgroup_size(TILE, TILE)
fn matmul_tiled_bf16acc(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let m = mm_params.m;
    let n = mm_params.n;
    let k_dim = mm_params.k;
    let batch = wg_id.z;

    let row = wg_id.y * TILE + local_id.y;
    let col = wg_id.x * TILE + local_id.x;
    let ty = local_id.y;
    let tx = local_id.x;

    var acc = 0.0;
    let num_tiles = (k_dim + TILE - 1u) / TILE;

    for (var t = 0u; t < num_tiles; t = t + 1u) {
        let a_col = t * TILE + tx;
        let b_row = t * TILE + ty;

        let a_row = wg_id.y * TILE + ty;
        if (a_row < m && a_col < k_dim) {
            tile_a[ty * TILE + tx] = mm_load_a(batch, a_row, a_col);
        } else {
            tile_a[ty * TILE + tx] = 0.0;
        }

        let b_col = wg_id.x * TILE + tx;
        if (b_row < k_dim && b_col < n) {
            tile_b[ty * TILE + tx] = mm_load_b(batch, b_row, b_col);
        } else {
            tile_b[ty * TILE + tx] = 0.0;
        }

        workgroupBarrier();

        for (var k = 0u; k < TILE; k = k + 1u) {
            acc += tile_a[ty * TILE + k] * tile_b[k * TILE + tx];
        }

        workgroupBarrier();
    }

    if (row < m && col < n) {
        mm_store_c(batch, row, col, acc);
    }
}

@compute @workgroup_size(TILE, TILE)
fn matmul_tiled_vec_bf16acc(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let m = mm_params.m;
    let n = mm_params.n;
    let k_dim = mm_params.k;
    let batch = wg_id.z;

    let row = wg_id.y * TILE + local_id.y;
    let col = wg_id.x * TILE + local_id.x;
    let ty = local_id.y;
    let tx = local_id.x;

    var acc = 0.0;
    let num_tiles = (k_dim + TILE - 1u) / TILE;

    for (var t = 0u; t < num_tiles; t = t + 1u) {
        let a_col = t * TILE + tx;
        let b_row = t * TILE + ty;

        let a_row = wg_id.y * TILE + ty;
        if (a_row < m && a_col < k_dim) {
            tile_a[ty * TILE + tx] = mm_load_a(batch, a_row, a_col);
        } else {
            tile_a[ty * TILE + tx] = 0.0;
        }

        let b_col = wg_id.x * TILE + tx;
        if (b_row < k_dim && b_col < n) {
            tile_b[ty * TILE + tx] = mm_load_b(batch, b_row, b_col);
        } else {
            tile_b[ty * TILE + tx] = 0.0;
        }

        workgroupBarrier();

        for (var k = 0u; k < TILE; k = k + VEC) {
            if (k + VEC <= TILE) {
                acc += tile_dot_vec4_bf16(ty, tx, k);
            } else {
                for (var kk = k; kk < TILE; kk = kk + 1u) {
                    acc += tile_a[ty * TILE + kk] * tile_b[kk * TILE + tx];
                }
            }
        }

        workgroupBarrier();
    }

    if (row < m && col < n) {
        mm_store_c(batch, row, col, acc);
    }
}
