// PCG-based random number generation (uniform and normal).
//
// Entry points: rand_uniform_f32, rand_normal_f32

const WG_SIZE: u32 = 32u;
const UNIF01_INV32: f32 = 2.328306436538696289e-10;
const TWO_PI: f32 = 6.28318530718;

struct RandomParams {
    elem_count: u32,
    seed_lo: u32,
    seed_hi: u32,
    param0: f32,
    param1: f32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0)
var<storage, read_write> output_buf: array<f32>;

@group(0) @binding(1)
var<storage, read> random_params: RandomParams;

fn pcg_hash(input: u32) -> u32 {
    let state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand_u01(seed_lo: u32, seed_hi: u32, idx: u32) -> f32 {
    let mixed = pcg_hash(seed_lo ^ idx ^ pcg_hash(seed_hi + idx));
    return f32(mixed) * UNIF01_INV32;
}

fn grid_stride_x(num_wg: vec3<u32>) -> u32 {
    return WG_SIZE * num_wg.x;
}

@compute @workgroup_size(WG_SIZE)
fn rand_uniform_f32(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let count = random_params.elem_count;
    let lo = random_params.param0;
    let hi = random_params.param1;
    let diff = hi - lo;
    let stride = grid_stride_x(num_wg);
    let off = 1u - count % 2u;

    for (var i = gid.x; i < count; i = i + stride) {
        let u = rand_u01(random_params.seed_lo, random_params.seed_hi, i);
        output_buf[i] = u * diff + lo;
        let mirror = count - off - i;
        if (mirror < count && mirror != i) {
            let u2 = rand_u01(
                random_params.seed_lo,
                random_params.seed_hi,
                i ^ 0x9E3779B9u,
            );
            output_buf[mirror] = u2 * diff + lo;
        }
    }
}

@compute @workgroup_size(WG_SIZE)
fn rand_normal_f32(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let count = random_params.elem_count;
    let mean = random_params.param0;
    let stddev = random_params.param1;
    let stride = grid_stride_x(num_wg);
    let off = 1u - count % 2u;

    for (var i = gid.x; i < count; i = i + stride) {
        let u1 = max(rand_u01(random_params.seed_lo, random_params.seed_hi, i), 1e-7);
        let u2 = rand_u01(
            random_params.seed_lo,
            random_params.seed_hi,
            i ^ 0x85EBCA6Bu,
        );
        let mag = stddev * sqrt(-2.0 * log(u1));
        let angle = TWO_PI * u2;
        output_buf[i] = mag * cos(angle) + mean;

        let mirror = count - off - i;
        if (mirror < count && mirror != i) {
            output_buf[mirror] = mag * sin(angle) + mean;
        }
    }
}
