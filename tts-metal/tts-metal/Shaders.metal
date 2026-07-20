//
//  Shaders.metal
//  tts-metal
//
//  Metal compute kernels for Kitten TTS — port of src/shaders.ts WGSL.
//  All buffers are `device` storage; params use a small constant buffer per kernel.
//

#include <metal_stdlib>
using namespace metal;

// ── Layer Normalization ─────────────────────────────────────────────────────

struct LayerNormParams {
    uint batch_size;
    uint hidden_size;
    float eps;
};

kernel void layer_norm_kernel(device const float* input  [[buffer(0)]],
                              device const float* gamma [[buffer(1)]],
                              device const float* beta  [[buffer(2)]],
                              device float*       output [[buffer(3)]],
                              constant LayerNormParams& params [[buffer(4)]],
                              uint gid [[thread_position_in_grid]]) {
    if (gid >= params.batch_size) return;
    uint off = gid * params.hidden_size;
    float sum = 0.0;
    for (uint i = 0; i < params.hidden_size; ++i) sum += input[off + i];
    float mean = sum / float(params.hidden_size);
    float vsum = 0.0;
    for (uint i = 0; i < params.hidden_size; ++i) {
        float d = input[off + i] - mean;
        vsum += d * d;
    }
    float var = vsum / float(params.hidden_size);
    float inv_std = 1.0 / sqrt(var + params.eps);
    for (uint i = 0; i < params.hidden_size; ++i) {
        output[off + i] = (input[off + i] - mean) * inv_std * gamma[i] + beta[i];
    }
}

// ── Tiled Matrix Multiply ───────────────────────────────────────────────────

struct MatmulParams {
    uint M;
    uint K;
    uint N;
    uint use_bias;
};

kernel void matmul_kernel(device const float* A    [[buffer(0)]],
                          device const float* B    [[buffer(1)]],
                          device const float* bias [[buffer(2)]],
                          device float*       out  [[buffer(3)]],
                          constant MatmulParams& params [[buffer(4)]],
                          uint2 gid [[thread_position_in_grid]],
                          uint2 lid [[thread_position_in_threadgroup]]) {
    constexpr uint TILE = 16;
    threadgroup float tileA[256];
    threadgroup float tileB[256];

    uint row = gid.x;
    uint col = gid.y;
    uint lr = lid.x;
    uint lc = lid.y;
    float sum = 0.0;
    uint numTiles = (params.K + TILE - 1) / TILE;
    for (uint t = 0; t < numTiles; ++t) {
        uint aCol = t * TILE + lc;
        tileA[lr * TILE + lc] = (row < params.M && aCol < params.K) ? A[row * params.K + aCol] : 0.0;
        uint bRow = t * TILE + lr;
        tileB[lr * TILE + lc] = (bRow < params.K && col < params.N) ? B[bRow * params.N + col] : 0.0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < TILE; ++k) {
            sum += tileA[lr * TILE + k] * tileB[k * TILE + lc];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < params.M && col < params.N) {
        if (params.use_bias != 0u) sum += bias[col];
        out[row * params.N + col] = sum;
    }
}

// ── MatMul + GELU fused ─────────────────────────────────────────────────────

kernel void matmul_gelu_kernel(device const float* A    [[buffer(0)]],
                               device const float* B    [[buffer(1)]],
                               device const float* bias [[buffer(2)]],
                               device float*       out  [[buffer(3)]],
                               constant MatmulParams& params [[buffer(4)]],
                               uint2 gid [[thread_position_in_grid]],
                               uint2 lid [[thread_position_in_threadgroup]]) {
    constexpr uint TILE = 16;
    threadgroup float tileA[256];
    threadgroup float tileB[256];
    uint row = gid.x;
    uint col = gid.y;
    uint lr = lid.x;
    uint lc = lid.y;
    float sum = 0.0;
    uint numTiles = (params.K + TILE - 1) / TILE;
    for (uint t = 0; t < numTiles; ++t) {
        uint aCol = t * TILE + lc;
        tileA[lr * TILE + lc] = (row < params.M && aCol < params.K) ? A[row * params.K + aCol] : 0.0;
        uint bRow = t * TILE + lr;
        tileB[lr * TILE + lc] = (bRow < params.K && col < params.N) ? B[bRow * params.N + col] : 0.0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < TILE; ++k) {
            sum += tileA[lr * TILE + k] * tileB[k * TILE + lc];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < params.M && col < params.N) {
        sum += bias[col];
        float c = 0.7978845608;
        float x = sum;
        float inner = clamp(c * (x + 0.044715 * x * x * x), -44.0, 44.0);
        out[row * params.N + col] = 0.5 * x * (1.0 + tanh(inner));
    }
}

kernel void matmul_relu_kernel(device const float* A    [[buffer(0)]],
                               device const float* B    [[buffer(1)]],
                               device const float* bias [[buffer(2)]],
                               device float*       out  [[buffer(3)]],
                               constant MatmulParams& params [[buffer(4)]],
                               uint2 gid [[thread_position_in_grid]],
                               uint2 lid [[thread_position_in_threadgroup]]) {
    constexpr uint TILE = 16;
    threadgroup float tileA[256];
    threadgroup float tileB[256];
    uint row = gid.x;
    uint col = gid.y;
    uint lr = lid.x;
    uint lc = lid.y;
    float sum = 0.0;
    uint numTiles = (params.K + TILE - 1) / TILE;
    for (uint t = 0; t < numTiles; ++t) {
        uint aCol = t * TILE + lc;
        tileA[lr * TILE + lc] = (row < params.M && aCol < params.K) ? A[row * params.K + aCol] : 0.0;
        uint bRow = t * TILE + lr;
        tileB[lr * TILE + lc] = (bRow < params.K && col < params.N) ? B[bRow * params.N + col] : 0.0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < TILE; ++k) {
            sum += tileA[lr * TILE + k] * tileB[k * TILE + lc];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < params.M && col < params.N) {
        if (params.use_bias != 0u) sum += bias[col];
        out[row * params.N + col] = max(sum, 0.0f);
    }
}

// ── Conv1d ──────────────────────────────────────────────────────────────────

struct Conv1dParams {
    uint in_channels;
    uint out_channels;
    uint kernel_size;
    uint input_length;
    uint output_length;
    uint padding;
    uint stride;
    uint dilation;
    uint use_bias;
};

kernel void conv1d_kernel(device const float* input  [[buffer(0)]],
                          device const float* weight [[buffer(1)]],
                          device const float* bias   [[buffer(2)]],
                          device float*       output [[buffer(3)]],
                          constant Conv1dParams& params [[buffer(4)]],
                          uint gid [[thread_position_in_grid]]) {
    uint out_ch = gid / params.output_length;
    uint out_pos = gid % params.output_length;
    if (out_ch >= params.out_channels) return;
    float sum = 0.0;
    for (uint ic = 0; ic < params.in_channels; ++ic) {
        for (uint k = 0; k < params.kernel_size; ++k) {
            int in_pos_raw = int(out_pos * params.stride) + int(k * params.dilation) - int(params.padding);
            if (in_pos_raw >= 0 && uint(in_pos_raw) < params.input_length) {
                uint w_idx = out_ch * params.in_channels * params.kernel_size + ic * params.kernel_size + k;
                uint in_idx = ic * params.input_length + uint(in_pos_raw);
                sum += input[in_idx] * weight[w_idx];
            }
        }
    }
    if (params.use_bias != 0u) sum += bias[out_ch];
    output[out_ch * params.output_length + out_pos] = sum;
}

// ── Conv1d (tiled weight row) ───────────────────────────────────────────────

kernel void conv1d_tiled_kernel(device const float* input  [[buffer(0)]],
                                device const float* weight [[buffer(1)]],
                                device const float* bias   [[buffer(2)]],
                                device float*       output [[buffer(3)]],
                                constant Conv1dParams& params [[buffer(4)]],
                                uint2 wid [[threadgroup_position_in_grid]],
                                uint2 lid [[thread_position_in_threadgroup]]) {
    constexpr uint WGT = 256;
    threadgroup float w_tile[4096];
    uint out_ch = wid.y;
    uint row_size = params.in_channels * params.kernel_size;
    uint w_base = out_ch * row_size;
    for (uint i = lid.x; i < row_size; i += WGT) w_tile[i] = weight[w_base + i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint out_pos = wid.x * WGT + lid.x;
    if (out_pos >= params.output_length) return;
    float sum = 0.0;
    for (uint ic = 0; ic < params.in_channels; ++ic) {
        uint in_base = ic * params.input_length;
        uint wt_base = ic * params.kernel_size;
        for (uint k = 0; k < params.kernel_size; ++k) {
            int in_pos_raw = int(out_pos * params.stride) + int(k * params.dilation) - int(params.padding);
            if (in_pos_raw >= 0 && uint(in_pos_raw) < params.input_length) {
                sum += input[in_base + uint(in_pos_raw)] * w_tile[wt_base + k];
            }
        }
    }
    if (params.use_bias != 0u) sum += bias[out_ch];
    output[out_ch * params.output_length + out_pos] = sum;
}

// ── Element-wise activations ────────────────────────────────────────────────

struct SizeParams { uint size; };
struct SizeAlphaParams { uint size; float alpha; };


kernel void gelu_kernel(device const float* input  [[buffer(0)]],
                       device float*       output [[buffer(1)]],
                       constant SizeParams& params [[buffer(2)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid >= params.size) return;
    float x = input[gid];
    float c = 0.7978845608;
    float inner = clamp(c * (x + 0.044715 * x * x * x), -44.0, 44.0);
    output[gid] = 0.5 * x * (1.0 + tanh(inner));
}

kernel void tanh_kernel(device const float* input  [[buffer(0)]],
                       device float*       output [[buffer(1)]],
                       constant SizeParams& params [[buffer(2)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid >= params.size) return;
    output[gid] = tanh(input[gid]);
}

kernel void sigmoid_kernel(device const float* input  [[buffer(0)]],
                           device float*       output [[buffer(1)]],
                           constant SizeParams& params [[buffer(2)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= params.size) return;
    output[gid] = 1.0 / (1.0 + exp(-input[gid]));
}


// ── Softmax ─────────────────────────────────────────────────────────────────

struct SoftmaxParams {
    uint batch_size;
    uint dim_size;
};

kernel void softmax_kernel(device const float* input  [[buffer(0)]],
                           device float*       output [[buffer(1)]],
                           constant SoftmaxParams& params [[buffer(2)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= params.batch_size) return;
    uint off = gid * params.dim_size;
    float max_val = input[off];
    for (uint i = 1; i < params.dim_size; ++i) max_val = max(max_val, input[off + i]);
    float exp_sum = 0.0;
    for (uint i = 0; i < params.dim_size; ++i) {
        float e = exp(input[off + i] - max_val);
        output[off + i] = e;
        exp_sum += e;
    }
    for (uint i = 0; i < params.dim_size; ++i) output[off + i] /= exp_sum;
}

// ── Multi-Head Attention ────────────────────────────────────────────────────

struct MhaParams {
    uint seq_len;
    uint num_heads;
    uint head_dim;
    float scale;
};

kernel void mha_kernel(device const float* Q   [[buffer(0)]],
                      device const float* K   [[buffer(1)]],
                      device const float* V   [[buffer(2)]],
                      device float*       out [[buffer(3)]],
                      constant MhaParams& params [[buffer(4)]],
                      uint2 gid [[thread_position_in_grid]]) {
    uint dim_idx = gid.x;
    uint head_query = gid.y;
    uint head_idx = head_query / params.seq_len;
    uint q_pos = head_query % params.seq_len;
    if (dim_idx >= params.head_dim || head_idx >= params.num_heads) return;
    uint hd = params.head_dim, nh = params.num_heads, sl = params.seq_len;
    uint q_base = q_pos * nh * hd + head_idx * hd;
    float max_score = -1e10;
    for (uint k = 0; k < sl; ++k) {
        uint k_base = k * nh * hd + head_idx * hd;
        float score = 0.0;
        for (uint d = 0; d < hd; ++d) score += Q[q_base + d] * K[k_base + d];
        score *= params.scale;
        max_score = max(max_score, score);
    }
    float exp_sum = 0.0, weighted = 0.0;
    for (uint k = 0; k < sl; ++k) {
        uint k_base = k * nh * hd + head_idx * hd;
        float score = 0.0;
        for (uint d = 0; d < hd; ++d) score += Q[q_base + d] * K[k_base + d];
        score *= params.scale;
        float w = exp(score - max_score);
        exp_sum += w;
        uint v_base = k * nh * hd + head_idx * hd;
        weighted += w * V[v_base + dim_idx];
    }
    out[q_pos * nh * hd + head_idx * hd + dim_idx] = weighted / exp_sum;
}

// ── Add ─────────────────────────────────────────────────────────────────────

kernel void add_kernel(device const float* a   [[buffer(0)]],
                       device const float* b   [[buffer(1)]],
                       device float*       out [[buffer(2)]],
                       constant SizeParams& params [[buffer(3)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid >= params.size) return;
    out[gid] = a[gid] + b[gid];
}

// ── Scale ───────────────────────────────────────────────────────────────────

struct ScaleParams { uint size; uint _pad; float scale; };

kernel void scale_kernel(device const float* input [[buffer(0)]],
                         device float*       out   [[buffer(1)]],
                         constant ScaleParams& params [[buffer(2)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid >= params.size) return;
    out[gid] = input[gid] * params.scale;
}

// ── Fused add + scale: (a + b) * scale in one pass (one buffer round-trip) ────

kernel void add_scale_kernel(device const float* a   [[buffer(0)]],
                             device const float* b   [[buffer(1)]],
                             device float*       out [[buffer(2)]],
                             constant ScaleParams& params [[buffer(3)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= params.size) return;
    out[gid] = (a[gid] + b[gid]) * params.scale;
}

// ── Transpose 2D ────────────────────────────────────────────────────────────

struct TransposeParams { uint rows; uint cols; };

kernel void transpose_kernel(device const float* input [[buffer(0)]],
                             device float*       out   [[buffer(1)]],
                             constant TransposeParams& params [[buffer(2)]],
                             uint gid [[thread_position_in_grid]]) {
    uint total = params.rows * params.cols;
    if (gid >= total) return;
    uint row = gid / params.cols;
    uint col = gid % params.cols;
    out[col * params.rows + row] = input[gid];
}



// ── Register-blocked GEMM: out[M,N] = A[M,K] · B[K,N] (+bias per column) ─────
// 64×64 output tile per threadgroup, each of 256 threads computes a 4×4 subtile.
// Much higher arithmetic intensity than one-output-per-thread; used for the
// Mamba projections which dominate RE-USE runtime.
kernel void reuse_matmul_kernel(device const float* A    [[buffer(0)]],
                                device const float* B    [[buffer(1)]],
                                device const float* bias [[buffer(2)]],
                                device float*       out  [[buffer(3)]],
                                constant MatmulParams& p [[buffer(4)]],
                                uint2 tgid [[threadgroup_position_in_grid]],
                                uint2 lid2 [[thread_position_in_threadgroup]]) {
    constexpr uint BM = 64, BN = 64, BK = 16, TM = 4, TN = 4;
    threadgroup float As[BM * BK];
    threadgroup float Bs[BK * BN];
    uint lid = lid2.x;
    uint bm0 = tgid.y * BM;
    uint bn0 = tgid.x * BN;
    uint tRow = lid / 16;            // 0..15
    uint tCol = lid % 16;            // 0..15

    float acc[TM][TN];
    for (uint i = 0; i < TM; ++i) for (uint j = 0; j < TN; ++j) acc[i][j] = 0.0;

    uint M = p.M, K = p.K, N = p.N;
    for (uint k0 = 0; k0 < K; k0 += BK) {
        for (uint i = lid; i < BM * BK; i += 256) {
            uint r = i / BK, c = i % BK;
            uint gr = bm0 + r, gc = k0 + c;
            As[i] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0;
        }
        for (uint i = lid; i < BK * BN; i += 256) {
            uint r = i / BN, c = i % BN;
            uint gr = k0 + r, gc = bn0 + c;
            Bs[i] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < BK; ++kk) {
            float af[TM], bf[TN];
            for (uint i = 0; i < TM; ++i) af[i] = As[(tRow * TM + i) * BK + kk];
            for (uint j = 0; j < TN; ++j) bf[j] = Bs[kk * BN + tCol * TN + j];
            for (uint i = 0; i < TM; ++i)
                for (uint j = 0; j < TN; ++j)
                    acc[i][j] = fma(af[i], bf[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0; i < TM; ++i) {
        uint gr = bm0 + tRow * TM + i;
        if (gr >= M) continue;
        for (uint j = 0; j < TN; ++j) {
            uint gc = bn0 + tCol * TN + j;
            if (gc >= N) continue;
            float v = acc[i][j];
            if (p.use_bias != 0u) v += bias[gc];
            out[gr * N + gc] = v;
        }
    }
}

// ── Register-blocked GEMM with fused activation (GELU / ReLU) ────────────────
// Same 64×64 tiling as reuse_matmul_kernel; applies an activation at store so the
// ConvNeXt pointwise convs (pwconv1+GELU, attn FFN conv_1+ReLU) fuse the epilogue
// and avoid a separate elementwise pass. act: 0 = none, 1 = GELU(tanh), 2 = ReLU.
static inline float reuse_activate(float x, uint act) {
    if (act == 1u) {
        float inner = clamp(0.7978845608f * (x + 0.044715f * x * x * x), -44.0f, 44.0f);
        return 0.5f * x * (1.0f + tanh(inner));
    } else if (act == 2u) {
        return max(x, 0.0f);
    }
    return x;
}

template <uint ACT>
static inline void reuse_matmul_act_impl(device const float* A, device const float* B,
                                         device const float* bias, device float* out,
                                         constant MatmulParams& p, uint2 tgid, uint lid,
                                         threadgroup float* As, threadgroup float* Bs) {
    constexpr uint BM = 64, BN = 64, BK = 16, TM = 4, TN = 4;
    uint bm0 = tgid.y * BM;
    uint bn0 = tgid.x * BN;
    uint tRow = lid / 16;
    uint tCol = lid % 16;
    float acc[TM][TN];
    for (uint i = 0; i < TM; ++i) for (uint j = 0; j < TN; ++j) acc[i][j] = 0.0;
    uint M = p.M, K = p.K, N = p.N;
    for (uint k0 = 0; k0 < K; k0 += BK) {
        for (uint i = lid; i < BM * BK; i += 256) {
            uint r = i / BK, c = i % BK;
            uint gr = bm0 + r, gc = k0 + c;
            As[i] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0;
        }
        for (uint i = lid; i < BK * BN; i += 256) {
            uint r = i / BN, c = i % BN;
            uint gr = k0 + r, gc = bn0 + c;
            Bs[i] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < BK; ++kk) {
            float af[TM], bf[TN];
            for (uint i = 0; i < TM; ++i) af[i] = As[(tRow * TM + i) * BK + kk];
            for (uint j = 0; j < TN; ++j) bf[j] = Bs[kk * BN + tCol * TN + j];
            for (uint i = 0; i < TM; ++i)
                for (uint j = 0; j < TN; ++j)
                    acc[i][j] = fma(af[i], bf[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0; i < TM; ++i) {
        uint gr = bm0 + tRow * TM + i;
        if (gr >= M) continue;
        for (uint j = 0; j < TN; ++j) {
            uint gc = bn0 + tCol * TN + j;
            if (gc >= N) continue;
            float v = acc[i][j];
            if (p.use_bias != 0u) v += bias[gc];
            out[gr * N + gc] = reuse_activate(v, ACT);
        }
    }
}

kernel void reuse_matmul_gelu_kernel(device const float* A    [[buffer(0)]],
                                     device const float* B    [[buffer(1)]],
                                     device const float* bias [[buffer(2)]],
                                     device float*       out  [[buffer(3)]],
                                     constant MatmulParams& p [[buffer(4)]],
                                     uint2 tgid [[threadgroup_position_in_grid]],
                                     uint2 lid2 [[thread_position_in_threadgroup]]) {
    threadgroup float As[64 * 16];
    threadgroup float Bs[16 * 64];
    reuse_matmul_act_impl<1u>(A, B, bias, out, p, tgid, lid2.x, As, Bs);
}

kernel void reuse_matmul_relu_kernel(device const float* A    [[buffer(0)]],
                                     device const float* B    [[buffer(1)]],
                                     device const float* bias [[buffer(2)]],
                                     device float*       out  [[buffer(3)]],
                                     constant MatmulParams& p [[buffer(4)]],
                                     uint2 tgid [[threadgroup_position_in_grid]],
                                     uint2 lid2 [[thread_position_in_threadgroup]]) {
    threadgroup float As[64 * 16];
    threadgroup float Bs[16 * 64];
    reuse_matmul_act_impl<2u>(A, B, bias, out, p, tgid, lid2.x, As, Bs);
}

// ── Tiled 2D convolution: one threadgroup per output channel, weights cached ──
// in threadgroup memory and reused across all spatial positions. Used when the
// per-channel weight row (in_ch*kh*kw) fits the tile; the dense blocks (256*3*3
// = 2304) dominate RE-USE runtime, so this is the hot path.

// ── PReLU (per-channel slope) ───────────────────────────────────────────────
struct ReuseChanLenParams { uint channels; uint length; uint slope_shared; };

kernel void reuse_prelu_kernel(device const float* input  [[buffer(0)]],
                               device const float* slope  [[buffer(1)]],
                               device float*       output [[buffer(2)]],
                               constant ReuseChanLenParams& p [[buffer(3)]],
                               uint gid [[thread_position_in_grid]]) {
    uint total = p.channels * p.length;
    if (gid >= total) return;
    uint ch = p.slope_shared ? 0 : (gid / p.length);
    float x = input[gid];
    output[gid] = (x >= 0.0) ? x : slope[ch] * x;
}


// ── Depthwise Conv1d (per-channel, symmetric padding) ───────────────────────
struct LavaDwParams { uint channels; uint length; uint ksize; uint pad; };

kernel void lava_dwconv1d_kernel(device const float* input  [[buffer(0)]],  // [C, T]
                                 device const float* weight [[buffer(1)]],  // [C, K]
                                 device const float* bias   [[buffer(2)]],  // [C]
                                 device float*       output [[buffer(3)]],  // [C, T]
                                 constant LavaDwParams& p [[buffer(4)]],
                                 uint gid [[thread_position_in_grid]]) {
    uint total = p.channels * p.length;
    if (gid >= total) return;
    uint c = gid / p.length;
    uint t = gid % p.length;
    uint in_base = c * p.length;
    uint w_base = c * p.ksize;
    float sum = bias[c];
    for (uint k = 0; k < p.ksize; ++k) {
        int ip = int(t) + int(k) - int(p.pad);
        if (ip >= 0 && uint(ip) < p.length) {
            sum = fma(input[in_base + uint(ip)], weight[w_base + k], sum);
        }
    }
    output[gid] = sum;
}

// ── ConvNeXt tail: out[d,t] = residual[d,t] + gamma[d] * h[t,d] ──────────────
// h is [T, dim] (row-major), residual and output are [dim, T] (channel-major).
struct LavaGammaParams { uint dim; uint length; };

kernel void lava_gamma_residual_kernel(device const float* h        [[buffer(0)]],  // [T, dim]
                                       device const float* residual [[buffer(1)]],  // [dim, T]
                                       device const float* gamma    [[buffer(2)]],  // [dim]
                                       device float*       output   [[buffer(3)]],  // [dim, T]
                                       constant LavaGammaParams& p [[buffer(4)]],
                                       uint gid [[thread_position_in_grid]]) {
    uint total = p.dim * p.length;
    if (gid >= total) return;
    uint d = gid / p.length;
    uint t = gid % p.length;
    output[gid] = residual[gid] + gamma[d] * h[t * p.dim + d];
}

// ═══════════════════════════════════════════════════════════════════════════
//  Supertonic 3 kernels
//  Flow-matching latent TTS. Feature maps are channel-major [C, T] unless noted
//  row-major [T, C]. ConvNeXt reuses lava_dwconv1d / lava_gamma_residual / matmul
//  / layer_norm; the pieces below are Supertonic-specific.
// ═══════════════════════════════════════════════════════════════════════════

struct StSizeParams { uint size; };

// erf approximation (Abramowitz & Stegun 7.1.26), max abs error ~1.5e-7.
static inline float st_erf(float x) {
    float s = (x < 0.0f) ? -1.0f : 1.0f;
    float ax = fabs(x);
    float t = 1.0f / (1.0f + 0.3275911f * ax);
    float y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t
                       - 0.284496736f) * t + 0.254829592f) * t * exp(-ax * ax);
    return s * y;
}

// ── Exact GELU (erf form, matches ONNX Gelu/Erf) ────────────────────────────
kernel void st_gelu_erf_kernel(device const float* input  [[buffer(0)]],
                               device float*       output [[buffer(1)]],
                               constant StSizeParams& p [[buffer(2)]],
                               uint gid [[thread_position_in_grid]]) {
    if (gid >= p.size) return;
    float x = input[gid];
    output[gid] = 0.5f * x * (1.0f + st_erf(x * 0.70710678118f));
}

// ── Softplus ────────────────────────────────────────────────────────────────
kernel void st_softplus_kernel(device const float* input  [[buffer(0)]],
                               device float*       output [[buffer(1)]],
                               constant StSizeParams& p [[buffer(2)]],
                               uint gid [[thread_position_in_grid]]) {
    if (gid >= p.size) return;
    float x = input[gid];
    output[gid] = (x > 20.0f) ? x : log(1.0f + exp(x));
}

// ── FiLM conditioning (channel-major): out[c,t] = x[c,t]*(1+scale[c]) + shift[c]
// scale/shift are length-C vectors (a linear projection of time+style, broadcast
// across T). Set add_one=0 to skip the (1+·) affine offset.
struct StFilmParams { uint channels; uint length; uint add_one; };

kernel void st_film_kernel(device const float* x      [[buffer(0)]],
                           device const float* scale  [[buffer(1)]],  // [C]
                           device const float* shift  [[buffer(2)]],  // [C]
                           device float*       out    [[buffer(3)]],  // [C, T]
                           constant StFilmParams& p [[buffer(4)]],
                           uint gid [[thread_position_in_grid]]) {
    uint total = p.channels * p.length;
    if (gid >= total) return;
    uint c = gid / p.length;
    float s = scale[c] + (p.add_one != 0u ? 1.0f : 0.0f);
    out[gid] = x[gid] * s + shift[c];
}

// ── Broadcast-add a length-C vector across T (channel-major) ─────────────────
kernel void st_add_col_kernel(device const float* x   [[buffer(0)]],  // [C, T]
                              device const float* vec [[buffer(1)]],  // [C]
                              device float*       out [[buffer(2)]],  // [C, T]
                              constant StFilmParams& p [[buffer(3)]],
                              uint gid [[thread_position_in_grid]]) {
    uint total = p.channels * p.length;
    if (gid >= total) return;
    out[gid] = x[gid] + vec[gid / p.length];
}

// ── Rotary position embedding, applied to a [L, H*D] row-major tensor ────────
// Rotates each head's D dims in (i, i+D/2) pairs by angle pos*theta_i.
// theta_i = base^(-2i/D). One thread per (l, head, i<D/2).
struct StRopeParams { uint seq_len; uint num_heads; uint head_dim; float base; };

kernel void st_rope_kernel(device const float* x   [[buffer(0)]],   // [L, H*D]
                           device float*       out [[buffer(1)]],   // [L, H*D]
                           constant StRopeParams& p [[buffer(2)]],
                           uint gid [[thread_position_in_grid]]) {
    uint hf = p.head_dim / 2u;
    uint total = p.seq_len * p.num_heads * hf;
    if (gid >= total) return;
    uint i = gid % hf;
    uint h = (gid / hf) % p.num_heads;
    uint l = gid / (hf * p.num_heads);
    float freq = pow(p.base, -2.0f * float(i) / float(p.head_dim));
    float ang = float(l) * freq;
    float c = cos(ang), s = sin(ang);
    uint base = l * p.num_heads * p.head_dim + h * p.head_dim;
    float x0 = x[base + i];
    float x1 = x[base + i + hf];
    out[base + i]      = x0 * c - x1 * s;
    out[base + i + hf] = x0 * s + x1 * c;
}

// ── Masked multi-head attention (self or cross) ──────────────────────────────
// Q:[Lq, H*D], K/V:[Lkv, H*D] row-major, interleaved heads. key_mask:[Lkv]
// (1 = keep, 0 = masked). One thread per (q_pos, head, d). scale premultiplied.
struct StAttnParams { uint lq; uint lkv; uint num_heads; uint head_dim; float scale; };

kernel void st_mha_kernel(device const float* Q        [[buffer(0)]],
                          device const float* K        [[buffer(1)]],
                          device const float* V        [[buffer(2)]],
                          device const float* key_mask [[buffer(3)]],  // [Lkv] or null
                          device float*       out      [[buffer(4)]],
                          constant StAttnParams& p [[buffer(5)]],
                          uint gid [[thread_position_in_grid]]) {
    uint d = gid % p.head_dim;
    uint h = (gid / p.head_dim) % p.num_heads;
    uint q = gid / (p.head_dim * p.num_heads);
    if (q >= p.lq) return;
    uint HD = p.num_heads * p.head_dim;
    uint q_base = q * HD + h * p.head_dim;
    float mx = -1e30f;
    for (uint k = 0; k < p.lkv; ++k) {
        if (key_mask && key_mask[k] == 0.0f) continue;
        uint k_base = k * HD + h * p.head_dim;
        float sc = 0.0f;
        for (uint e = 0; e < p.head_dim; ++e) sc += Q[q_base + e] * K[k_base + e];
        sc *= p.scale;
        mx = max(mx, sc);
    }
    float denom = 0.0f, acc = 0.0f;
    for (uint k = 0; k < p.lkv; ++k) {
        if (key_mask && key_mask[k] == 0.0f) continue;
        uint k_base = k * HD + h * p.head_dim;
        float sc = 0.0f;
        for (uint e = 0; e < p.head_dim; ++e) sc += Q[q_base + e] * K[k_base + e];
        sc *= p.scale;
        float w = exp(sc - mx);
        denom += w;
        acc += w * V[k_base + d];
    }
    out[q * HD + h * p.head_dim + d] = (denom > 0.0f) ? acc / denom : 0.0f;
}

// ── VITS relative-position attention (self-attn with windowed rel-pos keys) ──
// Q/K/V:[L, H*D]. emb_rel_k/emb_rel_v: [1, 2*window+1, D] — SHARED across heads.
// Following VITS: relative positions with |k-q| > window contribute ZERO (the
// reference zero-pads the (2w+1) embeddings out to 2L-1). scale = 1/sqrt(D) is
// applied to both content and relative logits (q is pre-scaled in the reference).
struct StRelAttnParams { uint seq_len; uint num_heads; uint head_dim; uint window; float scale; };

kernel void st_rel_attn_kernel(device const float* Q        [[buffer(0)]],
                               device const float* K        [[buffer(1)]],
                               device const float* V        [[buffer(2)]],
                               device const float* rel_k    [[buffer(3)]],  // [1,2w+1,D] shared
                               device const float* rel_v    [[buffer(4)]],  // [1,2w+1,D] shared
                               device const float* key_mask [[buffer(5)]],  // [L] or null
                               device float*       out      [[buffer(6)]],
                               constant StRelAttnParams& p [[buffer(7)]],
                               uint gid [[thread_position_in_grid]]) {
    uint d = gid % p.head_dim;
    uint h = (gid / p.head_dim) % p.num_heads;
    uint q = gid / (p.head_dim * p.num_heads);
    if (q >= p.seq_len) return;
    uint HD = p.num_heads * p.head_dim;
    uint q_base = q * HD + h * p.head_dim;
    int win = int(p.window);
    float mx = -1e30f;
    for (uint k = 0; k < p.seq_len; ++k) {
        if (key_mask && key_mask[k] == 0.0f) continue;
        uint k_base = k * HD + h * p.head_dim;
        float sc = 0.0f;
        for (uint e = 0; e < p.head_dim; ++e) sc += Q[q_base + e] * K[k_base + e];
        int rel = int(k) - int(q);
        if (rel >= -win && rel <= win) {          // in-window: add relative-key logit
            uint ridx = uint(rel + win);
            float rsc = 0.0f;
            for (uint e = 0; e < p.head_dim; ++e)
                rsc += Q[q_base + e] * rel_k[ridx * p.head_dim + e];
            sc += rsc;
        }
        sc *= p.scale;
        mx = max(mx, sc);
    }
    float denom = 0.0f, acc = 0.0f;
    for (uint k = 0; k < p.seq_len; ++k) {
        if (key_mask && key_mask[k] == 0.0f) continue;
        uint k_base = k * HD + h * p.head_dim;
        float sc = 0.0f;
        for (uint e = 0; e < p.head_dim; ++e) sc += Q[q_base + e] * K[k_base + e];
        int rel = int(k) - int(q);
        float relV = 0.0f;
        if (rel >= -win && rel <= win) {
            uint ridx = uint(rel + win);
            float rsc = 0.0f;
            for (uint e = 0; e < p.head_dim; ++e)
                rsc += Q[q_base + e] * rel_k[ridx * p.head_dim + e];
            sc += rsc;
            relV = rel_v[ridx * p.head_dim + d];
        }
        float w = exp(sc * p.scale - mx);
        denom += w;
        acc += w * (V[k_base + d] + relV);
    }
    out[q * HD + h * p.head_dim + d] = (denom > 0.0f) ? acc / denom : 0.0f;
}

// ── Causal, edge-padded 1D conv (full) — AE decoder convs use pad_left=dil*(k-1),
// pad_right=0, mode='edge'. Out-of-range left indices clamp to sample 0. ────────
struct StCausalConvParams { uint in_ch; uint out_ch; uint ksz; uint length; uint dilation; uint use_bias; };

kernel void st_causal_conv1d_kernel(device const float* input  [[buffer(0)]],  // [in_ch, L]
                                    device const float* weight [[buffer(1)]],  // [out_ch, in_ch, k]
                                    device const float* bias   [[buffer(2)]],  // [out_ch]
                                    device float*       out    [[buffer(3)]],  // [out_ch, L]
                                    constant StCausalConvParams& p [[buffer(4)]],
                                    uint gid [[thread_position_in_grid]]) {
    uint total = p.out_ch * p.length;
    if (gid >= total) return;
    uint oc = gid / p.length;
    uint t  = gid % p.length;
    uint padL = p.dilation * (p.ksz - 1u);
    float sum = (p.use_bias != 0u) ? bias[oc] : 0.0f;
    for (uint ic = 0; ic < p.in_ch; ++ic) {
        uint in_base = ic * p.length;
        uint w_base = (oc * p.in_ch + ic) * p.ksz;
        for (uint k = 0; k < p.ksz; ++k) {
            int ip = int(t) + int(k * p.dilation) - int(padL);
            uint idx = ip < 0 ? 0u : uint(ip);        // edge (replicate) padding on the left
            sum += input[in_base + idx] * weight[w_base + k];
        }
    }
    out[gid] = sum;
}

// ── Causal, edge-padded depthwise 1D conv (AE decoder ConvNeXt dwconv) ────────
kernel void st_causal_dwconv1d_kernel(device const float* input  [[buffer(0)]],  // [C, L]
                                      device const float* weight [[buffer(1)]],  // [C, 1, k]
                                      device const float* bias   [[buffer(2)]],  // [C]
                                      device float*       out    [[buffer(3)]],  // [C, L]
                                      constant StCausalConvParams& p [[buffer(4)]],
                                      uint gid [[thread_position_in_grid]]) {
    uint total = p.out_ch * p.length;   // out_ch == channels for depthwise
    if (gid >= total) return;
    uint c = gid / p.length;
    uint t = gid % p.length;
    uint padL = p.dilation * (p.ksz - 1u);
    uint in_base = c * p.length;
    uint w_base = c * p.ksz;
    float sum = (p.use_bias != 0u) ? bias[c] : 0.0f;
    for (uint k = 0; k < p.ksz; ++k) {
        int ip = int(t) + int(k * p.dilation) - int(padL);
        uint idx = ip < 0 ? 0u : uint(ip);
        sum += input[in_base + idx] * weight[w_base + k];
    }
    out[gid] = sum;
}

// ── Sinusoidal timestep embedding: emb[t, 2i]=sin(t*f_i), [t,2i+1]=cos ───────
// One thread per (row, i<dim/2). freqs f_i = exp(-i/(half-1) * log(max_period)).
struct StSinusoidParams { uint rows; uint dim; float max_period; };

kernel void st_sinusoid_kernel(device const float* t    [[buffer(0)]],  // [rows]
                               device float*       out  [[buffer(1)]],  // [rows, dim]
                               constant StSinusoidParams& p [[buffer(2)]],
                               uint gid [[thread_position_in_grid]]) {
    uint hf = p.dim / 2u;
    uint total = p.rows * hf;
    if (gid >= total) return;
    uint i = gid % hf;
    uint r = gid / hf;
    float denom = (hf > 1u) ? float(hf - 1u) : 1.0f;
    float freq = exp(-log(p.max_period) * float(i) / denom);
    float ang = t[r] * freq;
    out[r * p.dim + i]      = sin(ang);
    out[r * p.dim + hf + i] = cos(ang);
}

// ── PReLU/tanh helpers already exist (reuse_prelu_kernel, tanh_kernel) ───────

// ── Dilated Depthwise Conv1d for Supertonic ──────────────────────────────
struct StDwParams { uint channels; uint length; uint ksize; uint pad; uint dilation; };

kernel void st_dwconv1d_kernel(device const float* input  [[buffer(0)]],  // [C, T]
                               device const float* weight [[buffer(1)]],  // [C, K]
                               device const float* bias   [[buffer(2)]],  // [C]
                               device float*       output [[buffer(3)]],  // [C, T]
                               constant StDwParams& p [[buffer(4)]],
                               uint gid [[thread_position_in_grid]]) {
    uint total = p.channels * p.length;
    if (gid >= total) return;
    uint c = gid / p.length;
    uint t = gid % p.length;
    uint in_base = c * p.length;
    uint w_base = c * p.ksize;
    float sum = bias[c];
    for (uint k = 0; k < p.ksize; ++k) {
        int ip = int(t) + int(k) * int(p.dilation) - int(p.pad);
        if (ip >= 0 && uint(ip) < p.length) {
            sum = fma(input[in_base + uint(ip)], weight[w_base + k], sum);
        }
    }
    output[gid] = sum;
}

// ── Symmetric edge(replicate)-padded depthwise conv (vector_field convnext) ───
// Same as st_dwconv1d but out-of-range indices clamp to the boundary sample
// (ONNX Pad mode='edge', pads=[dil*(k-1)/2 each side]).
kernel void st_dwconv1d_edge_kernel(device const float* input  [[buffer(0)]],  // [C, T]
                                    device const float* weight [[buffer(1)]],  // [C, K]
                                    device const float* bias   [[buffer(2)]],  // [C]
                                    device float*       output [[buffer(3)]],  // [C, T]
                                    constant StDwParams& p [[buffer(4)]],
                                    uint gid [[thread_position_in_grid]]) {
    uint total = p.channels * p.length;
    if (gid >= total) return;
    uint c = gid / p.length;
    uint t = gid % p.length;
    uint in_base = c * p.length;
    uint w_base = c * p.ksize;
    float sum = bias[c];
    for (uint k = 0; k < p.ksize; ++k) {
        int ip = int(t) + int(k) * int(p.dilation) - int(p.pad);
        if (ip < 0) ip = 0;
        if (ip >= int(p.length)) ip = int(p.length) - 1;
        sum = fma(input[in_base + uint(ip)], weight[w_base + k], sum);
    }
    output[gid] = sum;
}
