//
//  Shaders.metal
//  tts-metal
//
//  Metal compute kernels for Kitten TTS — port of src/shaders.ts WGSL.
//  All buffers are `device` storage; params use a small constant buffer per kernel.
//

#include <metal_stdlib>
using namespace metal;

// ── Embedding Lookup ────────────────────────────────────────────────────────

struct EmbeddingParams {
    uint seq_len;
    uint embed_dim;
    uint vocab_size;
};

kernel void embedding_kernel(device const float*  embeddings [[buffer(0)]],
                             device const int32_t* input_ids [[buffer(1)]],
                             device float*       output    [[buffer(2)]],
                             constant EmbeddingParams& params [[buffer(3)]],
                             uint gid [[thread_position_in_grid]]) {
    uint seq_idx = gid / params.embed_dim;
    uint dim_idx = gid % params.embed_dim;
    if (seq_idx >= params.seq_len) return;
    uint token_id = uint(input_ids[seq_idx]);
    uint off = token_id * params.embed_dim + dim_idx;
    output[gid] = embeddings[off];
}

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

// ── Instance Normalization (one WG per channel) ─────────────────────────────

struct InstanceNormParams {
    uint channels;
    uint length;
    float eps;
};

kernel void instance_norm_kernel(device const float* input  [[buffer(0)]],
                                 device float*       output [[buffer(1)]],
                                 constant InstanceNormParams& params [[buffer(2)]],
                                 uint2 wid [[threadgroup_position_in_grid]],
                                 uint2 lid [[thread_position_in_threadgroup]]) {
    constexpr uint WGT = 256;
    threadgroup float red[256];
    threadgroup float sh_mean;
    threadgroup float sh_inv_std;
    uint ch = wid.x;
    bool active = ch < params.channels;
    uint tid = lid.x;
    uint L = params.length;
    uint base = active ? ch * L : 0;
    float s = 0.0;
    for (uint i = tid; i < L; i += WGT) s += input[base + i];
    red[tid] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = WGT / 2; stride > 0; stride >>= 1) {
        if (tid < stride) red[tid] += red[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) sh_mean = red[0] / float(L);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean = sh_mean;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float vs = 0.0;
    for (uint i = tid; i < L; i += WGT) {
        float d = input[base + i] - mean;
        vs += d * d;
    }
    red[tid] = vs;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = WGT / 2; stride > 0; stride >>= 1) {
        if (tid < stride) red[tid] += red[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) sh_inv_std = 1.0 / sqrt(red[0] / float(L) + params.eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_std = sh_inv_std;
    for (uint i = tid; i < L; i += WGT) {
        output[base + i] = (input[base + i] - mean) * inv_std;
    }
}

// ── AdaIN (channels-first) ──────────────────────────────────────────────────

struct AdainParams {
    uint channels;
    uint length;
};

kernel void adain_kernel(device const float* normed   [[buffer(0)]],
                         device const float* style_fc [[buffer(1)]],
                         device float*       output    [[buffer(2)]],
                         constant AdainParams& params [[buffer(3)]],
                         uint gid [[thread_position_in_grid]]) {
    uint ch = gid / params.length;
    uint pos = gid % params.length;
    if (ch >= params.channels) return;
    float scale = style_fc[ch];
    float bias = style_fc[params.channels + ch];
    output[gid] = normed[gid] * (scale + 1.0) + bias;
}

// ── AdaIN (row-major) ────────────────────────────────────────────────────────

struct AdainRowMajorParams {
    uint channels;
    uint total;
};

kernel void adain_row_major_kernel(device const float* normed   [[buffer(0)]],
                                   device const float* style_fc [[buffer(1)]],
                                   device float*       output   [[buffer(2)]],
                                   constant AdainRowMajorParams& params [[buffer(3)]],
                                   uint gid [[thread_position_in_grid]]) {
    if (gid >= params.total) return;
    uint ch = gid % params.channels;
    float scale = style_fc[ch];
    float bias = style_fc[params.channels + ch];
    output[gid] = normed[gid] * (scale + 1.0) + bias;
}

// ── Snake activation ────────────────────────────────────────────────────────

kernel void snake_kernel(device const float* input  [[buffer(0)]],
                         device const float* alpha  [[buffer(1)]],
                         device float*       output [[buffer(2)]],
                         constant AdainParams& params [[buffer(3)]],
                         uint gid [[thread_position_in_grid]]) {
    uint ch = gid / params.length;
    if (ch >= params.channels) return;
    float x = input[gid];
    float a = alpha[ch];
    float sa = sin(a * x);
    output[gid] = x + sa * sa / a;
}

// ── Element-wise activations ────────────────────────────────────────────────

struct SizeParams { uint size; };
struct SizeAlphaParams { uint size; float alpha; };

kernel void leaky_relu_kernel(device const float* input  [[buffer(0)]],
                              device float*       output [[buffer(1)]],
                              constant SizeAlphaParams& params [[buffer(2)]],
                              uint gid [[thread_position_in_grid]]) {
    if (gid >= params.size) return;
    float x = input[gid];
    output[gid] = (x >= 0.0) ? x : params.alpha * x;
}

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

// ── ConvTranspose1d (HiFi-GAN upsampling) ──────────────────────────────────

struct ConvTranspose1dParams {
    uint in_channels;
    uint out_channels;
    uint kernel_size;
    uint input_length;
    uint output_length;
    uint stride;
    uint padding;
    uint use_bias;
};

kernel void conv_transpose1d_kernel(device const float* input  [[buffer(0)]],
                                    device const float* weight [[buffer(1)]],
                                    device const float* bias   [[buffer(2)]],
                                    device float*       output [[buffer(3)]],
                                    constant ConvTranspose1dParams& params [[buffer(4)]],
                                    uint gid [[thread_position_in_grid]]) {
    uint out_ch = gid / params.output_length;
    uint out_pos = gid % params.output_length;
    if (out_ch >= params.out_channels) return;
    uint stride = params.stride;
    uint K = params.kernel_size;
    uint L_in = params.input_length;
    uint C_in = params.in_channels;
    uint op = out_pos + params.padding;
    uint k0 = op % stride;
    float sum = 0.0;
    for (uint k = k0; k < K; k += stride) {
        int num = int(op) - int(k);
        if (num < 0) break;
        uint in_pos = uint(num) / stride;
        if (in_pos < L_in) {
            uint w_col = out_ch * K + k;
            for (uint ic = 0; ic < C_in; ++ic) {
                sum += input[ic * L_in + in_pos] * weight[ic * params.out_channels * K + w_col];
            }
        }
    }
    if (params.use_bias != 0u) sum += bias[out_ch];
    output[out_ch * params.output_length + out_pos] = sum;
}

// ── Depthwise ConvTranspose1d ───────────────────────────────────────────────

struct DepthwiseConvTParams {
    uint channels;
    uint kernel_size;
    uint input_length;
    uint output_length;
    uint stride;
    uint padding;
};

kernel void depthwise_conv_transpose1d_kernel(device const float* input  [[buffer(0)]],
                                              device const float* weight [[buffer(1)]],
                                              device float*       output [[buffer(2)]],
                                              constant DepthwiseConvTParams& params [[buffer(3)]],
                                              uint gid [[thread_position_in_grid]]) {
    uint ch = gid / params.output_length;
    uint out_pos = gid % params.output_length;
    if (ch >= params.channels) return;
    float sum = 0.0;
    for (uint k = 0; k < params.kernel_size; ++k) {
        int num = int(out_pos) + int(params.padding) - int(k);
        if (num >= 0 && uint(num) % params.stride == 0u) {
            uint in_pos = uint(num) / params.stride;
            if (in_pos < params.input_length) {
                uint w_idx = ch * params.kernel_size + k;
                uint in_idx = ch * params.input_length + in_pos;
                sum += input[in_idx] * weight[w_idx];
            }
        }
    }
    output[ch * params.output_length + out_pos] = sum;
}

// ── Resize 1D (nearest) ─────────────────────────────────────────────────────

struct Resize1dParams {
    uint channels;
    uint input_length;
    uint output_length;
};

kernel void resize1d_kernel(device const float* input  [[buffer(0)]],
                            device float*       output [[buffer(1)]],
                            constant Resize1dParams& params [[buffer(2)]],
                            uint gid [[thread_position_in_grid]]) {
    uint ch = gid / params.output_length;
    uint out_pos = gid % params.output_length;
    if (ch >= params.channels) return;
    uint in_pos = out_pos * params.input_length / params.output_length;
    output[ch * params.output_length + out_pos] = input[ch * params.input_length + in_pos];
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

// ── Concat channels-first ───────────────────────────────────────────────────

struct ConcatChannelsParams {
    uint channels_a;
    uint channels_b;
    uint length;
};

kernel void concat_channels_kernel(device const float* a   [[buffer(0)]],
                                   device const float* b   [[buffer(1)]],
                                   device float*       out [[buffer(2)]],
                                   constant ConcatChannelsParams& params [[buffer(3)]],
                                   uint gid [[thread_position_in_grid]]) {
    uint total = (params.channels_a + params.channels_b) * params.length;
    if (gid >= total) return;
    uint ch = gid / params.length;
    uint pos = gid % params.length;
    if (ch < params.channels_a) out[gid] = a[ch * params.length + pos];
    else out[gid] = b[(ch - params.channels_a) * params.length + pos];
}

// ── Concat broadcast (A[rows,cols_a] + B[cols_b]) ───────────────────────────

struct ConcatBroadcastParams {
    uint rows;
    uint cols_a;
    uint cols_b;
};

kernel void concat_broadcast_kernel(device const float* a   [[buffer(0)]],
                                    device const float* b   [[buffer(1)]],
                                    device float*       out [[buffer(2)]],
                                    constant ConcatBroadcastParams& params [[buffer(3)]],
                                    uint gid [[thread_position_in_grid]]) {
    uint total_cols = params.cols_a + params.cols_b;
    uint total = params.rows * total_cols;
    if (gid >= total) return;
    uint row = gid / total_cols;
    uint col = gid % total_cols;
    if (col < params.cols_a) out[gid] = a[row * params.cols_a + col];
    else out[gid] = b[col - params.cols_a];
}

// ── Reflection Pad 1D ───────────────────────────────────────────────────────

struct ReflectionPadParams {
    uint channels;
    uint input_length;
    uint pad_left;
    uint pad_right;
};

kernel void reflection_pad1d_kernel(device const float* input [[buffer(0)]],
                                    device float*       out   [[buffer(1)]],
                                    constant ReflectionPadParams& params [[buffer(2)]],
                                    uint gid [[thread_position_in_grid]]) {
    uint out_length = params.input_length + params.pad_left + params.pad_right;
    uint ch = gid / out_length;
    uint out_pos = gid % out_length;
    if (ch >= params.channels) return;
    uint in_pos;
    if (out_pos < params.pad_left) {
        in_pos = params.pad_left - out_pos;
    } else if (out_pos >= params.pad_left + params.input_length) {
        uint overshoot = out_pos - params.pad_left - params.input_length;
        in_pos = params.input_length - 2 - overshoot;
    } else {
        in_pos = out_pos - params.pad_left;
    }
    out[ch * out_length + out_pos] = input[ch * params.input_length + in_pos];
}

// ── Alpha-weighted residual ─────────────────────────────────────────────────

kernel void alpha_residual_kernel(device const float* current   [[buffer(0)]],
                                  device const float* residual  [[buffer(1)]],
                                  device const float* alpha     [[buffer(2)]],
                                  device float*       out        [[buffer(3)]],
                                  constant AdainParams& params [[buffer(4)]],
                                  uint gid [[thread_position_in_grid]]) {
    uint ch = gid / params.length;
    if (ch >= params.channels) return;
    out[gid] = current[gid] + alpha[ch] * residual[gid];
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

// ── Bidirectional LSTM ──────────────────────────────────────────────────────

struct LstmParams {
    uint seq_len;
    uint input_size;
    uint hidden_size;
    uint num_directions;
};

kernel void lstm_kernel(device const float* input   [[buffer(0)]],
                        device const float* W       [[buffer(1)]],
                        device const float* R       [[buffer(2)]],
                        device const float* bias    [[buffer(3)]],
                        device float*       output  [[buffer(4)]],
                        constant LstmParams& params [[buffer(5)]],
                        uint2 gid [[thread_position_in_grid]]) {
    uint h_idx = gid.x;
    uint dir = gid.y;
    bool is_valid = (h_idx < params.hidden_size) && (dir < params.num_directions);
    uint H = params.hidden_size;
    uint H4 = H * 4u;
    uint IS = params.input_size;
    uint SL = params.seq_len;
    uint safe_h = is_valid ? h_idx : 0u;
    uint safe_dir = is_valid ? dir : 0u;
    uint gate_i = safe_h;
    uint gate_o = H + safe_h;
    uint gate_f = 2u * H + safe_h;
    uint gate_c = 3u * H + safe_h;
    uint bias_base = safe_dir * 8u * H;
    float b_wi = 0, b_wo = 0, b_wf = 0, b_wc = 0;
    float b_ri = 0, b_ro = 0, b_rf = 0, b_rc = 0;
    if (is_valid) {
        b_wi = bias[bias_base + safe_h];
        b_wo = bias[bias_base + H + safe_h];
        b_wf = bias[bias_base + 2u * H + safe_h];
        b_wc = bias[bias_base + 3u * H + safe_h];
        b_ri = bias[bias_base + 4u * H + safe_h];
        b_ro = bias[bias_base + 5u * H + safe_h];
        b_rf = bias[bias_base + 6u * H + safe_h];
        b_rc = bias[bias_base + 7u * H + safe_h];
    }
    float h_val = 0.0, c_val = 0.0;
    uint w_base = safe_dir * IS * H4;
    uint r_base = safe_dir * H * H4;
    for (uint step = 0; step < SL; ++step) {
        if (is_valid) {
            uint t = (safe_dir == 0u) ? step : (SL - 1u - step);
            float gi = b_wi + b_ri, go = b_wo + b_ro, gf = b_wf + b_rf, gc = b_wc + b_rc;
            for (uint j = 0; j < IS; ++j) {
                float xv = input[t * IS + j];
                uint w_off = w_base + j * H4;
                gi += xv * W[w_off + gate_i];
                go += xv * W[w_off + gate_o];
                gf += xv * W[w_off + gate_f];
                gc += xv * W[w_off + gate_c];
            }
            if (step > 0u) {
                uint prev_t = (safe_dir == 0u) ? (step - 1u) : (SL - step);
                uint prev_base = prev_t * params.num_directions * H + safe_dir * H;
                for (uint j = 0; j < H; ++j) {
                    float hp = output[prev_base + j];
                    uint r_off = r_base + j * H4;
                    gi += hp * R[r_off + gate_i];
                    go += hp * R[r_off + gate_o];
                    gf += hp * R[r_off + gate_f];
                    gc += hp * R[r_off + gate_c];
                }
            }
            float i_g = 1.0 / (1.0 + exp(-clamp(gi, -44.0, 44.0)));
            float o_g = 1.0 / (1.0 + exp(-clamp(go, -44.0, 44.0)));
            float f_g = 1.0 / (1.0 + exp(-clamp(gf, -44.0, 44.0)));
            float c_g = tanh(clamp(gc, -44.0, 44.0));
            c_val = f_g * c_val + i_g * c_g;
            h_val = o_g * tanh(clamp(c_val, -44.0, 44.0));
            output[t * params.num_directions * H + safe_dir * H + safe_h] = h_val;
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
}

// ── Expansion (row-major and channel-first) ─────────────────────────────────

struct ExpandParams {
    uint seq_len;
    uint dim;
    uint total_frames;
};

kernel void expand_row_major_kernel(device const float* input  [[buffer(0)]],
                                    device const uint*  cumsum [[buffer(1)]],
                                    device float*       output [[buffer(2)]],
                                    constant ExpandParams& params [[buffer(3)]],
                                    uint gid [[thread_position_in_grid]]) {
    uint total = params.total_frames * params.dim;
    if (gid >= total) return;
    uint frame = gid / params.dim;
    uint d = gid % params.dim;
    uint lo = 0, hi = params.seq_len;
    while (lo < hi) {
        uint mid = (lo + hi) / 2;
        if (cumsum[mid] <= frame) lo = mid + 1; else hi = mid;
    }
    output[gid] = input[lo * params.dim + d];
}

kernel void expand_channel_first_kernel(device const float* input  [[buffer(0)]],
                                        device const uint*  cumsum [[buffer(1)]],
                                        device float*       output [[buffer(2)]],
                                        constant ExpandParams& params [[buffer(3)]],
                                        uint gid [[thread_position_in_grid]]) {
    uint total = params.total_frames * params.dim;
    if (gid >= total) return;
    uint channel = gid / params.total_frames;
    uint frame = gid % params.total_frames;
    uint lo = 0, hi = params.seq_len;
    while (lo < hi) {
        uint mid = (lo + hi) / 2;
        if (cumsum[mid] <= frame) lo = mid + 1; else hi = mid;
    }
    output[gid] = input[lo * params.dim + channel];
}

// ── iSTFT ───────────────────────────────────────────────────────────────────

struct IstftParams {
    uint gen_length;
    uint waveform_length;
    uint bins;
    uint kernel_size;
    uint stride;
};

kernel void istft_kernel(device const float* conv_post    [[buffer(0)]],
                         device const float* weight_real [[buffer(1)]],
                         device const float* weight_imag [[buffer(2)]],
                         device float*       output      [[buffer(3)]],
                         constant IstftParams& params [[buffer(4)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid >= params.waveform_length) return;
    uint out_pos = gid;
    float sum = 0.0;
    for (uint k = 0; k < params.kernel_size; ++k) {
        if (out_pos < k) continue;
        uint rem = out_pos - k;
        if (rem % params.stride != 0u) continue;
        uint t = rem / params.stride;
        if (t >= params.gen_length) continue;
        for (uint b = 0; b < params.bins; ++b) {
            float mag_val = conv_post[b * params.gen_length + t];
            float ph_val = conv_post[(b + params.bins) * params.gen_length + t];
            float mag = exp(mag_val);
            float sin_ph = sin(ph_val);
            float real_comp = mag * cos(sin_ph);
            float imag_comp = mag * sin(sin_ph);
            sum += real_comp * weight_real[b * params.kernel_size + k]
                 - imag_comp * weight_imag[b * params.kernel_size + k];
        }
    }
    output[out_pos] = sum;
}