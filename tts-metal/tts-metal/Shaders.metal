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

// ═══════════════════════════════════════════════════════════════════════════
//  RE-USE (SEMamba) speech-enhancement kernels
//  Tensor layout convention below: 2D feature maps are stored channel-major,
//  row-major within each channel:  x[c, h, w] = buf[(c*H + h)*W + w].
//  Sequence tensors for Mamba are [batch, L, D] row-major:
//      x[b, l, d] = buf[(b*L + l)*D + d].
// ═══════════════════════════════════════════════════════════════════════════

// ── Forward STFT → compressed magnitude (log1p) + phase ─────────────────────
// One thread per (freq bin f, frame t). `signal` is already reflect-padded by
// n_fft/2 on each side, so window sample k of frame t is signal[t*hop + k].

struct ReuseStftParams {
    uint n_fft;      // == win_size
    uint hop;
    uint num_freq;   // n_fft/2 + 1
    uint num_frames;
};

kernel void reuse_stft_kernel(device const float* signal [[buffer(0)]],
                              device const float* window [[buffer(1)]],
                              device float*       mag    [[buffer(2)]],
                              device float*       pha    [[buffer(3)]],
                              constant ReuseStftParams& p [[buffer(4)]],
                              uint gid [[thread_position_in_grid]]) {
    uint total = p.num_freq * p.num_frames;
    if (gid >= total) return;
    uint f = gid / p.num_frames;
    uint t = gid % p.num_frames;
    float re = 0.0, im = 0.0;
    float coef = -2.0 * M_PI_F * float(f) / float(p.n_fft);
    // Phasor recurrence: (ck, sk) = (cos(coef*k), sin(coef*k)) advanced by one step
    // per iteration instead of calling cos/sin for every sample.
    float cs = cos(coef), sn = sin(coef);
    float ck = 1.0, sk = 0.0;
    uint base = t * p.hop;
    for (uint k = 0; k < p.n_fft; ++k) {
        float s = window[k] * signal[base + k];
        re = fma(s, ck, re);
        im = fma(s, sk, im);
        float nck = ck * cs - sk * sn;
        sk = ck * sn + sk * cs;
        ck = nck;
    }
    float m = sqrt(re * re + im * im);
    mag[gid] = log(1.0 + m);          // log1p compression (relu_log1p)
    pha[gid] = atan2(im, re);
}

// ── General 2D convolution (bias optional) ──────────────────────────────────
struct ReuseConv2dParams {
    uint in_ch;  uint out_ch;
    uint H;      uint W;
    uint kh;     uint kw;
    uint pad_h;  uint pad_w;
    uint str_h;  uint str_w;
    uint dil_h;  uint dil_w;
    uint out_h;  uint out_w;
    uint use_bias; uint _pad;
};

kernel void reuse_conv2d_kernel(device const float* input  [[buffer(0)]],
                                device const float* weight [[buffer(1)]],
                                device const float* bias   [[buffer(2)]],
                                device float*       output [[buffer(3)]],
                                constant ReuseConv2dParams& p [[buffer(4)]],
                                uint gid [[thread_position_in_grid]]) {
    uint total = p.out_ch * p.out_h * p.out_w;
    if (gid >= total) return;
    uint ow = gid % p.out_w;
    uint oh = (gid / p.out_w) % p.out_h;
    uint oc = gid / (p.out_w * p.out_h);
    float sum = 0.0;
    for (uint ic = 0; ic < p.in_ch; ++ic) {
        uint in_ch_base = ic * p.H * p.W;
        uint w_base = ((oc * p.in_ch) + ic) * p.kh * p.kw;
        for (uint r = 0; r < p.kh; ++r) {
            int ih = int(oh * p.str_h) + int(r * p.dil_h) - int(p.pad_h);
            if (ih < 0 || uint(ih) >= p.H) continue;
            uint in_row = in_ch_base + uint(ih) * p.W;
            uint w_row = w_base + r * p.kw;
            for (uint c = 0; c < p.kw; ++c) {
                int iw = int(ow * p.str_w) + int(c * p.dil_w) - int(p.pad_w);
                if (iw < 0 || uint(iw) >= p.W) continue;
                sum = fma(input[in_row + uint(iw)], weight[w_row + c], sum);
            }
        }
    }
    if (p.use_bias != 0u) sum += bias[oc];
    output[gid] = sum;
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

// ── Tiled 2D convolution: one threadgroup per output channel, weights cached ──
// in threadgroup memory and reused across all spatial positions. Used when the
// per-channel weight row (in_ch*kh*kw) fits the tile; the dense blocks (256*3*3
// = 2304) dominate RE-USE runtime, so this is the hot path.
kernel void reuse_conv2d_tiled_kernel(device const float* input  [[buffer(0)]],
                                      device const float* weight [[buffer(1)]],
                                      device const float* bias   [[buffer(2)]],
                                      device float*       output [[buffer(3)]],
                                      constant ReuseConv2dParams& p [[buffer(4)]],
                                      uint2 wid [[threadgroup_position_in_grid]],
                                      uint2 lid [[thread_position_in_threadgroup]]) {
    constexpr uint WG = 256;
    threadgroup half wtile[4096];               // fp16 weights: 2× rate, half the LDS
    uint oc = wid.y;
    uint row_size = p.in_ch * p.kh * p.kw;
    uint w_base = oc * row_size;
    for (uint i = lid.x; i < row_size; i += WG) wtile[i] = half(weight[w_base + i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint sp = wid.x * WG + lid.x;               // spatial output index
    uint out_hw = p.out_h * p.out_w;
    if (sp >= out_hw) return;
    uint ow = sp % p.out_w;
    uint oh = sp / p.out_w;

    // fp16 multiply, fp32 accumulate. Convs are feed-forward, so fp16 rounding does
    // not compound the way it would through the recurrent Mamba scan.
    float sum = 0.0;
    for (uint ic = 0; ic < p.in_ch; ++ic) {
        uint in_ch_base = ic * p.H * p.W;
        uint wc_base = ic * p.kh * p.kw;
        for (uint r = 0; r < p.kh; ++r) {
            int ih = int(oh * p.str_h) + int(r * p.dil_h) - int(p.pad_h);
            if (ih < 0 || uint(ih) >= p.H) continue;
            uint in_row = in_ch_base + uint(ih) * p.W;
            uint w_row = wc_base + r * p.kw;
            for (uint c = 0; c < p.kw; ++c) {
                int iw = int(ow * p.str_w) + int(c * p.dil_w) - int(p.pad_w);
                if (iw < 0 || uint(iw) >= p.W) continue;
                sum += float(half(input[in_row + uint(iw)]) * wtile[w_row + c]);
            }
        }
    }
    if (p.use_bias != 0u) sum += bias[oc];
    output[oc * out_hw + sp] = sum;
}

// ── InstanceNorm2d with affine (one threadgroup per channel) ─────────────────
struct ReuseInstNorm2dParams { uint channels; uint length; float eps; };

kernel void reuse_instance_norm2d_kernel(device const float* input  [[buffer(0)]],
                                         device const float* gamma  [[buffer(1)]],
                                         device const float* beta   [[buffer(2)]],
                                         device float*       output [[buffer(3)]],
                                         constant ReuseInstNorm2dParams& p [[buffer(4)]],
                                         uint2 wid [[threadgroup_position_in_grid]],
                                         uint2 lid [[thread_position_in_threadgroup]]) {
    constexpr uint WGT = 256;
    threadgroup float red[256];
    threadgroup float sh_mean;
    threadgroup float sh_inv;
    uint ch = wid.x;
    bool active = ch < p.channels;
    uint tid = lid.x;
    uint L = p.length;
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
    float vs = 0.0;
    for (uint i = tid; i < L; i += WGT) { float d = input[base + i] - mean; vs += d * d; }
    red[tid] = vs;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = WGT / 2; stride > 0; stride >>= 1) {
        if (tid < stride) red[tid] += red[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) sh_inv = 1.0 / sqrt(red[0] / float(L) + p.eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!active) return;
    float inv = sh_inv;
    float g = gamma[ch], b = beta[ch];
    for (uint i = tid; i < L; i += WGT) {
        output[base + i] = (input[base + i] - mean) * inv * g + b;
    }
}

// ── PReLU (per-channel slope) ───────────────────────────────────────────────
struct ReuseChanLenParams { uint channels; uint length; };

kernel void reuse_prelu_kernel(device const float* input  [[buffer(0)]],
                               device const float* slope  [[buffer(1)]],
                               device float*       output [[buffer(2)]],
                               constant ReuseChanLenParams& p [[buffer(3)]],
                               uint gid [[thread_position_in_grid]]) {
    uint total = p.channels * p.length;
    if (gid >= total) return;
    uint ch = gid / p.length;
    float x = input[gid];
    output[gid] = (x >= 0.0) ? x : slope[ch] * x;
}

// ── Constant (zero) pad 2D ──────────────────────────────────────────────────
struct ReusePad2dParams {
    uint channels; uint H; uint W;
    uint pad_top; uint pad_bottom; uint pad_left; uint pad_right;
};

kernel void reuse_pad2d_kernel(device const float* input  [[buffer(0)]],
                               device float*       output [[buffer(1)]],
                               constant ReusePad2dParams& p [[buffer(2)]],
                               uint gid [[thread_position_in_grid]]) {
    uint outH = p.H + p.pad_top + p.pad_bottom;
    uint outW = p.W + p.pad_left + p.pad_right;
    uint total = p.channels * outH * outW;
    if (gid >= total) return;
    uint ow = gid % outW;
    uint oh = (gid / outW) % outH;
    uint c  = gid / (outW * outH);
    if (oh < p.pad_top || oh >= p.pad_top + p.H ||
        ow < p.pad_left || ow >= p.pad_left + p.W) {
        output[gid] = 0.0;
        return;
    }
    uint ih = oh - p.pad_top;
    uint iw = ow - p.pad_left;
    output[gid] = input[(c * p.H + ih) * p.W + iw];
}

// ── Sub-pixel width shuffle for SPConvTranspose2d ───────────────────────────
// in : [out_ch*r, H, W]   out : [out_ch, H, W*r]
// out[c, h, w*r + rr] = in[rr*out_ch + c, h, w]
struct ReusePixelShuffleParams { uint out_ch; uint r; uint H; uint W; };

kernel void reuse_pixelshuffle_w_kernel(device const float* input  [[buffer(0)]],
                                        device float*       output [[buffer(1)]],
                                        constant ReusePixelShuffleParams& p [[buffer(2)]],
                                        uint gid [[thread_position_in_grid]]) {
    uint Wr = p.W * p.r;
    uint total = p.out_ch * p.H * Wr;
    if (gid >= total) return;
    uint ow = gid % Wr;
    uint oh = (gid / Wr) % p.H;
    uint c  = gid / (Wr * p.H);
    uint w  = ow / p.r;
    uint rr = ow % p.r;
    uint in_ch = rr * p.out_ch + c;
    output[gid] = input[(in_ch * p.H + oh) * p.W + w];
}

// ── Per-channel spatial transpose (swap H and W) ────────────────────────────
struct ReuseTransposeHWParams { uint channels; uint H; uint W; };

kernel void reuse_transpose_hw_kernel(device const float* input  [[buffer(0)]],
                                      device float*       output [[buffer(1)]],
                                      constant ReuseTransposeHWParams& p [[buffer(2)]],
                                      uint gid [[thread_position_in_grid]]) {
    uint total = p.channels * p.H * p.W;
    if (gid >= total) return;
    uint w = gid % p.W;
    uint h = (gid / p.W) % p.H;
    uint c = gid / (p.W * p.H);
    // out[c, w, h] over dims (H'=W, W'=H)
    output[(c * p.W + w) * p.H + h] = input[gid];
}

// ── Flip a [batch, L, D] tensor along L ─────────────────────────────────────
struct ReuseFlipParams { uint batch; uint L; uint dim; };

kernel void reuse_flip_kernel(device const float* input  [[buffer(0)]],
                              device float*       output [[buffer(1)]],
                              constant ReuseFlipParams& p [[buffer(2)]],
                              uint gid [[thread_position_in_grid]]) {
    uint total = p.batch * p.L * p.dim;
    if (gid >= total) return;
    uint d = gid % p.dim;
    uint l = (gid / p.dim) % p.L;
    uint b = gid / (p.dim * p.L);
    uint src_l = p.L - 1 - l;
    output[gid] = input[(b * p.L + src_l) * p.dim + d];
}

// ── Mamba depthwise causal conv over L + bias + SiLU ────────────────────────
// xz : [batch*L, in_stride] with the "x" channels in cols [0, d_inner).
// out: [batch*L, d_inner] = SiLU(conv1d_causal(x) + conv_bias)
struct ReuseMambaConvParams {
    uint batch; uint L; uint d_inner; uint in_stride; uint k;
};

kernel void reuse_mamba_conv_kernel(device const float* xz      [[buffer(0)]],
                                    device const float* weight  [[buffer(1)]],  // [d_inner, 1, k]
                                    device const float* bias    [[buffer(2)]],  // [d_inner]
                                    device float*       out     [[buffer(3)]],  // [batch*L, d_inner]
                                    constant ReuseMambaConvParams& p [[buffer(4)]],
                                    uint gid [[thread_position_in_grid]]) {
    uint total = p.batch * p.L * p.d_inner;
    if (gid >= total) return;
    uint d = gid % p.d_inner;
    uint l = (gid / p.d_inner) % p.L;
    uint b = gid / (p.d_inner * p.L);
    float sum = bias[d];
    uint wbase = d * p.k;
    // causal taps: input positions l-(k-1) .. l
    for (uint j = 0; j < p.k; ++j) {
        int lp = int(l) + int(j) - int(p.k - 1);
        if (lp < 0) continue;
        uint row = b * p.L + uint(lp);
        sum += xz[row * p.in_stride + d] * weight[wbase + j];
    }
    out[gid] = sum / (1.0 + exp(-sum));   // SiLU
}

// ── Mamba selective scan (sequential over L, parallel over batch*d_inner) ───
struct ReuseScanParams {
    uint batch; uint L; uint d_inner; uint d_state; uint dbl_stride; uint xz_stride;
};

kernel void reuse_selective_scan_kernel(device const float* u       [[buffer(0)]],  // [batch*L, d_inner]
                                        device const float* dt_raw  [[buffer(1)]],  // [batch*L, d_inner]
                                        device const float* xdbl    [[buffer(2)]],  // [batch*L, dbl_stride], B@[dt_rank..], C@[..]
                                        device const float* xz      [[buffer(3)]],  // [batch*L, xz_stride], z@[d_inner..]
                                        device const float* A_log   [[buffer(4)]],  // [d_inner, d_state]
                                        device const float* Dvec    [[buffer(5)]],  // [d_inner]
                                        device const float* dt_bias [[buffer(6)]],  // [d_inner]
                                        device float*       y       [[buffer(7)]],  // [batch*L, d_inner]
                                        constant ReuseScanParams& p [[buffer(8)]],
                                        uint gid [[thread_position_in_grid]]) {
    uint total = p.batch * p.d_inner;
    if (gid >= total) return;
    uint d = gid % p.d_inner;
    uint b = gid / p.d_inner;
    uint N = p.d_state;
    uint dt_rank = p.dbl_stride - 2u * N;   // B offset = dt_rank, C offset = dt_rank+N

    float A[16];
    float h[16];
    for (uint n = 0; n < N; ++n) { A[n] = -exp(A_log[d * N + n]); h[n] = 0.0; }
    float Dd = Dvec[d];
    float dbias = dt_bias[d];

    for (uint l = 0; l < p.L; ++l) {
        uint row = b * p.L + l;
        float uu = u[row * p.d_inner + d];
        float dtr = dt_raw[row * p.d_inner + d] + dbias;
        // softplus
        float dt = (dtr > 20.0) ? dtr : log(1.0 + exp(dtr));
        float z = xz[row * p.xz_stride + p.d_inner + d];
        uint dblbase = row * p.dbl_stride;
        float acc = 0.0;
        float dtu = dt * uu;
        for (uint n = 0; n < N; ++n) {
            float Bn = xdbl[dblbase + dt_rank + n];
            float Cn = xdbl[dblbase + dt_rank + N + n];
            float dA = exp(dt * A[n]);
            h[n] = fma(dA, h[n], dtu * Bn);
            acc = fma(h[n], Cn, acc);
        }
        acc += Dd * uu;
        acc *= z / (1.0 + exp(-z));   // * SiLU(z)
        y[row * p.d_inner + d] = acc;
    }
}

// ── amp/phase → complex spectrum with artifact-frame zeroing ────────────────
// mag_out[f,t] = expm1(relu(amp[f,t])); if >50% of a frame's bins are zero,
// the whole frame is zeroed. Emits real[f,t], imag[f,t].
struct ReuseSpecParams { uint num_freq; uint num_frames; };

kernel void reuse_zero_frac_kernel(device const float* amp      [[buffer(0)]],  // [F, T]
                                   device float*       zero_frac [[buffer(1)]], // [T]
                                   constant ReuseSpecParams& p [[buffer(2)]],
                                   uint gid [[thread_position_in_grid]]) {
    if (gid >= p.num_frames) return;
    uint t = gid;
    uint zeros = 0;
    for (uint f = 0; f < p.num_freq; ++f) {
        if (amp[f * p.num_frames + t] <= 0.0) zeros++;   // expm1(relu(x))==0  <=>  x<=0
    }
    zero_frac[t] = float(zeros) / float(p.num_freq);
}

kernel void reuse_spec_to_complex_kernel(device const float* amp       [[buffer(0)]],  // [F, T]
                                         device const float* pha       [[buffer(1)]],  // [F, T]
                                         device const float* zero_frac [[buffer(2)]],  // [T]
                                         device float*       out_real  [[buffer(3)]],  // [F, T]
                                         device float*       out_imag  [[buffer(4)]],  // [F, T]
                                         constant ReuseSpecParams& p [[buffer(5)]],
                                         uint gid [[thread_position_in_grid]]) {
    uint total = p.num_freq * p.num_frames;
    if (gid >= total) return;
    uint t = gid % p.num_frames;
    float a = amp[gid];
    float m = (a > 0.0) ? (exp(a) - 1.0) : 0.0;   // expm1(relu(amp))
    if (zero_frac[t] > 0.5) m = 0.0;
    float ph = pha[gid];
    out_real[gid] = m * cos(ph);
    out_imag[gid] = m * sin(ph);
}

// ── Per-frame windowed inverse rFFT ─────────────────────────────────────────
// real/imag: [F, T] onesided (F = n_fft/2+1). Output frames[t, k] windowed.
struct ReuseIrfftParams { uint n_fft; uint num_freq; uint num_frames; };

kernel void reuse_irfft_kernel(device const float* re     [[buffer(0)]],  // [F, T]
                               device const float* im     [[buffer(1)]],  // [F, T]
                               device const float* window [[buffer(2)]],  // [n_fft]
                               device float*       frames [[buffer(3)]],  // [T, n_fft]
                               constant ReuseIrfftParams& p [[buffer(4)]],
                               uint gid [[thread_position_in_grid]]) {
    uint total = p.num_frames * p.n_fft;
    if (gid >= total) return;
    uint k = gid % p.n_fft;
    uint t = gid / p.n_fft;
    uint N = p.n_fft;
    uint nyq_idx = N / 2;   // Nyquist index
    float acc = re[0 * p.num_frames + t];                       // DC (f=0)
    float coef = 2.0 * M_PI_F * float(k) / float(N);
    // Phasor recurrence over frequency bins: (cf, sf) = (cos(coef*f), sin(coef*f)).
    float cs = cos(coef), sn = sin(coef);
    float cf = cs, sf = sn;                                     // start at f = 1
    for (uint f = 1; f < nyq_idx; ++f) {
        acc += 2.0 * (re[f * p.num_frames + t] * cf - im[f * p.num_frames + t] * sf);
        float ncf = cf * cs - sf * sn;
        sf = cf * sn + sf * cs;
        cf = ncf;
    }
    // Nyquist term (f = N/2): cos(pi*k) = (-1)^k, sin term drops out for real signal
    float nyq = re[nyq_idx * p.num_frames + t];
    acc += nyq * ((k & 1u) ? -1.0 : 1.0);
    acc /= float(N);
    frames[t * N + k] = acc * window[k];
}

// ── Overlap-add with window-envelope normalisation, then centre-trim ────────
struct ReuseOlaParams {
    uint n_fft; uint hop; uint num_frames; uint pad_len; uint out_len; uint trim;
};

kernel void reuse_ola_kernel(device const float* frames [[buffer(0)]],  // [T, n_fft]
                             device const float* window [[buffer(1)]],  // [n_fft]
                             device float*       output [[buffer(2)]],  // [out_len]
                             constant ReuseOlaParams& p [[buffer(3)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= p.out_len) return;
    uint n = gid + p.trim;                 // position in padded coordinates
    float num = 0.0, den = 0.0;
    // frames t with t*hop <= n < t*hop + n_fft  →  k = n - t*hop in [0, n_fft)
    uint t_lo = (n >= p.n_fft - 1) ? ((n - (p.n_fft - 1) + p.hop - 1) / p.hop) : 0u;
    uint t_hi = n / p.hop;
    for (uint t = t_lo; t <= t_hi && t < p.num_frames; ++t) {
        uint k = n - t * p.hop;
        if (k >= p.n_fft) continue;
        float w = window[k];
        num += frames[t * p.n_fft + k];
        den += w * w;
    }
    output[gid] = (den > 1e-11) ? (num / den) : 0.0;
}

// ── Generic 3D permute ──────────────────────────────────────────────────────
// Output axis j reads input axis perm[j]; output dims = (d[p0], d[p1], d[p2]).
struct ReusePermute3dParams { uint d0; uint d1; uint d2; uint p0; uint p1; uint p2; };

kernel void reuse_permute3d_kernel(device const float* input  [[buffer(0)]],
                                   device float*       output [[buffer(1)]],
                                   constant ReusePermute3dParams& p [[buffer(2)]],
                                   uint gid [[thread_position_in_grid]]) {
    uint d[3]; d[0] = p.d0; d[1] = p.d1; d[2] = p.d2;
    uint perm[3]; perm[0] = p.p0; perm[1] = p.p1; perm[2] = p.p2;
    uint o0 = d[perm[0]], o1 = d[perm[1]], o2 = d[perm[2]];
    uint total = o0 * o1 * o2;
    if (gid >= total) return;
    uint c2 = gid % o2;
    uint c1 = (gid / o2) % o1;
    uint c0 = gid / (o2 * o1);
    uint incoord[3];
    incoord[perm[0]] = c0;
    incoord[perm[1]] = c1;
    incoord[perm[2]] = c2;
    uint in_lin = (incoord[0] * d[1] + incoord[1]) * d[2] + incoord[2];
    output[gid] = input[in_lin];
}

// ── Slice contiguous columns out of a [rows, in_stride] matrix ───────────────
struct ReuseSliceColsParams { uint rows; uint in_stride; uint col_off; uint col_count; };

kernel void reuse_slice_cols_kernel(device const float* input  [[buffer(0)]],
                                    device float*       output [[buffer(1)]],
                                    constant ReuseSliceColsParams& p [[buffer(2)]],
                                    uint gid [[thread_position_in_grid]]) {
    uint total = p.rows * p.col_count;
    if (gid >= total) return;
    uint r = gid / p.col_count;
    uint c = gid % p.col_count;
    output[gid] = input[r * p.in_stride + p.col_off + c];
}

// ── Concatenate two [rows, *] matrices along the last dim ───────────────────
struct ReuseConcatColsParams { uint rows; uint cols_a; uint cols_b; };

kernel void reuse_concat_cols_kernel(device const float* a   [[buffer(0)]],
                                     device const float* b   [[buffer(1)]],
                                     device float*       out [[buffer(2)]],
                                     constant ReuseConcatColsParams& p [[buffer(3)]],
                                     uint gid [[thread_position_in_grid]]) {
    uint total_cols = p.cols_a + p.cols_b;
    uint total = p.rows * total_cols;
    if (gid >= total) return;
    uint r = gid / total_cols;
    uint c = gid % total_cols;
    out[gid] = (c < p.cols_a) ? a[r * p.cols_a + c]
                              : b[r * p.cols_b + (c - p.cols_a)];
}

// ── Crop a [inH, inW] map to its top-left [outH, outW] corner ────────────────
struct ReuseCropParams { uint in_h; uint in_w; uint out_h; uint out_w; };

kernel void reuse_crop2d_kernel(device const float* input  [[buffer(0)]],
                                device float*       output [[buffer(1)]],
                                constant ReuseCropParams& p [[buffer(2)]],
                                uint gid [[thread_position_in_grid]]) {
    uint total = p.out_h * p.out_w;
    if (gid >= total) return;
    uint w = gid % p.out_w;
    uint h = gid / p.out_w;
    output[gid] = input[h * p.in_w + w];
}

// ── Elementwise atan2(y, x) ─────────────────────────────────────────────────
kernel void reuse_atan2_kernel(device const float* y   [[buffer(0)]],
                               device const float* x   [[buffer(1)]],
                               device float*       out [[buffer(2)]],
                               constant SizeParams& p [[buffer(3)]],
                               uint gid [[thread_position_in_grid]]) {
    if (gid >= p.size) return;
    out[gid] = atan2(y[gid], x[gid]);
}
// ═══════════════════════════════════════════════════════════════════════════
//  LavaSR v2 (Vocos) bandwidth-extension kernels
//  Mel → Conv1d embed → 8× ConvNeXt → LayerNorm → ISTFTHead.
//  Sequence tensors are [dim, T] (channel-major) or [T, dim] (row-major);
//  the reused matmul / irfft / ola / conv1d / transpose / layernorm / gelu
//  kernels handle the heavy lifting; only the pieces below are LavaSR-specific.
// ═══════════════════════════════════════════════════════════════════════════

// ── Forward STFT → linear magnitude [F, T] (power=1, no compression) ─────────
// `signal` is already reflect-padded, so window sample k of frame t is
// signal[t*hop + k]. Uses a phasor recurrence instead of per-sample cos/sin.
kernel void lava_stft_mag_kernel(device const float* signal [[buffer(0)]],
                                 device const float* window [[buffer(1)]],
                                 device float*       mag    [[buffer(2)]],
                                 constant ReuseStftParams& p [[buffer(3)]],
                                 uint gid [[thread_position_in_grid]]) {
    uint total = p.num_freq * p.num_frames;
    if (gid >= total) return;
    uint f = gid / p.num_frames;
    uint t = gid % p.num_frames;
    float re = 0.0, im = 0.0;
    float coef = -2.0 * M_PI_F * float(f) / float(p.n_fft);
    float cs = cos(coef), sn = sin(coef);
    float ck = 1.0, sk = 0.0;
    uint base = t * p.hop;
    for (uint k = 0; k < p.n_fft; ++k) {
        float s = window[k] * signal[base + k];
        re = fma(s, ck, re);
        im = fma(s, sk, im);
        float nck = ck * cs - sk * sn;
        sk = ck * sn + sk * cs;
        ck = nck;
    }
    mag[gid] = sqrt(re * re + im * im);
}

// ── safe_log: log(clip(x, min=1e-7)) — applied to the mel spectrogram ────────
kernel void lava_safe_log_kernel(device const float* input  [[buffer(0)]],
                                 device float*       output [[buffer(1)]],
                                 constant SizeParams& p [[buffer(2)]],
                                 uint gid [[thread_position_in_grid]]) {
    if (gid >= p.size) return;
    output[gid] = log(max(input[gid], 1e-7));
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

// ── ISTFTHead: split [T, 2F] into magnitude/phase, form complex spectrum ─────
struct LavaHeadParams { uint num_freq; uint num_frames; };

kernel void lava_head_to_complex_kernel(device const float* head [[buffer(0)]],  // [T, 2F]
                                        device float* out_real   [[buffer(1)]],  // [F, T]
                                        device float* out_imag   [[buffer(2)]],  // [F, T]
                                        constant LavaHeadParams& p [[buffer(3)]],
                                        uint gid [[thread_position_in_grid]]) {
    uint total = p.num_freq * p.num_frames;
    if (gid >= total) return;
    uint f = gid / p.num_frames;
    uint t = gid % p.num_frames;
    uint stride = 2 * p.num_freq;
    float magraw = head[t * stride + f];
    float ph = head[t * stride + p.num_freq + f];
    float m = min(exp(magraw), 1e2);   // exp then clamp, matching Vocos ISTFTHead
    out_real[f * p.num_frames + t] = m * cos(ph);
    out_imag[f * p.num_frames + t] = m * sin(ph);
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
// Q/K/V:[L, H*D]. emb_rel_k/emb_rel_v: [H, 2*window+1, D] (per-head rel embeds).
// Adds rel-position logits and rel-position value contributions. key_mask:[L].
struct StRelAttnParams { uint seq_len; uint num_heads; uint head_dim; uint window; float scale; };

kernel void st_rel_attn_kernel(device const float* Q        [[buffer(0)]],
                               device const float* K        [[buffer(1)]],
                               device const float* V        [[buffer(2)]],
                               device const float* rel_k    [[buffer(3)]],  // [H,2w+1,D]
                               device const float* rel_v    [[buffer(4)]],  // [H,2w+1,D]
                               device const float* key_mask [[buffer(5)]],  // [L] or null
                               device float*       out      [[buffer(6)]],
                               constant StRelAttnParams& p [[buffer(7)]],
                               uint gid [[thread_position_in_grid]]) {
    uint d = gid % p.head_dim;
    uint h = (gid / p.head_dim) % p.num_heads;
    uint q = gid / (p.head_dim * p.num_heads);
    if (q >= p.seq_len) return;
    uint HD = p.num_heads * p.head_dim;
    uint W = 2u * p.window + 1u;
    uint q_base = q * HD + h * p.head_dim;
    uint rel_base = h * W * p.head_dim;
    float mx = -1e30f;
    for (uint k = 0; k < p.seq_len; ++k) {
        if (key_mask && key_mask[k] == 0.0f) continue;
        uint k_base = k * HD + h * p.head_dim;
        float sc = 0.0f;
        for (uint e = 0; e < p.head_dim; ++e) sc += Q[q_base + e] * K[k_base + e];
        // relative key term: clamp (k - q) to [-window, window]
        int rel = int(k) - int(q);
        rel = clamp(rel, -int(p.window), int(p.window));
        uint ridx = uint(rel + int(p.window));
        float rsc = 0.0f;
        for (uint e = 0; e < p.head_dim; ++e)
            rsc += Q[q_base + e] * rel_k[rel_base + ridx * p.head_dim + e];
        sc = (sc + rsc) * p.scale;
        mx = max(mx, sc);
    }
    float denom = 0.0f;
    // first pass weights are recomputed; accumulate value + rel_value
    float acc = 0.0f;
    for (uint k = 0; k < p.seq_len; ++k) {
        if (key_mask && key_mask[k] == 0.0f) continue;
        uint k_base = k * HD + h * p.head_dim;
        float sc = 0.0f;
        for (uint e = 0; e < p.head_dim; ++e) sc += Q[q_base + e] * K[k_base + e];
        int rel = int(k) - int(q);
        rel = clamp(rel, -int(p.window), int(p.window));
        uint ridx = uint(rel + int(p.window));
        float rsc = 0.0f;
        for (uint e = 0; e < p.head_dim; ++e)
            rsc += Q[q_base + e] * rel_k[rel_base + ridx * p.head_dim + e];
        float w = exp((sc + rsc) * p.scale - mx);
        denom += w;
        acc += w * (V[k_base + d] + rel_v[rel_base + ridx * p.head_dim + d]);
    }
    out[q * HD + h * p.head_dim + d] = (denom > 0.0f) ? acc / denom : 0.0f;
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
    float denom = (hf > 1u) ? float(hf) : 1.0f;
    float freq = exp(-log(p.max_period) * float(i) / denom);
    float ang = t[r] * freq;
    out[r * p.dim + i]      = cos(ang);
    out[r * p.dim + hf + i] = sin(ang);
}

// ── PReLU/tanh helpers already exist (reuse_prelu_kernel, tanh_kernel) ───────

