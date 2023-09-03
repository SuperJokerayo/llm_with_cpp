#include "transformer.h"
#include <iostream>
#include <cmath>

#include <algorithm>
#include <numeric>

void resize_tensor_float_1d(tensor_float_1d& tensor, int size) {
    tensor.resize(size);
}

void resize_tensor_float_2d(tensor_float_2d& tensor, int size1, int size2) {
    tensor_float_2d(size1, tensor_float_1d(size2)).swap(tensor);   
}

void resize_tensor_float_3d(tensor_float_3d& tensor, int size1, int size2, int size3) {
    tensor_float_3d(size1, tensor_float_2d(size2, tensor_float_1d(size3))).swap(tensor);
}


void init_tensor_from_file(tensor_float_1d& tensor, FILE* fp) {
    int size = tensor.size();
    if (fread(tensor.data(), sizeof(float) * size, 1, fp) != 1) {
        fprintf(stderr, "Couldn't read weights!\n");
        exit(EXIT_FAILURE); 
    }
}

void init_tensor_from_file(tensor_float_2d& tensor, FILE* fp) {
    for(auto& t : tensor) init_tensor_from_file(t, fp);
}

void init_tensor_from_file(tensor_float_3d& tensor, FILE* fp) {
    for(auto& t : tensor) init_tensor_from_file(t, fp);
}

void rmsnorm(
    tensor_float_1d& o, 
    tensor_float_1d& x, 
    tensor_float_1d& weight
) {
    int size = x.size();
    float ss = std::accumulate(x.begin(), x.end(), 0.0f, [](float a, float b) {
        return a + b * b;
    });
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrt(ss);
    for(int j = 0; j < size; j++) {
        o[j] = x[j] * ss * weight[j];
    }
}

void softmax(tensor_float_1d& x) {
    int size = x.size();
    auto max_val = *std::max_element(x.begin(), x.end());
    auto sum = std::accumulate(x.begin(), x.end(), 0.0f, [max_val](float a, float b) {
        return a + exp(b - max_val);
    });
    std::for_each(x.begin(), x.end(), [sum](int a){return a / sum;});
}


void matmul(
    tensor_float_1d& xout, 
    tensor_float_1d& x, 
    tensor_float_2d& w
) {
    int d = w.size();
    int n = x.size();
    int j = 0;
    #pragma omp parallel for private(j)
    for(j = 0; j < d; j++) {
        float val = 0.0f;
        for(int k = 0; k < n; k++) {
            val += w[j][k] * x[k];
        }
        xout[j] = val;
    }
}

void add(
    tensor_float_1d& x, 
    tensor_float_1d& y
) {
    int size = x.size();
    #pragma omp parallel for
    for(int j = 0; j < size; j++) {
        x[j] += y[j];
    }
}


void copy(
    tensor_float_1d& xout, 
    tensor_float_1d& x
) {
    int size = x.size();
    #pragma omp parallel for
    for(int j = 0; j < size; j++) {
        xout[j] = x[j];
    }
}

void copy(
    tensor_float_2d& xout, 
    tensor_float_2d& x
) {
    int size = x.size();
    #pragma omp parallel for
    for(int j = 0; j < size; j++) {
        copy(xout[j], x[j]);
    }
}

void copy(
    tensor_float_3d& xout, 
    tensor_float_3d& x
) {
    int size = x.size();
    #pragma omp parallel for
    for(int j = 0; j < size; j++) {
        copy(xout[j], x[j]);
    }
}


void Transformer::init_transformer(const char* checkpoint_path) {
    FILE *fp = fopen(checkpoint_path, "rb");
    if (!fp) {
        fprintf(stderr, "Couldn't open file %s!\n", checkpoint_path); 
        exit(EXIT_FAILURE); 
    }
    
    if (fread(&config, sizeof(Config), 1, fp) != 1) { 
        fprintf(stderr, "Couldn't read network config!\n");
        exit(EXIT_FAILURE); 
    }

    int shared_weights = config.vocab_size > 0 ? 1 : 0;

    config.vocab_size = abs(config.vocab_size);

    init_transformerweight(fp);
    init_runstate();
    
    fclose(fp);
}

void Transformer::init_transformerweight(FILE* fp) {

    resize_tensor_float_2d(weights.token_embedding_table, config.vocab_size, config.dim);
    resize_tensor_float_2d(weights.rms_att_weight, config.n_layers, config.dim);
    resize_tensor_float_3d(weights.wq, config.n_layers, config.dim, config.dim);
    resize_tensor_float_3d(weights.wk, config.n_layers, config.dim, config.dim);
    resize_tensor_float_3d(weights.wv, config.n_layers, config.dim, config.dim);
    resize_tensor_float_3d(weights.wo, config.n_layers, config.dim, config.dim);
    resize_tensor_float_2d(weights.rms_ffn_weight, config.n_layers, config.dim);
    resize_tensor_float_3d(weights.w1, config.n_layers, config.dim, config.dim);
    resize_tensor_float_3d(weights.w2, config.n_layers, config.dim, config.dim);
    resize_tensor_float_3d(weights.w3, config.n_layers, config.dim, config.dim);
    resize_tensor_float_1d(weights.rms_final_weight, config.dim);

    int head_size = config.dim / config.n_heads;

    resize_tensor_float_2d(weights.freq_cis_real, config.seq_len, head_size / 2);
    resize_tensor_float_2d(weights.freq_cis_imag, config.seq_len, head_size / 2);
    
    init_tensor_from_file(weights.token_embedding_table, fp);
    init_tensor_from_file(weights.rms_att_weight, fp);
    init_tensor_from_file(weights.wq, fp);
    init_tensor_from_file(weights.wk, fp);
    init_tensor_from_file(weights.wv, fp);
    init_tensor_from_file(weights.wo, fp);
    init_tensor_from_file(weights.rms_ffn_weight, fp);
    init_tensor_from_file(weights.w1, fp);
    init_tensor_from_file(weights.w2, fp);
    init_tensor_from_file(weights.w3, fp);
    init_tensor_from_file(weights.rms_final_weight, fp);
    init_tensor_from_file(weights.freq_cis_real, fp);
    init_tensor_from_file(weights.freq_cis_imag, fp);
}


void Transformer::init_runstate() {
    int kv_dim = config.dim * config.n_kv_heads / config.n_heads;

    resize_tensor_float_1d(state.x, config.dim);
    resize_tensor_float_1d(state.xb, config.dim);
    resize_tensor_float_1d(state.xb2, config.dim);
    resize_tensor_float_1d(state.hb, config.dim);
    resize_tensor_float_1d(state.hb2, config.dim);
    resize_tensor_float_1d(state.q, config.dim);
    resize_tensor_float_1d(state.k, config.dim);
    resize_tensor_float_1d(state.v, config.dim);
    resize_tensor_float_1d(state.att, config.seq_len);
    resize_tensor_float_1d(state.logits, config.vocab_size);
    resize_tensor_float_3d(state.key_cache, config.n_layers, config.seq_len, kv_dim);
    resize_tensor_float_3d(state.value_cache, config.n_layers, config.seq_len, kv_dim);
}

void Transformer::forward(int token, int pos) {
    Config* p = &config;
    TransformerWeights* w = &weights;
    RunState* s = &state;

    int dim = p -> dim;
    int kv_dim = dim * p -> n_kv_heads / p -> n_heads;
    int kv_mul = p -> n_heads / p -> n_kv_heads;
    int hidden_dim = p -> hidden_dim;
    int head_size = dim / p -> n_heads;

    copy(s -> x, w -> token_embedding_table[token]);
    for(int l = 0; l < p -> n_layers; ++l) {

        // attention rmsnorm
        rmsnorm(s -> xb, s -> x, w -> rms_att_weight[l]);

        // attention qkv
        matmul(s -> q, s -> xb, w -> wq[l]);
        matmul(s -> k, s -> xb, w -> wk[l]);
        matmul(s -> v, s -> xb, w -> wv[l]);

        // RoPE relative positional encoding
        for(int h = 0; h < p -> n_heads; ++h) {
            int st = h * head_size;
            for(int i = 0; i < head_size; i += 2) {
                auto q0 = s -> q[st + i];
                auto q1 = s -> q[st + i + 1];
                auto k0 = s -> k[st + i];
                auto k1 = s -> k[st + i + 1];
                auto fcr = w -> freq_cis_real[pos][i / 2];
                auto fci = w -> freq_cis_imag[pos][i / 2];
                s -> q[st + i] = q0 * fcr - q1 * fci;
                s -> q[st + i + 1] = q0 * fci + q1 * fcr;
                s -> k[st + i] = k0 * fcr - k1 * fci;
                s -> k[st + i + 1] = k0 * fci + k1 * fcr;
            }
        }
        // save k/v in cache
        copy(s -> key_cache[l][pos], s -> k);
        copy(s -> value_cache[l][pos], s -> v);

        // multiquery attention
        int h;
        const float norm_factor = sqrt(head_size * 1.0f);
        #pragma omp parallel for private(h)
        for(h = 0; h < p -> n_heads; ++h) {
            for(int t = 0; t < pos; ++t) {
                auto score = 0.0f;
                for(int i = 0; i < head_size; ++i) {
                    score += s -> q[h * head_size + i] * s -> key_cache[l][t][h * head_size + i];
                }
                score /= norm_factor;
                s -> att[t] = score;
            }
            softmax(s -> att);

            for(int i = 0; i < head_size; ++i) {
                auto val = 0.0f;
                for(int t = 0; t < pos; ++t) {
                    val += s -> att[t] * s -> value_cache[l][t][h * head_size + i];
                }
                s -> v[h * head_size + i] = val;
            }
        }

        // final matmul to get the output of the attention
        matmul(s -> xb2, s -> xb, w -> wo[l]);

        // residual connection
        add(s -> x, s -> xb2);

        // ffn rmsnorm
        rmsnorm(s -> xb, s -> x, w -> rms_ffn_weight[l]);

        // ffn
        matmul(s -> hb, s -> xb, w -> w1[l]);
        matmul(s -> hb2, s -> xb, w -> w3[l]);

        // SwiGLU non-linearity 
        for(int i = 0; i < hidden_dim; ++i) {
            s -> hb[i] *= (1.0f / (1.0f + exp(-s -> hb[i]))) * s -> hb2[i];
        }

        // final matmul to get the output of the ffn
        matmul(s -> xb, s -> hb, w -> w2[l]);

        add(s -> x, s -> xb);
    }

    rmsnorm(s -> x, s -> x, w -> rms_final_weight);

    matmul(s -> logits, s -> x, w -> token_embedding_table);
}