#ifndef TRANSFORMER_H_
#define TRANSFORMER_H_

#include <cstdio>

#include <vector>
#include <string>

#include "config.h"

using tensor_float_1d = std::vector<float>;
using tensor_float_2d = std::vector<tensor_float_1d>;
using tensor_float_3d = std::vector<tensor_float_2d>;


class TransformerWeights {
public:
    tensor_float_2d token_embedding_table;
    
    tensor_float_2d rms_att_weight;
    tensor_float_2d rms_ffn_weight;
    
    tensor_float_3d wq, wk, wv, wo;

    tensor_float_3d w1, w2, w3;

    tensor_float_1d rms_final_weight;

    tensor_float_2d freq_cis_real;
    tensor_float_2d freq_cis_imag;

public:
    TransformerWeights() {}
    ~TransformerWeights() {}
};

class RunState {
public:
    tensor_float_1d x;
    tensor_float_1d xb;
    tensor_float_1d xb2;
    tensor_float_1d hb;
    tensor_float_1d hb2;
    tensor_float_1d q, k, v;

    tensor_float_1d att;

    tensor_float_1d logits;

    tensor_float_3d key_cache;
    tensor_float_3d value_cache;

public:
    RunState() {}
    ~RunState() {}

};

class Transformer {
public:
    Config config;
    TransformerWeights weights;
    RunState state;

public:
    Transformer() {}
    ~Transformer() {}

    void init_transformerweight(FILE* fp);
    void init_runstate();
    void init_transformer(const std::string checkpoint_path);
    void forward(int token, int pos);
};

#endif /* TRANSFORMER_H_ */