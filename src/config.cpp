#include "config.h"

Config::Config() {
    this->dim = 4096;
    this->hidden_dim = 0;
    this->n_layers = 6;
    this->n_heads = 6;
    this->n_kv_heads = 6;
    this->vocab_size = 32000;
    this->seq_len = 256;
}

Config::~Config() {}
