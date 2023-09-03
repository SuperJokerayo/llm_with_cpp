#ifndef CONFIG_H_
#define CONFIG_H_


/**
 *  network config read from checkpoint, not the config from config.ini
 *  export.py pack header order:
 *  dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len
 * 
*/


class Config {
public:
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;

public:
    Config();
    ~Config();
};

#endif /* CONFIG_H_ */