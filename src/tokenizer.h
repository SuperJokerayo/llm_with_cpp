#ifndef TOKENIZER_H_
#define TOKENIZER_H_

#include <vector>
#include <string>

class TokenIndex {
public:
    std::string str;
    int id;
};

class Tokenizer {
public:
    std::vector<std::string> vocab;
    std::vector<float> vocab_scores;
    std::vector<TokenIndex> sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512];

public:
    void init_tokenizer(const char* tokenizer_path, int vocab_size_);

    void encode(
        std::string text, 
        char bos, 
        char eos, 
        std::vector<int> tokens, 
        std::vector<int> n_tokens
    );
    void decode(int prev_token, int token);

};

#endif /* TOKENIZER_H_ */