#include "tokenizer.h"

#include <cstdio>
#include <cstdlib>

#include <algorithm>

void Tokenizer::init_tokenizer(const char* tokenizer_path, int vocab_size_) {
    vocab_size = vocab_size_;
    vocab.resize(vocab_size);
    vocab_scores.resize(vocab_size);
    sorted_vocab.clear();

    for(int i = 0; i < 256; i++) {
        byte_pieces[i * 2] = (unsigned char)i;
        byte_pieces[i * 2 + 1] = '\0';
    }

    FILE *fp = fopen(tokenizer_path, "rb");
    if (!fp) {
        fprintf(stderr, "couldn't load %s\n", tokenizer_path); 
        exit(EXIT_FAILURE); 
    }
    if (fread(&max_token_length, sizeof(int), 1, fp) != 1) { 
        fprintf(stderr, "failed read\n"); 
        exit(EXIT_FAILURE); 
    }
    int len;
    for (int i = 0; i < vocab_size; i++) {
        
        if (fread(&vocab_scores[i], sizeof(float), 1, fp) != 1) {
            fprintf(stderr, "failed read\n"); 
            exit(EXIT_FAILURE);
        }

        if (fread(&len, sizeof(int), 1, fp) != 1) { 
            fprintf(stderr, "failed read\n"); 
            exit(EXIT_FAILURE); 
        }

        vocab[i].resize(len);
        if (fread(&vocab[i], len, 1, fp) != 1) { 
            fprintf(stderr, "failed read\n"); 
            exit(EXIT_FAILURE); 
        }
    }
    fclose(fp);
}

void Tokenizer::encode(
    std::string text, 
    char bos, 
    char eos, 
    std::vector<int> tokens, 
    std::vector<int> n_tokens
) {
    if(text.empty()) {
        fprintf(stderr, "cannot encode NULL text\n"); 
        exit(EXIT_FAILURE);
    }

    if(sorted_vocab.empty()) {
        sorted_vocab.resize(vocab_size);
        for(int i = 0; i < vocab_size; i++) {
            sorted_vocab[i].str = vocab[i];
            sorted_vocab[i].id = i;
        }
        std::sort(sorted_vocab.begin(), sorted_vocab.end(), [](const TokenIndex& a, const TokenIndex& b) {
            return a.str < b.str;
        });
    }

    std::string str_buffer("\0", (max_token_length * 2 + 1 + 2));
    size_t str_len = 0;

    n_tokens.clear();

}