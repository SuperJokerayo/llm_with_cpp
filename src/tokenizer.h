#ifndef TOKENIZER_H_
#define TOKENIZER_H_

#include <sentencepiece_processor.h>

class Tokenizer {
public:
    sentencepiece::SentencePieceProcessor processor;
    int n_words;
    int bos_id;
    int eos_id;
    int pad_id;

public:
    void init_tokenizer(const std::string tokenizer_path);

    void encode(
        std::string text, 
        bool bos, 
        bool eos, 
        std::vector<int>& tokens
    );

    void decode(
        std::vector<int>& tokens,
        std::string& response
    );

    void decode(
        int token, 
        std::string& response
    );

};

#endif /* TOKENIZER_H_ */