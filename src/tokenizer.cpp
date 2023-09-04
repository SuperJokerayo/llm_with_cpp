#include "tokenizer.h"

#include <iostream>

void Tokenizer::init_tokenizer(const std::string tokenizer_path) {
    const auto status = processor.Load(tokenizer_path);
    if(!status.ok()) {
        std::cerr << status.ToString() << std::endl;
    }

    n_words = processor.GetPieceSize();
    bos_id = processor.bos_id();
    eos_id = processor.eos_id();
    pad_id = processor.pad_id();
}

void Tokenizer::encode(
    std::string text, 
    bool bos, 
    bool eos, 
    std::vector<int>& tokens
) { 
    
    if(text.empty()) {
        std::cerr << "Empty text cannot be encoded!" << std::endl;
        exit(EXIT_FAILURE);
    }
    
    processor.Encode(text, &tokens);

    if(bos) {
        tokens.insert(tokens.begin(), bos_id);
    }

    if(eos) {
        tokens.emplace_back(eos_id);
    }
}

void Tokenizer::decode(
    std::vector<int>& tokens,
    std::string& response
) {
    processor.Decode(tokens, &response);
}

void Tokenizer::decode(
    int token,
    std::string& response
) {
    processor.Decode(std::vector<int>{token}, &response);
}


