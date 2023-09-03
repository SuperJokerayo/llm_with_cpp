#include "llm.h"

#include <ctime>
#include <iostream>

#include <vector>

long time_in_ms() {
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

void safe_printf(std::string piece) {
    if (piece.empty()) { return;}
    if (piece[0] == '\0') { return;}
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return; 
        }
    }
    std::cout << piece;
}

void LLM::generate(
    Tokenizer* tokenizer, 
    Sampler* sampler, 
    Transformer* transformer, 
    const std::string& prompt,
    int steps
) {
    std::vector<int> prompt_tokens;
    tokenizer -> encode(prompt, true, false, prompt_tokens);
    
    int num_prompt_tokens = prompt_tokens.size();

    if (num_prompt_tokens < 1) {
        std::cerr <<  "something is wrong, expected at least 1 prompt token\n" << std::endl;
        exit(EXIT_FAILURE);
    }

    long start = 0;
    int next;
    int token = prompt_tokens[start];
    int pos = 0;

    std::vector<int> tokens{token};

    while(pos < steps) {
        transformer -> forward(token, pos);
        
        if(pos < num_prompt_tokens - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            next = sampler -> sample(transformer -> state.logits);
        }
        pos++;

        if(next == tokenizer -> eos_id) {
            break;
        }

        // std::string response = "";
        // tokenizer -> decode(next, response);
        // safe_printf(response);
        // std::cout << response;
        // fflush(stdout);
        tokens.push_back(next);
        token = next;
        if (start == 0) { start = time_in_ms(); }
    }
    for(auto t : tokens) std::cout << t << " ";
    std::cout << std::endl;
    std::string response = "";
    tokenizer -> decode(tokens, response);
    std::cout << response;
    std::cout << std::endl;

    if (pos > 1) {
        long end = time_in_ms();
        std::cout <<  "achieved tok/s: " << (pos - 1) / (double)(end - start) * 1000 << std::endl;
    }

}


void LLM::chat(
    Tokenizer* tokenizer, 
    Sampler* sampler, 
    Transformer* transformer, 
    const std::string& prompt, 
    const std::string& system_prompt
){};