#ifndef LLM_H_
#define LLM_H_

#include <string>

#include "tokenizer.h"
#include "sampler.h"
#include "transformer.h"

class LLM {
private:
    std::string mode;

public:
    void init_llm(const std::string mode_) {
        mode = mode_;
    }
    void set_mode(const std::string mode_) {
        mode = mode_;
    }
    void generate(
        Tokenizer* tokenizer, 
        Sampler* sampler, 
        Transformer* transformer, 
        const std::string& prompt,
        int steps
    );
    void chat(
        Tokenizer* tokenizer, 
        Sampler* sampler, 
        Transformer* transformer, 
        const std::string& prompt, 
        const std::string& system_prompt
    );
};

#endif /* LLM_H_ */