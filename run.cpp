#include "src/config.h"
#include "src/tokenizer.h"
#include "src/transformer.h"
#include "src/sampler.h"
#include "src/llm.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>

void config_error() {
    std::cerr << "Invalid config!" << std::endl;
    exit(EXIT_FAILURE);
}

void read_config(std::unordered_map<std::string, std::string>& config_map) {
    std::string config_path = "../config.ini";
    std::ifstream fp;
    fp.open(config_path, std::ios::in);
    std::string line;
    while(getline(fp, line)) {
        for(int i = 0; i < line.size(); i++) {
            if(line[i] == '=') {
                std::string key = line.substr(0, i);
                std::string value;
                if(line.back() == '\r')
                    value = line.substr(i + 1, line.size() - i - 2);
                else
                    value = line.substr(i + 1, line.size() - i - 1);
                config_map[key] = value;
                break;
            }
        }
    }
    fp.close();
}

int main() {

    std::cout.tie(NULL);

    std::cout << "Loading config..." << std::endl;
    std::unordered_map<std::string, std::string> config_map;
    read_config(config_map);
    if(config_map.find("checkpoint_path") == config_map.end() or
       config_map.find("tokenizer_path") == config_map.end()  or
       config_map.find("temperature") == config_map.end()     or
       config_map.find("top_p") == config_map.end()           or
       config_map.find("steps") == config_map.end()           or
       config_map.find("prompt") == config_map.end()          or
       config_map.find("rng_seed") == config_map.end()        or
       config_map.find("mode") == config_map.end()            or
       config_map.find("system_prompt") == config_map.end()
    ) {
        config_error();
    }
    const std::string checkpoint_path = config_map["checkpoint_path"].substr(1, config_map["checkpoint_path"].size() - 2);
    const std::string tokenizer_path = config_map["tokenizer_path"].substr(1, config_map["tokenizer_path"].size() - 2);
    const float temperature = std::stof(config_map["temperature"]);
    const float top_p = std::stof(config_map["top_p"]);
    const std::string prompt = config_map["prompt"].substr(1, config_map["prompt"].size() - 2);
    const unsigned long long rng_seed = std::stoi(config_map["rng_seed"]);
    const std::string mode = config_map["mode"].substr(1, config_map["mode"].size() - 2);
    const std::string system_prompt = config_map["system_prompt"].substr(1, config_map["system_prompt"].size() - 2);

    int steps = std::stoi(config_map["steps"]);

    if (temperature < 0.0 or top_p <= 0.0 or 1.0 < top_p or steps <= 0)
        config_error();

    std::cout << "Loading checkpoint..." << std::endl;
    // std::cout << checkpoint_path << std::endl;

    Tokenizer tokenizer;
    tokenizer.init_tokenizer(tokenizer_path);

    Transformer transformer;
    transformer.init_transformer(checkpoint_path);

    if (steps == 0 or steps > transformer.config.seq_len) 
        steps = transformer.config.seq_len;

    Sampler sampler;
    sampler.init_sampler(top_p, temperature, rng_seed);
    
    if (mode == "generate") {
        std::cout << "Generating..." << std::endl;
        LLM llm;
        llm.init_llm(mode);
        llm.generate(&tokenizer, &sampler, &transformer, prompt, steps);
    } else {
        std::cout << "Developing..." << std::endl;
    }

    return 0;
}
