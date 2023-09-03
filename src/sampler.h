#ifndef SAMPLER_H_
#define SAMPLER_H_

#include <vector>

class Sampler {
private:
    float p;
    float temperature;
    unsigned long long rng_seed;

public:
    void init_sampler(float p_, float temperature_, unsigned long long rng_seed_) {
        p = p_;
        temperature = temperature_;
        rng_seed = rng_seed_;
    }
    void set_p(float p_) {
        p = p_;
    }
    void set_temperature(float temperature_) {
        temperature = temperature_;
    }
    void set_rng_seed(unsigned long long rng_seed_) {
        rng_seed = rng_seed_;
    }
    int sample_top_p(std::vector<float>& logits);
    int sample_argmax(std::vector<float>& logits);
    int sample(std::vector<float>& logits);
};

#endif /* SAMPLER_H_ */