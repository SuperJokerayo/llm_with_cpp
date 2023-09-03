#include "sampler.h"

#include <utility>
#include <algorithm>
#include <numeric>
#include <cmath>

static void softmax(std::vector<float>& x) {
    int size = x.size();
    auto max_val = *std::max_element(x.begin(), x.end());
    auto sum = std::accumulate(x.begin(), x.end(), 0.0f, [max_val](float a, float b) {
        return a + std::exp(b - max_val);
    });
    std::for_each(x.begin(), x.end(), [sum](int a){return a / sum;});
}

unsigned int random_u32(unsigned long long *state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

float random_f32(unsigned long long *state) {
    return (random_u32(state) >> 8) / 16777216.0f;
}

int Sampler::sample_top_p(std::vector<float>& logits) {
    int n = logits.size();
    std::vector<std::pair<float, int>> candidates;
    const float cutoff = (1.0f - p) / (n - 1);

    for(int i = 0; i < n; i++) {
        if(logits[i] > cutoff) {
            candidates.push_back({logits[i], i});
        }
    }

    std::sort(candidates.begin(), candidates.end(), [](auto& a, auto& b) {
        return a.first > b.first;
    }); 

    float cumulative_prob = 0.0f;
    int last_idx = candidates.size() - 1;
    for(int i = 0; i < candidates.size(); i++) {
        cumulative_prob += candidates[i].first;
        if(cumulative_prob > p) {
            last_idx = i;
            break;
        }
    }
    
    float coin = random_f32(&rng_seed);
    float r = coin * cumulative_prob;
    float cdf = 0.0f;

    for(int i = 0; i <= last_idx; i++) {
        cdf += candidates[i].first;
        if(cdf > r) {
            return candidates[i].second;
        }
    }
    return candidates[last_idx].second;
}

int Sampler::sample_argmax(std::vector<float>& logits) {
    return std::max_element(logits.begin(), logits.end()) - logits.begin();
}

int Sampler::sample(std::vector<float>& logits) {
    int next;
    if(temperature < 1e-5) {
        next = sample_argmax(logits);
    } else {
        std::for_each(logits.begin(), logits.end(), [this](float& logit) {
            logit /= temperature;
        });

        softmax(logits);
        
        next = sample_top_p(logits);
    }
    return next;
}