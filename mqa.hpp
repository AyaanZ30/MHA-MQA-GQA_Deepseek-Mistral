# ifndef MQA_HPP
# define MQA_HPP

#include "attention_common.hpp"
#include <vector>
class MultiQueryAttention{
    private:
        int num_heads;
        int d_model;
        int d_k;       // (same for d_q)
        int d_v;

        // 'H' heads  --> divided into 'G' groups (each group has its own Query matrice but shared K-V matrices)
        std::vector<std::vector<std::vector<float>>> W_q;    // Multiple Q projections
        std::vector<std::vector<float>> W_k;                // Single K projection
        std::vector<std::vector<float>> W_v;                // Single V projection
        std::vector<std::vector<float>> W_o;                // Output projection
    
    public:
        MultiQueryAttention(int num_heads, int d_model);

        std::vector<std::vector<float>> forward(const std::vector<std::vector<float>>& X);

        // Memory usage analysis
        void printMemoryUsage(const std::vector<std::vector<float>> &x);

        // Get attention weights for analysis
        std::vector<std::vector<std::vector<float>>> getAttentionWeights(const std::vector<std::vector<float>>& X);

    private:
        void initializeWeights();
        std::vector<std::vector<float>> splitHeads(const std::vector<std::vector<float>>& X, int head_size);
        std::vector<std::vector<float>> combineHeads(const std::vector<std::vector<float>>& X);
};

# endif
