# ifndef MHA_HPP
# define MHA_HPP 

# include "attention_common.hpp"

# include <vector>

class MultiHeadAttention{
    private:
        int d_model;
        int num_heads;
        int d_k;   // (also d_q) (used in softmax as sqrt(d_k))
        int d_v; 

        // 3D shapes of Q, K, V weights : [N_heads, dim_model, dim_head(dim_model / N_heads)]
        std::vector<std::vector<std::vector<float>>> W_q;
        std::vector<std::vector<std::vector<float>>> W_k;
        std::vector<std::vector<std::vector<float>>> W_v;

        // 2D shape of output weight : [D_model, D_model] (D_model => N_heads * dim_head)
        std::vector<std::vector<float>> W_o;
    
    public:
    // constructor
        MultiHeadAttention(int num_heads, int d_model);

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