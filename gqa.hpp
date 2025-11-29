# ifndef GQA_HPP
# define GQA_HPP

#include "attention_common.hpp"
#include <vector>

class GroupedQueryAttention{
    private:
        int num_heads;
        int num_kv_heads;
        int d_model;
        int d_k;
        int d_v;
        int heads_per_group;

        // weight matrics : Grouped k-v projections 
        // each group has a W_q matrice and shared k-v matrices (common for all heads within a group)

        /*
        For the Queries, you have num_heads separate projections.
        For the Keys and Values, you have num_kv_heads separate projections.
        */
        std::vector<std::vector<std::vector<float>>> W_q;
        std::vector<std::vector<std::vector<float>>> W_k;
        std::vector<std::vector<std::vector<float>>> W_v;
        std::vector<std::vector<float>> W_o;
};

# endif