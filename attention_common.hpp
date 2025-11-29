# ifndef ATTENTION_COMMON_HPP

# define ATTENTION_COMMON_HPP

# include <cmath>
# include <algorithm>
# include <vector>
# include <numeric>
# include <iostream>
# include <memory>

class AttentionCommon{
    public:
        static std::vector<std::vector<float>> transpose(const std::vector<std::vector<float>> &M);
        
        static std::vector<std::vector<float>> matmul(const std::vector<std::vector<float>> &A, const std::vector<std::vector<float>> &B);

        static std::vector<std::vector<float>> softmax(const std::vector<std::vector<float>> &M);

        static std::vector<std::vector<float>> createMatrix(int rows, int cols, float value = 0.0f);

        static void printMatrix(const std::vector<std::vector<float>> &M, const std::string &name = "");

        static std::vector<std::vector<float>> textToEmbedding(const std::string &text, const int embedding_dim);

        static std::string embeddingToText(const std::vector<std::vector<float>> &embedding);

        // to calculate memory
        static size_t calculateMemoryKB(const std::vector<std::vector<float>> &M);
}; 


# endif