
# include "attention_common.hpp";

# include <vector>
# include <cmath>
# include <algorithm>
#include <random>
#include <map>

using namespace std;

/*LOGIC FOR:

1] transpose(m)
2] matmul (mat A, mat B)
3] softmax (m)
4] createMatrix(r, c, val)
5] printMatrix(m)
6] text-2-Embedding ("xyz...", emb_size)
7] embedding-2-text ([[0.2, 0.4...], [...]] => embedding)
8] calculateMemoryKB(m)

*/

vector<vector<float>> AttentionCommon::transpose(const vector<vector<float>> &M){
    if(M.empty()){
        return {};
    }
    std::vector<std::vector<float>> result(M[0].size(), std::vector<float>(M.size()));
    for(int i = 0 ; i < M.size() ; ++i){
        for(int j = 0 ; j < M[0].size() ; ++j){
            result[j][i] = M[i][j];
        }
    }return result;
}

vector<vector<float>> AttentionCommon::matmul(const vector<vector<float>> &A, const vector<vector<float>> &B){
    int A_rows = A.size(), A_cols = A[0].size();     // (p x q)
    int B_rows = B.size(), B_cols = B[0].size();     // (q x r)

    // prod matrix size : (p x r)
    std::vector<std::vector<float>> result(A_rows, std::vector<float>(B_cols, 0.0f));

    for(int i = 0 ; i < result.size() ; ++i){
        for(int j = 0 ; j < result[0].size() ; ++j){
            for(int k = 0 ; k < A_cols ; ++k){
                result[i][j] += (A[i][k] * B[k][j]);
            }
        }
    }return result;
}

std::vector<std::vector<float>> AttentionCommon::softmax(const std::vector<std::vector<float>>& M){
    vector<vector<float>> result = M;

    for(int i = 0 ; i < M.size() ; ++i){
        float max_val = *std::max_element(result.begin(), result.end());

        float sum = 0.0f;
        for(int j = 0 ; j < M[i].size() ; ++j){
            result[i][j] = exp(result[i][j] - max_val);
            sum += result[i][j];
        }

        for(int j = 0 ; j < M[i].size() ; ++j){
            result[i][j] /= sum;
        }
    }
    return result;
}

vector<vector<float>> AttentionCommon::createMatrix(int rows, int cols, float value){
    vector<vector<float>> result(rows, vector<float>(cols, value));
    return result;
}

void AttentionCommon::printMatrix(const vector<vector<float>>& M, const string &name){
    if (!name.empty()) {
        std::cout << name << " (" << M.size() << "x" << (M.empty() ? 0 : M[0].size()) << "):\n";
    }
    for(int i = 0 ; i < M.size() ; ++i){
        for(int j = 0 ; j < M[0].size() ; ++j){
            cout << M[i][j] << " ";
        }
        cout << "\n";
    }cout << endl;
}

vector<vector<float>> AttentionCommon::textToEmbedding(const string &text, int embedding_dim){
    vector<vector<float>> embeddings;
    map<char, int> char_2_idx;
    int idx = 0;

    // developing vocabulary
    for(char &c : text){
        if(char_2_idx.find(c) == char_2_idx.end()){    
            char_2_idx[c] = idx++;
        }
    }

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dist(-0.1f, 0.1f);

    for(char &c : text){
        vector<float> char_embedding(embedding_dim);     
        int char_idx = char_2_idx[c];

        for(int i = 0 ; i < embedding_dim ; ++i){
            char_embedding[i] = sin((char_idx + 1) * (i + 1) * 0.1f) * dist(gen) * 0.1f;
        }
        embeddings.push_back(char_embedding);
    }
    return embeddings;
}

string AttentionCommon::embeddingToText(vector<vector<float>> &embedding){
    string result;
    vector<char> alphabets = {' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'};

    for(const auto &emb : embedding){
        float val = abs(emb[0]);
        char corresponding_idx = static_cast<int>(val * alphabets.size()) % alphabets.size();
        result += alphabets[corresponding_idx];
    }
    return result;
}

size_t AttentionCommon::calculateMemoryKB(vector<vector<float>>& M){
    if(M.empty()) { return 0; }
    size_t total_elements = (M.size() * M[0].size());
    return ((total_elements * sizeof(float)) / 1024);
}
