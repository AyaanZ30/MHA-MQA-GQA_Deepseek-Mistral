# include "mha.hpp"
# include "attention_common.hpp"

# include <vector>
# include <random>

using namespace std;

MultiHeadAttention::MultiHeadAttention(int num_heads, int d_model) 
: num_heads(num_heads), d_model(d_model){
    d_k = (int)(d_model / num_heads);            // (d_k same as d_q)
    d_v = (int)(d_model / num_heads);
    initializeWeights();
}

void MultiHeadAttention::initializeWeights(){
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dist(-0.1f, 0.1f);

    /* initializing Q, K, V for each head (total => num_heads)

    num_heads : seperate w_q, w_v and w_k matrix for each head
    d_model : input dimension of the data (or embedding dimension)
    d_k : output dimension of data (also known as d_head / query / key dimension)

    ex : W_q[8][512][64] : Each attention head i (from i=0 to i=7) has its own distinct W_q(i) matrix of size (512 x 64)
    */

    W_q.resize(num_heads, vector<vector<float>>(d_model, vector<float>(d_k)));
    W_k.resize(num_heads, vector<vector<float>>(d_model, vector<float>(d_k)));
    W_v.resize(num_heads, vector<vector<float>>(d_model, vector<float>(d_v)));

    for(int h = 0 ; h < num_heads ; ++h){
        for(int i = 0 ; i < d_model ; ++i){
            for(int j = 0 ; j < d_k ; ++j){
                W_q[h][i][j] = dist(gen);
                W_k[h][i][j] = dist(gen);
            }
            for(int z = 0 ; z < d_v ; ++z){
                W_v[h][i][z] = dist(gen);
            }
        }
        
    }

    /* initialize output weights ([concat(heads), d_model])

        size(W_o) < [d_model, d_model] (always!)
        (as first param => concat(h1, h2, --- h(num_heads)) => num_heads * d_v => concatenated head dimension)
        (num_heads * d_v) < d_model (always!)
    */
    W_o = vector<vector<float>>(d_model, vector<float>(d_model));
    for(int i = 0 ; i < d_model ; ++i){
        for(int j = 0 ; j < d_model ; ++j){
            W_o[i][j] = dist(gen);
        }
    }
}

vector<vector<float>> MultiHeadAttention::forward(const vector<vector<float>> &X){
    int seq_len = X.size();

    vector<vector<float>> output(seq_len, vector<float>(d_model, 0.0f));
    vector<vector<vector<float>>> head_outputs(num_heads);

    for(int h = 0 ; h < num_heads ; ++h){
        auto Q = AttentionCommon::matmul(X, W_q[h]);
        auto K = AttentionCommon::matmul(X, W_k[h]);
        auto V = AttentionCommon::matmul(X, W_v[h]);

        auto K_t = AttentionCommon::transpose(K);
        auto scores = AttentionCommon::matmul(Q, K_t);

        float scale = 1.0f / sqrt(static_cast<float>(d_k));
        for(auto &row : scores){
            for(auto &val : row){
                val *= (scale);              // normalized for stability
            }
        }

        auto attention_weights = AttentionCommon::softmax(scores);

        // apply attention to values with the attention context vector from above step
        auto head_result = AttentionCommon::matmul(attention_weights, V);
        head_outputs[h] = head_result;
    }

    // concatenate outputs of each head to get final context vector
    for(int h = 0 ; h < num_heads ; ++h){
        for(int i = 0 ; i < seq_len ; ++i){
            for(int j = 0 ; j < d_v ; ++j){
                output[i][h * d_v + j] = head_outputs[h][i][j];
            }
        }
    } 
    output = AttentionCommon::matmul(output, W_o);
    return output;
}

void MultiHeadAttention::printMemoryUsage(const vector<vector<float>> &X){
    size_t kv_cache_memory = 0.0;

    for(int h = 0 ; h < num_heads ; ++h){
        auto K = AttentionCommon::matmul(X, W_k[h]);
        auto V = AttentionCommon::matmul(X, W_v[h]);
        kv_cache_memory += AttentionCommon::calculateMemoryKB(K) + AttentionCommon::calculateMemoryKB(V);
    }
    std::cout << "=== Multi-Head Attention Memory Usage ===\n";
    std::cout << "Number of heads: " << num_heads << "\n";
    std::cout << "Model dimension: " << d_model << "\n";
    std::cout << "Head dimension (d_k): " << d_k << "\n";
    std::cout << "KV Cache Memory: " << kv_cache_memory << " KB\n";
    std::cout << "Total Parameters: " << ((num_heads * 3 * d_model * d_k) + (d_model * d_model)) << "\n";
    std::cout << "=========================================\n\n";
}

vector<vector<vector<float>>> MultiHeadAttention::getAttentionWeights(const vector<vector<float>>& X){
    vector<vector<vector<float>>> all_attention_weights;

    for(int h = 0 ; h < num_heads ; ++h){
        auto Q = AttentionCommon::matmul(X, W_q[h]);
        auto K = AttentionCommon::matmul(X, W_k[h]);
        auto K_t = AttentionCommon::transpose(K);
        auto scores = AttentionCommon::matmul(Q, K_t);
        
        float scale = 1.0f / std::sqrt(static_cast<float>(d_k));
        for (auto& row : scores) {
            for (auto& val : row) {
                val *= scale;
            }
        }
        
        auto attention_weights = AttentionCommon::softmax(scores);
        all_attention_weights.push_back(attention_weights);
    }
    return all_attention_weights;
} 