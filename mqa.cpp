# include "mqa.hpp"
# include "attention_common.hpp"

# include <vector>
# include <random>

using namespace std;

MultiQueryAttention::MultiQueryAttention(int num_heads, int d_model)
: num_heads(num_heads), d_model(d_model){
    d_k = (d_model / num_heads);
    d_v = (d_model / num_heads);
    initializeWeights();
}

void MultiQueryAttention::initializeWeights(){
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dist(-0.1f, 0.1f);

    // only single W_k and W_v for all heads
    // Multiple Query projections (W_q)
    W_q.resize(num_heads, vector<vector<float>>(d_model, vector<float>(d_k)));
    W_k = vector<vector<float>>(d_model, vector<float>(d_k));
    W_v = vector<vector<float>>(d_model, vector<float>(d_v));

    for(int h = 0 ; h < num_heads ; ++h){
        for(int i = 0 ; i < d_model ; ++i){
            for(int j = 0 ; j < d_k ; ++j){
                W_q[h][i][j] = dist(gen);
                W_k[i][j] = dist(gen);
            }
        }
    }
    for (int i = 0; i < d_model; ++i) {
        for (int j = 0; j < d_v; ++j) {
            W_v[i][j] = dist(gen);
        }
    }

    // init output weights (dim[0] -> always less than d_model (seq_len < d_model))
    W_o = vector<vector<float>>(d_model, vector<float>(d_model));
    for (int i = 0; i < d_model; ++i) {
        for (int j = 0; j < d_model; ++j) {
            W_o[i][j] = dist(gen);
        }
    }
}

vector<vector<float>> MultiQueryAttention::forward(const vector<vector<float>> &X){
    int seq_len = X.size();
    vector<vector<float>> output(seq_len, vector<float>(d_model, 0.0f));
    vector<vector<vector<float>>> head_outputs(num_heads);

    auto K = AttentionCommon::matmul(X, W_k);
    auto V = AttentionCommon::matmul(X, W_v);
    for(int h = 0 ; h < num_heads ; ++h){
        auto Q = AttentionCommon::matmul(X, W_q[h]);

        auto K_t = AttentionCommon::transpose(K);
        auto scores = AttentionCommon::matmul(Q, K_t);

        float scale = 1.0f / std::sqrt(static_cast<float>(d_k));
        for (auto& row : scores) {
            for (auto& val : row) {
                val *= scale;
            }
        }
        
        auto attention_weights = AttentionCommon::softmax(scores);
        head_outputs[h] = AttentionCommon::matmul(attention_weights, V);
    }

    // concatenating heads
    for(int h = 0 ; h < num_heads ; ++h){
        for(int i = 0 ; i < seq_len ; ++i){
            for(int j = 0 ; j < d_model ; ++j){
                output[i][h * d_v + j] = head_outputs[h][i][j];
            }
        }
    }

    // final Linear projection
    output = AttentionCommon::matmul(output, W_o);
    return output;
}

void MultiQueryAttention::printMemoryUsage(const vector<vector<float>>& X){
    // Single K and V projections
    auto K = AttentionCommon::matmul(X, W_k);
    auto V = AttentionCommon::matmul(X, W_v);
    
    size_t kv_cache_memory = (AttentionCommon::calculateMemoryKB(K) + AttentionCommon::calculateMemoryKB(V));
    
    cout << "=== Multi-Query Attention Memory Usage ===\n";
    cout << "Number of heads: " << num_heads << "\n";
    cout << "Model dimension: " << d_model << "\n";
    cout << "Head dimension (d_k): " << d_k << "\n";
    cout << "KV Cache Memory: " << kv_cache_memory << " KB\n";
    cout << "Total Parameters: " << (num_heads * d_model * d_k + 2 * d_model * d_k + d_model * d_model) << "\n";
    cout << "==========================================\n\n";
}

vector<vector<vector<float>>> MultiQueryAttention::getAttentionWeights(const std::vector<std::vector<float>>& X) {
    vector<vector<vector<float>>> all_attention_weights;
    auto K = AttentionCommon::matmul(X, W_k);
    auto K_t = AttentionCommon::transpose(K);
    
    for (int h = 0; h < num_heads; ++h) {
        auto Q = AttentionCommon::matmul(X, W_q[h]);
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