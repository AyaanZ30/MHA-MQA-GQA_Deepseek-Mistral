# include "gqa.hpp"
# include "attention_common.hpp"

# include <vector>
# include <random>

using namespace std;

GroupedQueryAttention::GroupedQueryAttention(int num_heads, int num_kv_heads, int d_model) : num_heads(num_heads), num_kv_heads(num_kv_heads), d_model(d_model)
{
    // G => Group size = (num_heads / num of kv heads) heads per group
    if((num_heads % num_kv_heads) != 0){
        throw invalid_argument("num_heads must be divisible by num_kv_heads");
    }

    heads_per_group = (num_heads / num_kv_heads);
    d_k = (d_model / num_heads);
    d_v = (d_model / num_heads);
    initializeWeights();
}

void GroupedQueryAttention::initializeWeights(){
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dist(-0.1f, 0.1f);

    W_q.resize(num_heads, vector<vector<float>>(d_model, vector<float>(d_k)));
    W_k.resize(num_kv_heads, vector<vector<float>>(d_model, vector<float>(d_k)));
    W_v.resize(num_kv_heads, vector<vector<float>>(d_model, vector<float>(d_v)));

    for(int h = 0 ; h < num_heads ; ++h){
        for(int i = 0 ; i < d_model ; ++i){
            for(int j = 0 ; j < d_k ; ++j){
                W_q[h][i][j] = dist(gen);
            }
        }
    }
    for(int kvh = 0 ; kvh < num_kv_heads ; ++kvh){
        for(int i = 0 ; i < d_model ; ++i){
            for(int j = 0 ; j < d_k ; ++j){
                W_k[kvh][i][j] = dist(gen);
            }
        }
    }
    for(int kvh = 0; kvh < num_kv_heads; ++kvh) {
        for(int i = 0; i < d_model; ++i) {
            for(int k = 0; k < d_v; ++k) { 
                W_v[kvh][i][k] = dist(gen);
            }
        }
    }

    W_o = vector<vector<float>>(d_model, vector<float>(d_model));
    for (int i = 0; i < d_model; ++i) {
        for (int j = 0; j < d_model; ++j) {
            W_o[i][j] = dist(gen);
        }
    }
}

vector<vector<float>> GroupedQueryAttention::forward(const vector<vector<float>> &X){
    int seq_len = X.size();
    vector<vector<float>> output(seq_len, vector<float>(d_model, 0.0f));
    vector<vector<vector<float>>> head_outputs(num_heads);

    // Pre-compute K and V matrices for each group
    vector<vector<vector<float>>> K_groups, V_groups;
    for(int kvh = 0 ; kvh < num_kv_heads ; ++kvh){
        K_groups.push_back(AttentionCommon::matmul(X, W_k[kvh]));
        V_groups.push_back(AttentionCommon::matmul(X, W_v[kvh]));
    }

    for(int h = 0 ; h < num_heads ; ++h){
        int curr_group_idx = (h / heads_per_group);

        auto Q = AttentionCommon::matmul(X, W_q[h]);
        auto& K = K_groups[curr_group_idx];
        auto& V = V_groups[curr_group_idx];

        auto K_t = AttentionCommon::transpose(K);
        auto scores = AttentionCommon::matmul(Q, K_t);

        float scale = 1.0f / std::sqrt(static_cast<float>(d_k));
        for (auto& row : scores) {
            for (auto& val : row) {
                val *= scale;
            }
        }
        
        // Apply softmax
        auto attention_weights = AttentionCommon::softmax(scores);
        
        // Apply attention to values
        head_outputs[h] = AttentionCommon::matmul(attention_weights, V);
    }

    for(int h = 0 ; h < num_heads ; ++h){
        for(int i = 0 ; i < seq_len ; ++i){
            for(int j = 0 ; j < d_v ; ++j){
                output[i][(h * d_v) + j] = head_outputs[h][i][j];
            }
        }
    }

    // final linear projection
    output = AttentionCommon::matmul(output, W_o);

    return output;
}

void GroupedQueryAttention::printMemoryUsage(const vector<vector<float>>& X) {
    size_t kv_cache_memory = 0;
    
    // Calculate KV cache memory for groups
    for (int g = 0; g < num_kv_heads; ++g) {
        auto K = AttentionCommon::matmul(X, W_k[g]);
        auto V = AttentionCommon::matmul(X, W_v[g]);
        kv_cache_memory += AttentionCommon::calculateMemoryKB(K) + AttentionCommon::calculateMemoryKB(V);
    }
    
    cout << "=== Grouped Query Attention Memory Usage ===\n";
    cout << "Number of query heads: " << num_heads << "\n";
    cout << "Number of KV heads (groups): " << num_kv_heads << "\n";
    cout << "Heads per group: " << heads_per_group << "\n";
    cout << "Model dimension: " << d_model << "\n";
    cout << "Head dimension (d_k): " << d_k << "\n";
    cout << "KV Cache Memory: " << kv_cache_memory << " KB\n";
    cout << "Total Parameters: " << (num_heads * d_model * d_k + 2 * num_kv_heads * d_model * d_k + d_model * d_model) << "\n";
    cout << "============================================\n\n";
}

vector<vector<vector<float>>> GroupedQueryAttention::getAttentionWeights(const vector<vector<float>> &X)
{
    vector<vector<vector<float>>> all_attention_weights;

    // precompute K for each group
    vector<vector<vector<float>>> K_groups;
    for(int g = 0 ; g < num_kv_heads ; ++g){
        K_groups.push_back(AttentionCommon::matmul(X, W_k[g]));
    }

    for (int h = 0; h < num_heads; ++h) {
        int group_idx = h / heads_per_group;
        auto Q = AttentionCommon::matmul(X, W_q[h]);
        auto& K = K_groups[group_idx];
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