#include "mha.hpp"
#include "mqa.hpp"
#include "gqa.hpp"

#include <iostream>
#include <vector>

using namespace std;

int main(){
    cout << "=======    ATTENTION MECHANISMS COMPARISION    ======\n\n";

     // Configuration (similar to Mistral 7B)
    const int D_MODEL = 64;
    const int NUM_HEADS = 8;
    const int NUM_KV_HEADS = 2;
    const string TEXT = "She has a nice rack";

    cout << "Configuration:\n";
    cout << "Query Heads: " << NUM_HEADS << "\n";
    cout << "KV Heads (GQA): " << NUM_KV_HEADS << "\n";
    cout << "Model Dimension: " << D_MODEL << "\n";
    cout << "Input Text: " << TEXT << "\"\n\n";

    auto textEmbedding = AttentionCommon::textToEmbedding(TEXT, D_MODEL);
    cout << "Embedding size : " << "[" << textEmbedding.size() << textEmbedding[0].size() << "]" << endl;

    cout << "MULTI-HEAD ATTENTION (MHA)\n";
    MultiHeadAttention mha(NUM_HEADS, D_MODEL);
    mha.printMemoryUsage(textEmbedding);
    auto mha_output = mha.forward(textEmbedding);
    cout << "MHA Output shape: " << mha_output.size() << " x " << mha_output[0].size() << "\n";

    cout << "MULTI-QUERY ATTENTION (MHA)\n";
    MultiQueryAttention mqa(NUM_HEADS, D_MODEL);
    mqa.printMemoryUsage(textEmbedding);
    auto mqa_output = mqa.forward(textEmbedding);
    cout << "MQA Output shape: " << mqa_output.size() << " x " << mqa_output[0].size() << "\n";

    cout << "GROUPED-QUERY ATTENTION (MHA)\n";
    GroupedQueryAttention gqa(NUM_HEADS, NUM_KV_HEADS, D_MODEL);
    gqa.printMemoryUsage(textEmbedding);
    auto gqa_output = gqa.forward(textEmbedding);
    cout << "GQA Output shape: " << gqa_output.size() << " x " << gqa_output[0].size() << "\n";


    // Demonstrate text reconstruction (simplified)
    cout << "\n=== Text Reconstruction Demo ===\n";
    cout << "Original text: " << TEXT << "\n";
    
    auto reconstructed_mha = AttentionCommon::embeddingToText(mha_output);
    cout << "MHA reconstructed: " << reconstructed_mha << "\n";
    
    auto reconstructed_mqa = AttentionCommon::embeddingToText(mqa_output);
    cout << "MQA reconstructed: " << reconstructed_mqa << "\n";
    
    auto reconstructed_gqa = AttentionCommon::embeddingToText(gqa_output);
    cout << "GQA reconstructed: " << reconstructed_gqa << "\n";
}