import numpy as np
import json
import os
import gensim.downloader as api
from sklearn.preprocessing import normalize
from Load_model_output import path1, path2, path3, path4, path5, path6, path7, path8, path9, path10, path11, path12, path13, create_cleaned_dataset


SIMILARITY_THRESHOLD = 0.85
VECTOR_DIM = 300

# Load pre-trained Word2Vec model
word_vectors = api.load("word2vec-google-news-300")
print("Model loaded successfully.")

# Vectorize a sentence
def vectorize_sentence(sentence, model):
    words = sentence.lower().split()
    word_vecs = [model[word] for word in words if word in model.key_to_index]
    
    if not word_vecs:
        return np.zeros(VECTOR_DIM)
    
    sentence_vec = np.mean(word_vecs, axis=0)
    sentence_vec = np.nan_to_num(sentence_vec)
    
    return normalize(sentence_vec.reshape(1, -1))[0]

# Define similarity functions
def cosine_similarity(A, B):
    return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))

def euclidean_distance(A, B):
    return np.linalg.norm(A - B)

def manhattan_distance(A, B):
    return np.sum(np.abs(A - B))

def jaccard_similarity(A, B):
    intersection = np.sum(np.minimum(A, B))
    union = np.sum(np.maximum(A, B))
    return intersection / union

def dot_product(A, B):
    return np.dot(A, B)

def pearson_correlation(A, B):
    return np.corrcoef(A, B)[0, 1]

def rbf_kernel(A, B, gamma=0.1):
    return np.exp(-gamma * np.linalg.norm(A - B) ** 2)

def dice_coefficient(A, B):
    intersection = np.sum(np.minimum(A, B))
    return 2 * intersection / (np.sum(A) + np.sum(B))

# Cluster responses based on similarity
def cluster_responses(list_of_responses, model, threshold, similarity_metric):
    response_vectors = [vectorize_sentence(resp, model) for resp in list_of_responses]
    cluster = {}  # {cluster_id: {rep_idx: [member_indices...]}}
    cluster_id = 1

    for i, vec in enumerate(response_vectors):
        if not cluster:
            cluster[cluster_id] = {i: [i]}
            cluster_id += 1
            continue

        placed = False
        for j in list(cluster.keys()):
            rep_idx = next(iter(cluster[j].keys()))
            rep_vec = response_vectors[rep_idx]
            if similarity_metric == 'cosine':
                sim = cosine_similarity(vec, rep_vec)
            elif similarity_metric == 'euclidean':
                sim = -euclidean_distance(vec, rep_vec)  # Negative for reverse similarity
            elif similarity_metric == 'manhattan':
                sim = -manhattan_distance(vec, rep_vec)  # Negative for reverse similarity
            elif similarity_metric == 'jaccard':
                sim = jaccard_similarity(vec, rep_vec)
            elif similarity_metric == 'dot':
                sim = dot_product(vec, rep_vec)
            elif similarity_metric == 'pearson':
                sim = pearson_correlation(vec, rep_vec)
            elif similarity_metric == 'rbf':
                sim = rbf_kernel(vec, rep_vec)
            elif similarity_metric == 'dice':
                sim = dice_coefficient(vec, rep_vec)
            
            if sim >= threshold:
                cluster[j][rep_idx].append(i)
                placed = True
                break

        if not placed:
            cluster[cluster_id] = {i: [i]}
            cluster_id += 1

    # Assign cluster ids
    cluster_ids = [-1] * len(list_of_responses)
    for cid, m in cluster.items():
        rep_idx = next(iter(m.keys()))
        for idx in m[rep_idx]:
            cluster_ids[idx] = cid

    return cluster_ids

# Save clustering results for each metric in different directories
if __name__ == "__main__":
    similarity_metrics = ['cosine', 'euclidean', 'manhattan', 'jaccard', 'dot', 'pearson', 'rbf', 'dice']
    
    # Create output directories for each similarity metric
    for metric in similarity_metrics:
        output_dir = f"D:/LLM-Hallucination/data/meanings/{metric}_sim"
        os.makedirs(output_dir, exist_ok=True)

    # all_paths = [path1, path2, path3, path4, path5, path6, path7, path8, path9, path10, path11, path12, path13]
    all_paths = [path1]

    for file_path in all_paths:
        print(f"\n========================================================")
        print(f"STARTING PROCESSING FOR FILE: {os.path.basename(file_path)}")
        print(f"========================================================")

        cleaned_data = create_cleaned_dataset(file_path)
        final_output = {}

        for i in range(len(cleaned_data.questions)):
            prompt_key = f"prompt_{i}: {cleaned_data.questions[i]}"
            responses = cleaned_data.response_list[i]

            for metric in similarity_metrics:
                print(f"\n--- Clustering responses for prompt {i} using {metric} similarity ---")
                assigned_ids = cluster_responses(responses, word_vectors, SIMILARITY_THRESHOLD, metric)

                prompt_clusters = {}
                for j, response_text in enumerate(responses):
                    cluster_id = assigned_ids[j]
                    if cluster_id not in prompt_clusters:
                        prompt_clusters[cluster_id] = {"meaning_id": cluster_id, "members": []}
                    prompt_clusters[cluster_id]["members"].append(response_text)

                final_output[prompt_key] = list(prompt_clusters.values())

                # Save the output for the current metric
                base_file = os.path.basename(file_path)
                output_name = base_file.replace('.pickle', f'_{metric}_clusters.json')
                output_filename = f"D:/LLM-Hallucination/data/meanings/{metric}_sim/{output_name}"

                with open(output_filename, 'w') as f:
                    json.dump(final_output, f, indent=2)

                print(f"\n Final clusters for {base_file} saved to '{output_filename}'")
