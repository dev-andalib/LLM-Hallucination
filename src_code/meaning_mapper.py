import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import gensim.downloader as api
from Load_model_output import create_cleaned_dataset, path1, path2, path3, path4, path5, path6, path7, path8, path9, path10, path11, path12, path13
import os



SIMILARITY_THRESHOLD = 0.85 
VECTOR_DIM = 300 # Vector dimension for the chosen model


word_vectors = api.load("word2vec-google-news-300")
print("Model loaded successfully.")




def vectorize_sentence(sentence, model):
    words = sentence.lower().split()
    word_vecs = [model[word] for word in words if word in model.key_to_index]
    
    
    if not word_vecs:
        return np.zeros(VECTOR_DIM)
    
    # Calculate the mean of the word vectors
    sentence_vec = np.mean(word_vecs, axis=0)
    
    # Handle potential NaN values and ensure the vector is normalized (has a length of 1)
    sentence_vec = np.nan_to_num(sentence_vec)
        
    return normalize(sentence_vec.reshape(1, -1))[0]




def cluster_responses(list_of_responses, model, threshold):
    response_vectors = [vectorize_sentence(resp, model) for resp in list_of_responses]

    cluster_ids = [-1] * len(list_of_responses)
    next_cluster_id = 0
    
    for i in range(len(list_of_responses)):
        # If this response has not been assigned to a cluster 
        if cluster_ids[i] == -1:
            # start a new cluster.
            cluster_ids[i] = next_cluster_id
            
            #find all other un-clustered responses
            for j in range(i + 1, len(list_of_responses)):
                if cluster_ids[j] == -1:
                    vec_i = response_vectors[i].reshape(1, -1)
                    vec_j = response_vectors[j].reshape(1, -1)
                    
                    similarity = cosine_similarity(vec_i, vec_j)[0][0]
                    
                    if similarity > threshold:
                        # If  similar, assign the other response to the same cluster
                        cluster_ids[j] = next_cluster_id
            
            # Move to the next cluster ID
            next_cluster_id += 1
            
    return cluster_ids


if __name__ == "__main__":

    all_paths = [path1, path2, path3, path4, path5, path6, 
                 path7, path8, path9, 
                 path10, path11, path12, path13]
    

    for file_path in all_paths:
        print(f"\n========================================================")
        print(f"STARTING PROCESSING FOR FILE: {os.path.basename(file_path)}")
        print(f"========================================================")

        cleaned_data = create_cleaned_dataset(file_path)
        final_output = {}
        
        for i in range(len(cleaned_data.questions)):
            prompt_key = f"prompt_{i}: {cleaned_data.questions[i]}"
            responses = cleaned_data.response_list[i]

            print(f"\n--- Clustering responses for prompt {i} ---")
            assigned_ids = cluster_responses(responses, word_vectors, SIMILARITY_THRESHOLD)

            prompt_clusters = {}

            for j, response_text in enumerate(responses):
                cluster_id = assigned_ids[j]
                if cluster_id not in prompt_clusters:
                    prompt_clusters[cluster_id] = {
                        "meaning_id": cluster_id,
                        "members": []
                    }
                prompt_clusters[cluster_id]["members"].append(response_text)


            final_output[prompt_key] = list(prompt_clusters.values())



        base_file = os.path.basename(file_path)
        output_name = base_file.replace('.pickle', '_clusters.json')
        output_filename = f"D:/LLM-Hallucination/data/meanings/{output_name}"
        with open(output_filename, 'w') as f:
                json.dump(final_output, f, indent=2)

        print(f"\n Final clusters for {base_file} saved to '{output_filename}'")