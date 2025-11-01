import numpy as np
import warnings
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import gensim.downloader as api

# --- This is the setup part, you would run this once ---
SIMILARITY_THRESHOLD = 0.85
# It's recommended to load the model only once as it can be time-consuming
word_vectors = api.load("word2vec-google-news-300")
VECTOR_DIM = 300

# This list will store our cluster state.
live_clusters = []
next_cluster_id = 0

def vectorize_sentence(sentence, model):
    """Converts a sentence into a single, normalized vector."""
    words = sentence.lower().split()
    word_vecs = [model[word] for word in words if word in model.key_to_index]
    
    if not word_vecs:
        return np.zeros(VECTOR_DIM)
    
    sentence_vec = np.mean(word_vecs, axis=0)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        sentence_vec = np.nan_to_num(sentence_vec)
        
    return normalize(sentence_vec.reshape(1, -1))[0]


def process_new_response(response_text):
    global next_cluster_id 

    new_vector = vectorize_sentence(response_text, word_vectors)

    if np.all(new_vector == 0):
        print(f"  -> Skipping response (could not vectorize): '{response_text}'")
        return

    if not live_clusters:
        print(f"  -> Creating new cluster 0 with: '{response_text}'")
        live_clusters.append({
            "id": next_cluster_id,
            "members": [response_text],
            "vectors": [new_vector],
            "centroid": new_vector
        })
        next_cluster_id += 1
        return

    centroids = [cluster["centroid"] for cluster in live_clusters]
    # Ensure new_vector is correctly shaped for cosine_similarity
    similarities = cosine_similarity(new_vector.reshape(1, -1), np.array(centroids))[0]
    
    max_similarity = np.max(similarities)
    best_cluster_index = np.argmax(similarities)
    
    if max_similarity >= SIMILARITY_THRESHOLD:
        cluster = live_clusters[best_cluster_index]
        cluster["members"].append(response_text)
        cluster["vectors"].append(new_vector)
        
        cluster["centroid"] = np.mean(cluster["vectors"], axis=0)
        
        print(f"  -> Adding to cluster {cluster['id']} (sim: {max_similarity:.2f}): '{response_text}'")
    else:
        print(f"  -> Creating new cluster {next_cluster_id} (sim: {max_similarity:.2f}): '{response_text}'")
        live_clusters.append({
            "id": next_cluster_id,
            "members": [response_text],
            "vectors": [new_vector],
            "centroid": new_vector
        })
        next_cluster_id += 1



generated_responses = [
    {"text": "The capital of France is Paris."},
    {"text": "Paris is the capital city of France."},
    {"text": "The largest planet in our solar system is Jupiter."},
]

print("Processing initial responses...")
for response in generated_responses:
    process_new_response(response["text"])

print("\n--- Simulating new responses from a model ---\n")

# Now, a new response comes from your LLM
new_response_from_model_1 = "France's capital is the city of Paris."
process_new_response(new_response_from_model_1)

# Another new response
new_response_from_model_2 = "Jupiter is the biggest planet of all."
process_new_response(new_response_from_model_2)

# A completely different response
new_response_from_model_3 = "The sky is blue."
process_new_response(new_response_from_model_3)


# --- Final Output ---
print("\n--- Final Clusters ---")
final_clusters_list = []
for cluster in live_clusters:
    final_clusters_list.append({
        "meaning_id": cluster["id"],
        "members": cluster["members"]
    })

import json
print(json.dumps(final_clusters_list, indent=2))