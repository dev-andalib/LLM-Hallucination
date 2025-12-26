import json
import os
import glob
from bayesian_estimator import BayesianSemanticEntropy

# CONFIG
CLUSTERS_FILE = "D:/LLM-Hallucination/data/meanings/cosine_sim/nnnp-Llama-2-7b-chat--nq--results_cosine_clusters.json"
VARIANCE_THRESHOLD = 0.005  # The "Gamma" parameter from the paper

def run_adaptive_experiment():
    # Initialize Math Engine
    estimator = BayesianSemanticEntropy(alpha=1.0)
    
    with open(CLUSTERS_FILE, 'r') as f:
        data = json.load(f)
        
    results = {}
    
    print(f"Running Adaptive Loop on {len(data)} prompts...")

    for prompt_key, clusters in data.items():
        
        # 1. Flatten the clusters into a pool of available samples
        # We need to reconstruct the list of samples as if they are coming in one by one.
        # Ideally, we sort them by 'original_index' to simulate the real generation order.
        pool_of_samples = []
        
        for cluster in clusters:
            m_id = cluster['meaning_id']
            # We assume your JSON saved a list of probability values
            # AND ideally the original_index. 
            # If you didn't save original_index, we just append them (random order simulation)
            for i, prob in enumerate(cluster['probabilities']):
                pool_of_samples.append({
                    'meaning_id': m_id,
                    'probability': prob,
                    'text': cluster['members'][i] # Optional, just for tracking
                })
        
        # Sort by original index if you have it, otherwise shuffling mimics random generation
        # import random; random.shuffle(pool_of_samples) 
        
        current_samples = []
        final_entropy = 0.0
        final_N = 0
        final_var = 0.0
        
        # --- THE ADAPTIVE LOOP ---
        # Loop from N=1 to Total Available
        for n in range(len(pool_of_samples)):
            
            # Add ONE new sample to our "observed" set
            new_sample = pool_of_samples[n]
            current_samples.append(new_sample)
            
            # Calculate Stats
            entropy, variance = estimator.estimate_entropy(current_samples)
            
            # Stopping Rule (Algorithm 1)
            # usually we force at least N=2 to avoid 0-variance bugs with 1 sample
            if n >= 1 and variance < VARIANCE_THRESHOLD:
                final_entropy = entropy
                final_var = variance
                final_N = n + 1
                break # STOP!
            
            # If we run out of samples, we just take the last result
            final_entropy = entropy
            final_var = variance
            final_N = n + 1

        # Store result
        results[prompt_key] = {
            "entropy": final_entropy,
            "variance": final_var,
            "samples_used": final_N
        }
        
        if len(results) % 10 == 0:
            print(f"Processed {len(results)} prompts...")

    # Save
    with open("adaptive_results.json", "w") as f:
        json.dump(results, f, indent=2)
        print("Done.")

if __name__ == "__main__":
    run_adaptive_experiment()