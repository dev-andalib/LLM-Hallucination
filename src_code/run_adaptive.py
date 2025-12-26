import json

from bayesian_estimator import BayesianSemanticEntropy

# CONFIG
CLUSTERS_FILE = None
VARIANCE_THRESHOLD = 0.005  # The "Gamma" parameter from the paper

def run_adaptive_experiment():
    # Initialize Math Engine
    estimator = BayesianSemanticEntropy(alpha=1.0)
    
    with open(CLUSTERS_FILE, 'r') as f:
        data = json.load(f)
        
    results = {}
    
    print(f"Running Adaptive Loop on {len(data)} prompts...")

    for prompt_key, clusters in data.items():
        
        
        pool_of_samples = []
        
        for cluster in clusters:
            m_id = cluster['meaning_id']
            
            for i, prob in enumerate(cluster['probabilities']):
                pool_of_samples.append({
                    'meaning_id': m_id,
                    'probability': prob,
                    'text': cluster['members'][i] 
                })
        
        current_samples = []
        final_entropy = 0.0
        final_N = 0
        final_var = 0.0
        
        
        for n in range(len(pool_of_samples)):
            new_sample = pool_of_samples[n]
            current_samples.append(new_sample)
            entropy, variance = estimator.estimate_entropy(current_samples)
            
            if n >= 1 and variance < VARIANCE_THRESHOLD:
                final_entropy = entropy
                final_var = variance
                final_N = n + 1
                break
            
            
            final_entropy = entropy
            final_var = variance
            final_N = n + 1

        
        results[prompt_key] = {
            "entropy": final_entropy,
            "variance": final_var,
            "samples_used": final_N
        }
        
        if len(results) % 10 == 0:
            print(f"Processed {len(results)} prompts...")

    
    with open("adaptive_results.json", "w") as f:
        json.dump(results, f, indent=2)
        print("Done.")

if __name__ == "__main__":
    run_adaptive_experiment()