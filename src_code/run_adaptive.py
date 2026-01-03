import json
import os
from bayesian_estimator import BayesianSemanticEntropy

# CONFIG
# Paths to the directories containing the JSON files for each similarity metric
CLUSTERS_DIRECTORY_COSINE_SIM = r"D:/LLM HALL/LLM-Hallucination/data/meanings/cosine_sim"  # Cosine similarity files
CLUSTERS_DIRECTORY_PEARSON_SIM = r'D:/LLM HALL/LLM-Hallucination/data/meanings/pearson_sim'  # Pearson similarity files
CLUSTERS_DIRECTORY_RBF_SIM = r'D:/LLM HALL/LLM-Hallucination/data/meanings/rbf_sim'  # RBF similarity files

# Path to save the results
OUTPUT_DIRECTORY = r'D:\LLM HALL\LLM-Hallucination\result'

# Variance threshold from the paper (Gamma parameter)
VARIANCE_THRESHOLD = 0.005

def run_adaptive_experiment():
    # Initialize Math Engine
    estimator = BayesianSemanticEntropy(alpha=1.0)
    
    # List to store results from all files
    all_results = {}

    # Create a list of directories to process (cosine, pearson, rbf)
    directories = [
        (CLUSTERS_DIRECTORY_COSINE_SIM, 'cosine_sim'),
        (CLUSTERS_DIRECTORY_PEARSON_SIM, 'pearson_sim'),
        (CLUSTERS_DIRECTORY_RBF_SIM, 'rbf_sim')
    ]

    # Iterate through each similarity metric's directory
    for directory_path, metric_name in directories:
        print(f"Processing files in {metric_name} directory...")

        # Initialize results for the current metric
        metric_results = {}

        # Iterate through all JSON files in the current directory
        for filename in os.listdir(directory_path):
            if filename.endswith(".json"):  # Process only .json files
                file_path = os.path.join(directory_path, filename)
                
                try:
                    # Open and load the current JSON file
                    with open(file_path, 'r') as f:
                        data = json.load(f)

                    print(f"Running Adaptive Loop on {filename} ({len(data)} prompts)...")

                    results = {}

                    # Loop through all prompts in the file
                    for prompt_key, clusters in data.items():
                        pool_of_samples = []

                        # Collect samples from clusters
                        for cluster in clusters:
                            m_id = cluster['meaning_id']
                            for i, prob in enumerate(cluster['probabilities']):
                                pool_of_samples.append({
                                    'meaning_id': m_id,
                                    'probability': float(prob),  # Ensure prob is treated as a float
                                    'text': cluster['members'][i]  # Corresponding response text
                                })

                        # Initialize variables for entropy and variance calculation
                        current_samples = []
                        final_entropy = 0.0
                        final_N = 0
                        final_var = 0.0

                        # Run adaptive entropy loop
                        for n in range(len(pool_of_samples)):
                            new_sample = pool_of_samples[n]
                            current_samples.append(new_sample)
                            entropy, variance = estimator.estimate_entropy(current_samples)
                            
                            # If variance is below threshold, stop and store the result
                            if n >= 1 and variance < VARIANCE_THRESHOLD:
                                final_entropy = entropy
                                final_var = variance
                                final_N = n + 1
                                break

                            final_entropy = entropy
                            final_var = variance
                            final_N = n + 1

                        # Store the results for the current prompt
                        results[prompt_key] = {
                            "entropy": final_entropy,
                            "variance": final_var,
                            "samples_used": final_N
                        }

                    # Add the results from this file to the metric's results
                    metric_results[filename] = results
                    print(f"Processed {filename}...")

                except Exception as e:
                    print(f"Error processing {filename}: {e}")

        # Save the results for this metric to a separate file
        output_filename = f"{metric_name}_adaptive_results.json"
        output_filepath = os.path.join(OUTPUT_DIRECTORY, output_filename)
        
        # Create output directory if it doesn't exist
        os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

        # Save all the results for the current metric
        with open(output_filepath, "w") as f:
            json.dump(metric_results, f, indent=2)
            print(f"Done. Results saved to {output_filepath}.")

if __name__ == "__main__":
    run_adaptive_experiment()
