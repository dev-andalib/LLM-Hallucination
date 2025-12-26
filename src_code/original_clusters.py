import numpy as np
import json
import os
from Load_model_output import path1, path2, path3, path4, path5, path6, path7, path8, path9, path10, path11, path12, path13, create_cleaned_dataset


OUTPUT_DIR = "D:/LLM-Hallucination/data/original_authors"
os.makedirs(OUTPUT_DIR, exist_ok=True)

if __name__ == "__main__":
    # Same list of paths as your meaning_mapper
    all_paths = [path1, path2, path3, path4, path5, path6, path7, path8, path9, path10, path11, path12, path13]

    for file_path in all_paths:
        base_file = os.path.basename(file_path)
        print(f"\n========================================================")
        print(f"EXTRACTING ORIGINAL CLUSTERS: {base_file}")
        print(f"========================================================")

        cleaned_data = create_cleaned_dataset(file_path)
        final_output = {}

        # Loop through questions
        for i in range(len(cleaned_data.questions)):
            prompt_key = f"prompt_{i}: {cleaned_data.questions[i]}"
            
            responses = cleaned_data.response_list[i]
            log_probs_list = cleaned_data.token_log_probs[i]
            
            # DIRECT LOOKUP: Get the IDs already in the pickle file
            # These are usually integers like 0, 1, 2...
            original_ids = cleaned_data.semantic_ids[i]

            prompt_clusters = {}

            for j in range(len(responses)):
                # 1. Get the ID directly
                cluster_id = int(original_ids[j])
                
                # 2. Calculate Probability (Required for Bayesian Math)
                seq_prob = float(np.exp(np.sum(log_probs_list[j])))

                # 3. Build the structure
                if cluster_id not in prompt_clusters:
                    prompt_clusters[cluster_id] = {
                        "meaning_id": cluster_id, 
                        "members": [], 
                        "probabilities": []
                    }
                
                prompt_clusters[cluster_id]["members"].append(responses[j])
                prompt_clusters[cluster_id]["probabilities"].append(seq_prob)

            final_output[prompt_key] = list(prompt_clusters.values())

        # Save output matches your specific naming convention
        output_name = base_file.replace('.pickle', '_original_clusters.json')
        output_filename = os.path.join(OUTPUT_DIR, output_name)

        with open(output_filename, 'w') as f:
            json.dump(final_output, f, indent=2)

        print(f"Saved original clusters to '{output_filename}'")