import json
import numpy as np
from bayesian_estimator import BayesianSemanticEntropy

# --- PHASE 4: HYPERPARAMETERS ---
# ALPHA: 1.0 = "I have no bias". 0.1 = "I expect the model to be very confident".
# GAMMA: 0.005 is the standard "Variance Threshold" from the paper.
ALPHA = 1.0
GAMMA = 0.005 

# Path to your cosine clusters
CLUSTERS_FILE = "D:/LLM-Hallucination/data/meanings/cosine_sim/nnnp-Llama-2-7b-chat--nq--results_cosine_clusters.json"

def evaluate_performance():
    print(f"--- STARTING EVALUATION (Alpha={ALPHA}, Gamma={GAMMA}) ---")
    
    estimator = BayesianSemanticEntropy(alpha=ALPHA)
    
    with open(CLUSTERS_FILE, 'r') as f:
        data = json.load(f)

    # Stats trackers
    total_samples_used = []
    hard_questions = [] # Questions that needed many samples
    easy_questions = [] # Questions that stopped at N=2
    
    # Process a subset (first 50 prompts) for quick tuning
    prompt_keys = list(data.keys())[:50]

    for key in prompt_keys:
        clusters = data[key]
        
        # 1. Flatten Data (Simulate the pool of samples)
        # We assume the order in the list is the order they were generated
        pool = []
        # Extract samples from clusters and sort by original index if possible
        # (Simplified here: just flattening)
        for c in clusters:
            m_id = c['meaning_id']
            # Assuming you saved 'probabilities' and 'members' lists aligned
            for i, prob in enumerate(c['probabilities']):
                pool.append({
                    'meaning_id': m_id, 
                    'probability': prob,
                    'text': c['members'][i]
                })
        
        # 2. Run Adaptive Loop
        current_samples = []
        stopped_at_n = 0
        final_entropy = 0.0
        
        for n in range(len(pool)):
            current_samples.append(pool[n])
            
            # Run Math
            entropy, variance = estimator.estimate_entropy(current_samples)
            
            # Check Stop Condition (Must have at least 2 samples to measure variance)
            if n >= 1 and variance < GAMMA:
                stopped_at_n = n + 1
                final_entropy = entropy
                break
            
            stopped_at_n = n + 1
            final_entropy = entropy

        # 3. Record Stats
        total_samples_used.append(stopped_at_n)
        
        # Clean prompt text for display
        prompt_text = key.split(": ")[1][:60] + "..."
        
        entry = f"N={stopped_at_n} | Entropy={final_entropy:.2f} | Q: {prompt_text}"
        
        if stopped_at_n > 4:
            hard_questions.append(entry)
        elif stopped_at_n <= 2:
            easy_questions.append(entry)

    # --- PRINT REPORT CARD ---
    print("\n" + "="*40)
    print("RESULTS SUMMARY")
    print("="*40)
    print(f"Average Samples Needed: {np.mean(total_samples_used):.2f}")
    print(f"Savings: Used {np.mean(total_samples_used)} samples instead of 10.")
    
    print("\n[EASY QUESTIONS] (Stopped Early)")
    for q in easy_questions[:3]:
        print(q)
        
    print("\n[CONFUSING QUESTIONS] (Triggered Loop)")
    for q in hard_questions[:3]:
        print(q)

if __name__ == "__main__":
    evaluate_performance()