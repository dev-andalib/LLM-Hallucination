import json
import numpy as np
from bayesian_estimator import BayesianSemanticEntropy


ALPHA = 1.0
GAMMA = 0.005 


CLUSTERS_FILE = None

def evaluate_performance():
    print(f"--- STARTING EVALUATION (Alpha={ALPHA}, Gamma={GAMMA}) ---")
    
    estimator = BayesianSemanticEntropy(alpha=ALPHA)
    
    with open(CLUSTERS_FILE, 'r') as f:
        data = json.load(f)

    
    total_samples_used = []
    hard_questions = [] 
    easy_questions = [] 

    prompt_keys = list(data.keys())[:50]

    for key in prompt_keys:
        clusters = data[key]
        pool = []
        for c in clusters:
            m_id = c['meaning_id']
            for i, prob in enumerate(c['probabilities']):
                pool.append({
                    'meaning_id': m_id, 
                    'probability': prob,
                    'text': c['members'][i]
                })
        

        current_samples = []
        stopped_at_n = 0
        final_entropy = 0.0
        
        for n in range(len(pool)):
            current_samples.append(pool[n])
            entropy, variance = estimator.estimate_entropy(current_samples)
            if n >= 1 and variance < GAMMA:
                stopped_at_n = n + 1
                final_entropy = entropy
                break 
            stopped_at_n = n + 1
            final_entropy = entropy

        total_samples_used.append(stopped_at_n)
        prompt_text = key.split(": ")[1][:60] + "..."
        entry = f"N={stopped_at_n} | Entropy={final_entropy:.2f} | Q: {prompt_text}"
        
        if stopped_at_n > 4:
            hard_questions.append(entry)
        elif stopped_at_n <= 2:
            easy_questions.append(entry)

    
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