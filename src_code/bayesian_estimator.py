import numpy as np
from scipy.stats import dirichlet

class BayesianSemanticEntropy:
    def __init__(self, alpha=1.0, max_mc_samples=5000):
        """
        Args:
            alpha (float): The Dirichlet prior parameter (Paper uses 0.5 or 1.0).
            max_mc_samples (int): How many random vectors to generate for the integral 
                                  approximation (Appendix A.2).
        """
        self.alpha = alpha
        self.max_mc_samples = max_mc_samples

    def _shannon_entropy(self, probability_vector):
        """
        Calculates standard Shannon Entropy: H(x) = -sum(p * log(p))
        """
        # Add a tiny epsilon (1e-12) to prevent log(0) errors
        p = np.array(probability_vector) + 1e-12
        return -np.sum(p * np.log(p))

    def estimate_entropy(self, samples):
        """
        The Core Logic.
        
        Args:
            samples: A list of dicts. Example:
                     [ {'meaning_id': 1, 'probability': 0.4}, 
                       {'meaning_id': 1, 'probability': 0.3}, 
                       {'meaning_id': 2, 'probability': 0.1} ]
        
        Returns:
            float: The Estimated Semantic Entropy
        """
        if not samples:
            return 0.0

        # --- PART 1: The Constraint (Section 3.2, Eq 4) ---
        # "The true probability of a meaning is AT LEAST the sum of probabilities 
        # of the observed sequences belonging to that meaning."
        
        # 1. Sum probabilities per meaning
        meaning_counts = {}
        for s in samples:
            m_id = s['meaning_id']
            prob = s['probability']
            meaning_counts[m_id] = meaning_counts.get(m_id, 0.0) + prob

        # 2. Extract lower bounds
        # We need a fixed order, so we sort by ID
        unique_meanings = sorted(meaning_counts.keys())
        lower_bounds = np.array([meaning_counts[m] for m in unique_meanings])
        
        # Sanity Check: If LLM is >100% confident (due to float errors), entropy is 0.
        if np.sum(lower_bounds) >= 1.0 - 1e-6:
            return 0.0

        K_observed = len(unique_meanings)
        
        # --- PART 2: Handling Unknown K (Section 3.3) ---
        # We don't know the true number of meanings (K). It might be K_observed, 
        # or there might be meanings we haven't seen yet.
        # We iterate through a few hypotheses (e.g., K, K+1, ... K+5) and average the results.
        
        entropy_estimates = []
        
        # We test hypotheses: "What if there are 0 unseen meanings? What if there is 1?"
        for num_unseen in range(0, 5): 
            K_total = K_observed + num_unseen
            
            # --- PART 3: Monte Carlo Integration (Appendix A.2) ---
            # We need to calculate the average entropy of distributions that 
            # satisfy our constraints.
            
            # A. Setup Dirichlet Prior
            # Create a vector of alphas. [1.0, 1.0, ... 1.0]
            alpha_vec = np.full(K_total, self.alpha)
            
            # B. Generate Random Probability Vectors (The "Proposal Distribution")
            # This creates 'max_mc_samples' vectors, each summing to 1.0
            # Shape: (5000, K_total)
            try:
                candidate_vectors = dirichlet.rvs(alpha_vec, size=self.max_mc_samples)
            except ValueError:
                # Fallback if dimensions are invalid
                continue
            
            # C. Apply Constraints (Rejection Sampling)
            # We check the first K_observed columns against our lower_bounds.
            # (We don't check unseen meanings because their lower bound is 0, which is always true)
            
            observed_candidates = candidate_vectors[:, :K_observed]
            
            # Logic: For every row, is (Candidate >= LowerBound) for ALL columns?
            valid_mask = np.all(observed_candidates >= lower_bounds, axis=1)
            
            # Keep only the vectors that are mathematically possible given our data
            valid_vectors = candidate_vectors[valid_mask]
            
            # If no vectors passed the check (constraints too tight), skip this K
            if len(valid_vectors) == 0:
                continue
                
            # D. Calculate Entropy for valid vectors
            entropies = [self._shannon_entropy(vec) for vec in valid_vectors]
            
            # Average them to get Expected Entropy for this K
            avg_entropy_for_K = np.mean(entropies)
            entropy_estimates.append(avg_entropy_for_K)

        # --- PART 4: Final Aggregation ---
        # Average the estimates across our hypotheses about K
        if not entropy_estimates:
            return 0.0
            
        final_score = np.mean(entropy_estimates)
        return final_score