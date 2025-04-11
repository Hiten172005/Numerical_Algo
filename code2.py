import numpy as np
import pandas as pd
import os

class RealMedicalEnv:
    """
    Medical environment using real hospital data.
    States: Based on diagnosis_severity (1-5)
    Actions: Based on treatment_intensity
    """
    def __init__(self, hospital_data):
        self.data = hospital_data
        self.current_idx = 0
        self.num_states = 5  # Based on severity levels
        self.num_actions = 2  # Actions: 0 (no treatment), 1 (treatment)
        self.terminal_states = [0, self.num_states - 1]
        # Group data by patient for proper transitions
        self.patient_episodes = self.data.groupby('subject_id')
        self.patient_ids = list(self.patient_episodes.groups.keys())
        
    def reset(self):
        # Start with a random patient
        patient_id = np.random.choice(self.patient_ids)
        patient_data = self.patient_episodes.get_group(patient_id)
        self.current_patient = patient_data.index
        self.current_idx = 0
        return patient_data.iloc[0]['diagnosis_severity'] - 1
        
    def step(self, s, action):
        patient_data = self.data.loc[self.current_patient]
        current_row = patient_data.iloc[self.current_idx]
        
        # Ensure action matches treatment intensity
        data_action = 1 if current_row['treatment_intensity'] >= 3 else 0
        action = data_action  # Override with data-based action
        
        # Get next state from actual patient trajectory
        if self.current_idx + 1 < len(patient_data):
            next_row = patient_data.iloc[self.current_idx + 1]
            self.current_idx += 1
        else:
            next_row = current_row
            
        s_next = next_row['diagnosis_severity'] - 1
        
        # Reward based on outcome and treatment
        reward = -s_next  # Base reward is negative of severity
        if next_row['outcome'] == 0:  # Patient survived
            reward += 5
        if current_row['treatment_intensity'] >= 4:  # Intensive treatment cost
            reward -= 2
            
        # Episode ends if patient dies or reaches max episodes
        done = (next_row['outcome'] == 1) or (self.current_idx >= len(patient_data) - 1)
        return s_next, reward, done

def load_hospital_data(data_dir):
    """Load and preprocess hospital data."""
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory not found: {data_dir}")
        
    hospitals = []
    for filename in sorted(os.listdir(data_dir)):
        if filename.startswith('hospital_') and filename.endswith('.csv'):
            path = os.path.join(data_dir, filename)
            try:
                df = pd.read_csv(path)
                if len(df) > 0:  # Only add non-empty dataframes
                    hospitals.append(df)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    if not hospitals:
        raise ValueError("No valid hospital data files found")
    return hospitals

def feature_mapping(state, action, num_states, num_actions):
    """Generate feature vector for state-action pair."""
    phi = np.zeros(num_states * num_actions)
    index = state + action * num_states
    phi[index] = 1
    return phi

def policy_evaluation(samples, phi, gamma, num_features):
    """Perform policy evaluation using least-squares approximation."""
    A = np.zeros((num_features, num_features))
    b = np.zeros(num_features)
    for s, a, r, s_next, a_next in samples:
        phi_sa = phi(s, a)
        phi_s_next_a_next = phi(s_next, a_next)
        A += np.outer(phi_sa, phi_sa - gamma * phi_s_next_a_next)
        b += phi_sa * r
    theta = np.linalg.solve(A + 1e-5 * np.eye(num_features), b)  # Add regularization
    return theta

def policy_improvement(theta, phi, num_states, num_actions):
    """Improve policy by choosing actions that maximize Q-values."""
    def policy(state):
        q_values = [np.dot(theta, phi(state, a)) for a in range(num_actions)]
        return np.argmax(q_values)
    return policy

def lspi(env, samples, num_states, num_actions, gamma=0.9, max_iterations=10):
    """Least-Squares Policy Iteration (LSPI)."""
    num_features = num_states * num_actions
    phi = lambda s, a: feature_mapping(s, a, num_states, num_actions)
    theta = np.zeros(num_features)  # Initialize theta
    policy = lambda s: np.random.choice([0, 1])  # Start with random policy
    
    for iteration in range(max_iterations):
        # Collect samples using the current policy
        samples_with_actions = []
        for s, r, s_next in samples:
            a = policy(s)
            a_next = policy(s_next)
            samples_with_actions.append((s, a, r, s_next, a_next))
        
        # Policy Evaluation
        theta = policy_evaluation(samples_with_actions, phi, gamma, num_features)
        
        # Policy Improvement
        policy = policy_improvement(theta, phi, num_states, num_actions)
        
        print(f"Iteration {iteration + 1}: Theta = {theta}")
    
    return policy, theta

def collect_samples(env, episodes=20):
    """Collect samples from the environment."""
    samples = []
    for _ in range(episodes):
        s = env.reset()
        done = False
        while not done:
            s_next, reward, done = env.step(s, None)  # Use data-based action
            samples.append((s, reward, s_next))
            s = s_next
    return samples

if __name__ == "__main__":
    data_dir = "/Users/hitengarg/Documents/Numerical Algo/Project/Numerical_Algo/hospital_data"
    hospitals_data = load_hospital_data(data_dir)
    
    # Combine all hospital data for centralized LSPI
    all_hospital_data = pd.concat(hospitals_data)
    env = RealMedicalEnv(all_hospital_data)
    
    # Collect samples from the environment
    samples = collect_samples(env, episodes=100)
    
    # Run LSPI
    num_states = 5
    num_actions = 2
    policy, theta = lspi(env, samples, num_states, num_actions)
    
    print("\nLearned Policy:")
    for state in range(num_states):
        print(f"State {state}: Action {policy(state)}")