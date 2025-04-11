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
    
    def get_current_treatment(self):
        """Get the treatment intensity for the current state."""
        patient_data = self.data.loc[self.current_patient]
        current_row = patient_data.iloc[self.current_idx]
        return current_row['treatment_intensity']

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

def local_training_real(env, episodes=20, theta=None):
    """Collect samples using treatment data from historical records."""
    if theta is None:
        theta = np.zeros(env.num_states)
    
    samples = []
    for _ in range(episodes):
        s = env.reset()
        done = False
        while not done:
            # No need to determine action - step will use the data-based action
            s_next, reward, done = env.step(s, None)  # Pass None since action will be overridden
            samples.append((s, reward, s_next))
            s = s_next
    return samples

def LSTD(samples, num_states, gamma=0.9):
    """
    Least-Squares Temporal Difference (LSTD) learning.
    Uses one-hot encoding for state features.
    """
    A = np.zeros((num_states, num_states))
    b = np.zeros(num_states)
    for s, reward, s_next in samples:
        phi_s = np.zeros(num_states)
        phi_s[int(s)] = 1.0
        phi_s_next = np.zeros(num_states)
        phi_s_next[int(s_next)] = 1.0
        A += np.outer(phi_s, (phi_s - gamma * phi_s_next))
        b += phi_s * reward
    
    # Add small regularization term for numerical stability
    reg = 1e-5 * np.eye(num_states)
    theta = np.linalg.solve(A + reg, b)
    return theta

def federated_avg(local_thetas):
    """
    Federated averaging: compute mean of local theta estimates.
    """
    return np.mean(local_thetas, axis=0)

def centralized_training(env, num_states, episodes=60):
    """Centralized training using historical data actions."""
    theta = np.zeros(num_states)
    samples = []
    
    for _ in range(episodes):
        s = env.reset()
        done = False
        while not done:
            # No need to determine action - step will use the data-based action
            s_next, reward, done = env.step(s, None)  # Pass None since action will be overridden
            samples.append((s, reward, s_next))
            s = s_next
        # Update theta using collected samples
        theta = LSTD(samples, num_states)
    
    return theta

def compare_federated_vs_centralized_real(data_dir, num_rounds=5):
    hospitals_data = load_hospital_data(data_dir)
    num_states = 5
    global_theta = np.zeros(num_states)
    
    # Federated Learning
    print("\n--- Federated LSTD Training with Real Data ---")
    for r in range(num_rounds):
        local_thetas = []
        for hospital_data in hospitals_data:
            print(hospital_data)
            env = RealMedicalEnv(hospital_data)
            print(env)
            samples = local_training_real(env)
            theta = LSTD(samples, num_states)
            local_thetas.append(theta)
        global_theta = federated_avg(local_thetas)
        print(f"Round {r+1}: Averaged Theta: {global_theta.round(3)}")
    
    print(f"\nFinal Federated Theta: {global_theta.round(3)}")
    
    # Centralized Learning
    print("\n--- Centralized LSTD Training with Real Data ---")
    all_hospital_data = pd.concat(hospitals_data)
    env = RealMedicalEnv(all_hospital_data)
    centralized_theta = centralized_training(env, num_states, episodes=len(hospitals_data) * 20)
    print(f"\nFinal Centralized Theta: {centralized_theta.round(3)}")
    
    # Comparison
    print("\n--- Theta Comparison (Federated vs. Centralized) ---")
    for i in range(num_states):
        print(f"State {i}: Federated: {global_theta[i]:.3f} | Centralized: {centralized_theta[i]:.3f}")

def visualize_results(federated_theta, centralized_theta):
    """Visualize comparison between federated and centralized learning results."""
    import matplotlib.pyplot as plt
    
    states = range(len(federated_theta))
    
    plt.figure(figsize=(10, 6))
    plt.plot(states, federated_theta, 'b-o', label='Federated Learning')
    plt.plot(states, centralized_theta, 'r--s', label='Centralized Learning')
    
    plt.title('Comparison of Federated vs Centralized Learning')
    plt.xlabel('State (Severity Level)')
    plt.ylabel('Estimated Value')
    plt.xticks(states)
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    data_dir = "/Users/hitengarg/Documents/Numerical Algo/Project/Numerical_Algo/hospital_data"
    compare_federated_vs_centralized_real(data_dir)