import numpy as np
import pandas as pd
import os
from collections import defaultdict
import time

class RealMedicalEnv:
    def __init__(self, hospital_data):
        self.data = hospital_data
        self.current_idx = 0
        self.num_states = 5
        self.num_actions = 2  # 0=no treatment, 1=treatment
        self.terminal_states = [0, self.num_states - 1]
        self.patient_episodes = self.data.groupby('subject_id')
        self.patient_ids = list(self.patient_episodes.groups.keys())
        
    def reset(self):
        patient_id = np.random.choice(self.patient_ids)
        patient_data = self.patient_episodes.get_group(patient_id)
        self.current_patient = patient_data.index
        self.current_idx = 0
        return patient_data.iloc[0]['diagnosis_severity'] - 1
        
    def step(self, s, action):
        patient_data = self.data.loc[self.current_patient]
        current_row = patient_data.iloc[self.current_idx]
        
        # Get next state from actual trajectory
        if self.current_idx + 1 < len(patient_data):
            next_row = patient_data.iloc[self.current_idx + 1]
            self.current_idx += 1
        else:
            next_row = current_row
            
        s_next = next_row['diagnosis_severity'] - 1
        
        # Reward calculation
        reward = -s_next  # Base reward
        if next_row['outcome'] == 0:  # Survival bonus
            reward += 5
        if action == 1 and current_row['treatment_intensity'] >= 4:  # Treatment cost
            reward -= 2
            
        done = (next_row['outcome'] == 1) or (self.current_idx >= len(patient_data) - 1)
        return s_next, reward, done

def load_hospital_data(data_dir):
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory not found: {data_dir}")
        
    hospitals = []
    for filename in sorted(os.listdir(data_dir)):
        if filename.startswith('hospital_') and filename.endswith('.csv'):
            path = os.path.join(data_dir, filename)
            try:
                df = pd.read_csv(path)
                if len(df) > 0:
                    hospitals.append(df)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    if not hospitals:
        raise ValueError("No valid hospital data files found")
    return hospitals

def LSTDQ(samples, num_states, num_actions, Q, gamma=0.9):
    """LSTD-Q for policy evaluation"""
    n_features = num_states * num_actions
    A = np.zeros((n_features, n_features))
    b = np.zeros(n_features)
    
    for s, a, r, s_next in samples:
        # Current state-action features
        phi = np.zeros(n_features)
        phi[s * num_actions + a] = 1.0
        
        # Next state features using current policy
        phi_next = np.zeros(n_features)
        best_action = np.argmax([Q[s_next, a] for a in range(num_actions)])
        phi_next[s_next * num_actions + best_action] = 1.0
        
        A += np.outer(phi, (phi - gamma * phi_next))
        b += phi * r
    
    reg = 1e-5 * np.eye(n_features)
    theta = np.linalg.solve(A + reg, b)
    return theta.reshape(num_states, num_actions)

def policy_iteration(env, num_states, num_actions, iterations=5):
    """LSPI implementation"""
    # Initialize Q-function and policy
    Q = np.zeros((num_states, num_actions))
    policy = lambda s: np.random.choice([0, 1])  # Random initial policy
    
    for _ in range(iterations):
        # Collect samples using current policy
        samples = []
        for _ in range(20):  # 20 episodes per iteration
            s = env.reset()
            done = False
            while not done:
                a = policy(s)
                s_next, r, done = env.step(s, a)
                samples.append((s, a, r, s_next))
                s = s_next
        
        # Policy evaluation - pass current Q to LSTDQ
        Q = LSTDQ(samples, num_states, num_actions, Q)
        
        # Policy improvement
        policy = lambda s: np.argmax(Q[s])
    
    return Q, policy

def federated_lspi(hospitals_data, num_rounds=5, num_iterations=3):
    """Federated LSPI implementation"""
    num_states = 5
    num_actions = 2
    global_Q = np.zeros((num_states, num_actions))
    
    print("\n--- Federated LSPI Training ---")
    for r in range(num_rounds):
        local_Qs = []
        for hospital_data in hospitals_data:
            env = RealMedicalEnv(hospital_data)
            Q, _ = policy_iteration(env, num_states, num_actions, num_iterations)
            local_Qs.append(Q)
        
        # Federated averaging of Q-functions
        global_Q = np.mean(local_Qs, axis=0)
        print(f"Round {r+1} Global Q:\n{global_Q.round(3)}")
    
    # Final policy
    global_policy = lambda s: np.argmax(global_Q[s])
    return global_Q, global_policy

def centralized_lspi(hospitals_data, num_iterations=5):
    """Centralized LSPI implementation"""
    num_states = 5
    num_actions = 2
    all_data = pd.concat(hospitals_data)
    env = RealMedicalEnv(all_data)
    
    print("\n--- Centralized LSPI Training ---")
    Q, policy = policy_iteration(env, num_states, num_actions, num_iterations)
    print(f"Final Centralized Q:\n{Q.round(3)}")
    return Q, policy

def compare_lspi(data_dir):
    hospitals_data = load_hospital_data(data_dir)
    
    # Federated LSPI
    fed_Q, fed_policy = federated_lspi(hospitals_data)
    
    # Centralized LSPI
    cen_Q, cen_policy = centralized_lspi(hospitals_data)
    
    # Comparison
    print("\n--- Q-value Comparison ---")
    for s in range(5):
        print(f"State {s}: Fed Q={fed_Q[s].round(3)} | Cen Q={cen_Q[s].round(3)}")
    
    print("\n--- Policy Comparison ---")
    for s in range(5):
        print(f"State {s}: Fed Action={fed_policy(s)} | Cen Action={cen_policy(s)}")

if __name__ == "__main__":
    start_time = time.time()
    data_dir = "/home/khwaish-garg/Desktop/SEM-4/NA/Project/Numerical_Algo/hospital_data"
    compare_lspi(data_dir)
    end_time = time.time()
    print(f"Time taken to run the entire script: {end_time - start_time:.4f} seconds")