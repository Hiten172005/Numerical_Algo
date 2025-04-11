import pandas as pd
import numpy as np
from mpi4py import MPI
import math

class RealMedicalEnv:
    def __init__(self, csv_path, batch_size=8, use_softmax_policy=False, theta=None, temperature=1.0):
        self.df = pd.read_csv(csv_path)
        self.original_df = self.df.copy()
        self.index = 0
        self.batch_size = batch_size
        self.num_states = 5
        self.use_softmax_policy = use_softmax_policy
        self.theta = theta
        self.temperature = temperature

    def reset(self):
        self.df = self.original_df.sample(frac=1).reset_index(drop=True)  # Shuffle
        self.index = 0

    def _get_state(self, row):
        state_estimate = (row['diagnosis_severity'] + row['lab_severity'] + row['treatment_intensity']) / 3
        return int(min(max(state_estimate, 0), self.num_states - 1))

    def select_action_softmax(self, state):
        probs = softmax_policy(self.theta, state, temperature=self.temperature)
        return np.random.choice([0, 1], p=probs)

    def get_batch(self):
        if self.index >= len(self.df):
            return [], True

        batch_df = self.df.iloc[self.index : self.index + self.batch_size]
        self.index += self.batch_size
        done = self.index >= len(self.df)

        samples = []
        for _, row in batch_df.iterrows():
            s = self._get_state(row)

            if self.use_softmax_policy and self.theta is not None:
                action = self.select_action_softmax(s)
            else:
                composite_severity = row['diagnosis_severity'] + row['lab_severity'] + row['treatment_intensity']
                if row['admission_type'].upper() == "EMERGENCY":
                    action = 1
                elif any(keyword in row['primary_diagnosis'].upper() for keyword in ["DISTRESS", "SEPSIS", "CARDIAC"]):
                    action = 1
                elif composite_severity > 10:
                    action = 1
                else:
                    action = 0

            cost = 0
            if action == 1:
                cost = 2 + 0.5 * row['treatment_intensity']
            if row['los_hours'] > 100:
                cost += 1

            outcome_penalty = 5 if row['outcome'] == 1 else 0

            if action == 1:
                s_next = max(s - 1, 0)
            else:
                s_next = min(s + 1, self.num_states - 1)

            reward = -s_next - cost - outcome_penalty
            samples.append((s, reward, s_next))

        return samples, done

def feature(s, num_states):
    vec = np.zeros(num_states)
    vec[s] = 1
    return vec

def LSTD(samples, num_states, gamma=0.95):
    A = np.zeros((num_states, num_states))
    b = np.zeros(num_states)

    for s, r, s_next in samples:
        phi_s = feature(s, num_states)
        phi_s_next = feature(s_next, num_states)
        A += np.outer(phi_s, (phi_s - gamma * phi_s_next))
        b += phi_s * r

    theta = np.linalg.pinv(A) @ b
    return theta

def softmax_policy(theta, state, num_actions=2, temperature=1.0):
    next_states = [min(state + 1, len(theta) - 1), max(state - 1, 0)]
    q_values = np.array([-theta[s_prime] for s_prime in next_states])
    exp_q = np.exp(q_values / temperature)
    probs = exp_q / np.sum(exp_q)
    return probs

def local_training(env):
    samples = []
    env.reset()
    done = False
    while not done:
        batch_samples, done = env.get_batch()
        samples.extend(batch_samples)

    return samples

def federated_training_mpi(csv_paths, rounds=5, num_states=5):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    env = RealMedicalEnv(csv_paths[rank], batch_size=8)
    global_theta = np.zeros(num_states)

    for r in range(rounds):
        samples = local_training(env)
        local_theta = LSTD(samples, num_states)
        print(f"Rank {rank} in round {r+1}: local theta = {local_theta.round(3)}", flush=True)
        theta_sum = np.zeros(num_states)
        comm.Allreduce(local_theta, theta_sum, op=MPI.SUM)
        global_theta = theta_sum / size
        env.theta = global_theta  # So next round uses updated policy
        env.use_softmax_policy = True

        if rank == 0:
            print(f"Round {r+1}: Averaged Theta: {global_theta.round(3)}", flush=True)

    return global_theta

def centralized_training(csv_paths, num_states=5):
    all_samples = []
    for path in csv_paths:
        env = RealMedicalEnv(path, batch_size=8)
        samples = local_training(env)
        all_samples.extend(samples)
    theta = LSTD(all_samples, num_states)
    return theta

def evaluate_policy(theta, env, temperature=1.0):
    env.theta = theta
    env.temperature = temperature
    env.use_softmax_policy = True
    env.reset()
    total_reward = 0
    done = False

    while not done:
        batch_samples, done = env.get_batch()
        total_reward += sum(r for _, r, _ in batch_samples)

    return total_reward

def compare_federated_vs_centralized_MPI():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    num_states = 5

    csv_paths = [f"hospital_{i}.csv" for i in range(size)]

    global_theta = federated_training_mpi(csv_paths, rounds=5, num_states=num_states)

    if rank == 0:
        print(f"\nFinal Federated Theta: {global_theta.round(3)}", flush=True)

        centralized_theta = centralized_training(csv_paths, num_states=num_states)
        print(f"\nFinal Centralized Theta: {centralized_theta.round(3)}", flush=True)

        print("\n--- Theta Comparison (Federated vs. Centralized) ---", flush=True)
        for i in range(num_states):
            print(f"State {i}: Federated: {global_theta[i]:.3f} | Centralized: {centralized_theta[i]:.3f}", flush=True)

        # Evaluate policies
        for i, path in enumerate(csv_paths):
            eval_env = RealMedicalEnv(path, batch_size=8)
            reward_fed = evaluate_policy(global_theta, eval_env)
            eval_env = RealMedicalEnv(path, batch_size=8)
            reward_central = evaluate_policy(centralized_theta, eval_env)
            print(f"\nHospital {i} - Softmax Policy Eval: Federated Reward = {reward_fed:.2f}, Centralized Reward = {reward_central:.2f}", flush=True)

if __name__ == "__main__":
    compare_federated_vs_centralized_MPI()
