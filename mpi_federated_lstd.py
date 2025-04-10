from mpi4py import MPI
import numpy as np

class MedicalEnv:
    """
    A simple medical decision-making environment.
    States: 0 (healthy) to num_states-1 (critical).
    Terminal states: 0 (recovered) and num_states-1 (severe condition).
    Actions: 0 = no treatment, 1 = treatment.
    """
    def __init__(self, num_states=5):
        self.num_states = num_states
        self.terminal_states = [0, self.num_states - 1]

    def reset(self):
        # Initialize state to a value between 1 and num_states-2 (i.e., not terminal)
        s = np.random.randint(1, self.num_states - 1)
        return s

    def step(self, s, action):
        """
        Simulate a step in the environment.
        - Action 1 (treatment): With 70% chance, state improves (decreases by 1),
          with 30% chance, state worsens (increases by 1), if possible.
        - Action 0 (no treatment): With 50% chance, state worsens (increases by 1),
          otherwise stays the same.
        A treatment cost is subtracted from the reward.
        Reward is defined as negative of the new state (lower is better) and treatment cost.
        """
        if action == 1:  # Treatment applied
            if np.random.rand() < 0.7:
                s_next = max(s - 1, 0)
            else:
                s_next = min(s + 1, self.num_states - 1)
            cost = 2  # Treatment cost
        else:  # No treatment
            if np.random.rand() < 0.5:
                s_next = min(s + 1, self.num_states - 1)
            else:
                s_next = s
            cost = 0

        reward = - s_next - cost
        done = s_next in self.terminal_states
        return s_next, reward, done

def local_training(env, episodes=20):
    """
    Local training: collect samples from a single hospital (environment).
    Each sample is a tuple (state, reward, next_state).
    """
    samples = []
    for _ in range(episodes):
        s = env.reset()
        done = False
        while not done:
            action = np.random.choice([0, 1])
            s_next, reward, done = env.step(s, action)
            samples.append((s, reward, s_next))
            s = s_next
    return samples

def centralized_training(env, num_states, episodes=60):
    """
    Centralized training on one environment using all data.
    """
    samples = []
    for _ in range(episodes):
        s = env.reset()
        done = False
        while not done:
            action = np.random.choice([0, 1])
            s_next, reward, done = env.step(s, action)
            samples.append((s, reward, s_next))
            s = s_next
    theta = LSTD(samples, num_states)
    return theta

def LSTD(samples, num_states, gamma=0.9):
    """
    Least-Squares Temporal Difference (LSTD) learning.
    Uses one-hot encoding for state features.
    Solves: A theta = b, where:
      A = sum(phi(s) (phi(s)- gamma phi(s_next))^T)
      b = sum(phi(s) * reward)
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
    reg = 1e-5 * np.eye(num_states)
    theta = np.linalg.solve(A + reg, b)
    return theta

def federated_avg(local_thetas):
    """
    Federated averaging: average local theta estimates.
    """
    return np.mean(local_thetas, axis=0)

def federated_training_mpi(rounds=5, num_states=5):
    """
    Federated training using MPI.
    Each MPI process represents one hospital.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    global_theta = np.zeros(num_states)
    for r in range(rounds):
        env = MedicalEnv(num_states)
        samples = local_training(env, episodes=20)
        local_theta = LSTD(samples, num_states)
        print(f"Rank {rank} in round {r+1}: local theta = {local_theta.round(3)}", flush=True)
        theta_sum = np.zeros(num_states)
        # Sum all local thetas across processes
        comm.Allreduce(local_theta, theta_sum, op=MPI.SUM)
        global_theta = theta_sum / size
        if rank == 0:
            print(f"Round {r+1}: Averaged Theta: {global_theta.round(3)}")
    return global_theta

def compare_federated_vs_centralized_MPI():
    """
    Compare federated vs. centralized LSTD training using MPI.
    Rank 0 performs the centralized training and prints comparisons.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    num_states = 5

    if rank == 0:
        print("\n--- Federated LSTD Training (MPI) ---")
    global_theta = federated_training_mpi(rounds=5, num_states=num_states)

    # if rank == 0:
    #     print(f"\nFinal Federated Theta: {global_theta.round(3)}")
    #     print("\n--- Centralized LSTD Training ---")
    #     env = MedicalEnv(num_states)
    #     # Use an equivalent amount of episodes: total episodes = (num_hospitals * 20)
    #     centralized_theta = centralized_training(env, num_states, episodes=size * 20)
    #     print(f"\nFinal Centralized Theta: {centralized_theta.round(3)}")
    #     print("\n--- Theta Comparison (Federated vs. Centralized) ---")
    #     for i in range(num_states):
    #         print(f"State {i}: Federated: {global_theta[i]:.3f} | Centralized: {centralized_theta[i]:.3f}")

    if rank == 0:
        print("\n--- Federated LSTD Training (MPI) ---", flush=True)
    global_theta = federated_training_mpi(rounds=5, num_states=num_states)

    if rank == 0:
        print(f"\nFinal Federated Theta: {global_theta.round(3)}", flush=True)
        print("\n--- Centralized LSTD Training ---", flush=True)
        env = MedicalEnv(num_states)
        centralized_theta = centralized_training(env, num_states, episodes=size * 20)
        print(f"\nFinal Centralized Theta: {centralized_theta.round(3)}", flush=True)
        print("\n--- Theta Comparison (Federated vs. Centralized) ---", flush=True)
        for i in range(num_states):
            print(f"State {i}: Federated: {global_theta[i]:.3f} | Centralized: {centralized_theta[i]:.3f}", flush=True)

if __name__ == "__main__":
    compare_federated_vs_centralized_MPI()
