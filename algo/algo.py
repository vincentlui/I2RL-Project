import numpy as np
from tqdm import tqdm


class ValueIterationRL:
    def __init__(self, environment, theta=1e-4, gamma=0.99):
        self.env = environment
        self.theta = theta
        self.gamma = gamma
        self.trans_matrix = self.env.trans_matrix
        self.v_state = None
        self.best_action = None

    def value_iteration(self, max_iter=1000):
        total_error = 1e5
        rewards_penalty = np.ones((self.env.num_states, self.env.num_actions)) * -np.inf
        rewards_penalty[self.env.all_states_actions.nonzero()] = 0
        rewards = self.env.get_all_rewards() * np.ones_like(self.env.all_states_actions)
        rewards = rewards.T.reshape(-1, 1, self.env.num_states)
        v_state = np.ones_like(self.env.array_state_id) / -self.env.num_states
        i = 1
        pbar = tqdm(total=max_iter)
        print('Value iteration starts')
        while np.any(total_error > self.theta) and i < max_iter:
            tmp = np.sum(self.trans_matrix * (rewards + self.gamma * v_state.reshape(1, -1)), axis=-1)
            tmp += rewards_penalty.T
            new_v_state = np.max(tmp, axis=0)
            new_v_state[0] = 0
            total_error = np.abs(new_v_state - v_state)
            # print((total_error > self.theta).nonzero()[0])
            v_state = new_v_state
            i += 1
            pbar.update()

        pbar.close()
        print(f'Value iteration finished in iter {i}')
        self.v_state = v_state
        self.best_action = np.argmax(tmp, axis=0)
        self.env.set_optimal_policy(self.v_state, self.best_action)
        return v_state, self.best_action


    def value_iteration_sparse(self, max_iter=1000):
        total_error = 1e5
        rewards_penalty = np.ones((self.env.num_states, self.env.num_actions)) * -np.inf
        rewards_penalty[self.env.all_states_actions.nonzero()] = 0
        # print(rewards_penalty[88])
        rewards = self.env.get_all_rewards() * np.ones_like(self.env.all_states_actions)
        # rewards = rewards.T.reshape(-1, 1, self.env.num_states)
        v_state = np.ones_like(self.env.array_state_id) / -self.env.num_states
        i = 1
        pbar = tqdm(total=max_iter)
        print('Value iteration starts')
        best_action = None
        while np.any(total_error > self.theta) and i < max_iter:
            tmp = []
            for a in range(self.env.num_actions):
                tmp.append(self.trans_matrix[a] @ (rewards.T[a] + self.gamma * v_state) + rewards_penalty.T[a])
            # tmp = np.sum(self.trans_matrix * (rewards + self.gamma * v_state.reshape(1, -1)), axis=-1)
            new_v_state = np.max(tmp, axis=0)
            new_v_state[0] = 0
            best_action = np.argmax(tmp, axis=0)
            total_error = np.abs(new_v_state - v_state)
            v_state = new_v_state
            i += 1
            pbar.update()

        pbar.close()
        print(f'Value iteration finished in iter {i}')
        self.v_state = v_state
        self.best_action = best_action
        self.env.set_optimal_policy(self.v_state, self.best_action)
        return v_state, self.best_action
