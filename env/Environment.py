import numpy as np
from scipy.stats import binom
from operator import itemgetter
from tqdm import tqdm
from scipy.sparse import csr_matrix


class Environment:
    def __init__(self, num_class, num_bed, p_arrive, p_leave, delta, max_leave=3, sparse=True):
        self.num_class = num_class
        self.num_bed = num_bed
        self.p_arrive = np.array(p_arrive)
        self.p_leave = np.array(p_leave)
        self.delta = np.array(delta)
        self.max_leave = max_leave
        self.sparse = True
        self.trans_matrix = None
        self.distr_leave = [binom(range(self.num_bed + 1), p) for p in p_leave]
        self.v_optimal = None
        self.a_optimal = None

    def calc_dynamics(self):
        if self.sparse:
            return self.calc_dynamics_sparse()

        all_states = self.get_all_states()

        self.num_states = len(all_states)
        self.num_actions = self.num_class + 1
        self.array_state_id = self._get_state_id(all_states)
        trans_matrix = np.zeros((self.num_actions, self.num_states, self.num_states))
        all_states_actions = np.zeros((self.num_states, self.num_actions))
        self.id_to_index = self._get_id_to_index(self.array_state_id)

        print(f'Total states: {self.num_states}')
        print('Calculating dynamics...')

        for i, s in tqdm(enumerate(all_states), total=self.num_states):
            state = State(s[:-1], s[-1], self.num_bed)
            state_id = state.get_id()
            state_index = self.id_to_index[state_id]
            all_actions = state.all_actions()

            for j, a in enumerate(all_actions):
                next_state_id, p_trans, reward = self._get_all_next_states(s, a)
                next_state_ind = self._get_index_from_state_ids(next_state_id)
                p_trans_by_index = np.zeros(self.num_states)
                p_trans_by_index[next_state_ind] = p_trans

                all_states_actions[state_index, a] = 1
                trans_matrix[a, state_index, :] = p_trans_by_index

        self.trans_matrix = trans_matrix
        self.all_states_actions = all_states_actions
        print('Calculating dynamics... finished')

    def calc_dynamics_sparse(self):
        all_states = self.get_all_states()

        self.num_states = len(all_states)
        self.num_actions = self.num_class + 1
        self.array_state_id = self._get_state_id(all_states)
        # trans_matrix = np.zeros((self.num_actions, self.num_states, self.num_states))
        all_states_actions = np.zeros((self.num_states, self.num_actions))
        self.id_to_index = self._get_id_to_index(self.array_state_id)

        print(f'Total states: {self.num_states}')
        print('Calculating dynamics...')

        list_p = [[] for i in range(self.num_actions)]
        list_ind = [[] for i in range(self.num_actions)]
        list_ptr = [[0] for i in range(self.num_actions)]

        for i, s in tqdm(enumerate(all_states), total=self.num_states):
            state = State(s[:-1], s[-1], self.num_bed)
            state_id = state.get_id()
            state_index = self.id_to_index[state_id]
            all_actions = state.all_actions()

            for j in range(self.num_actions):
                if np.isin(j, all_actions):
                    a = j
                    next_state_id, p_trans, reward = self._get_all_next_states(s, a)
                    list_ind[a].append(self._get_index_from_state_ids(next_state_id))
                    list_p[a].append(p_trans)
                    list_ptr[a].append(list_ptr[a][-1] + len(p_trans))
                    all_states_actions[state_index, a] = 1
                else:
                    list_ptr[j].append(list_ptr[j][-1])

                # trans_matrix[a, state_index, :] = p_trans_by_index

        all_p = [np.concatenate(x, axis=0) for x in list_p]
        all_ind = [np.concatenate(x, axis=0) for x in list_ind]
        self.trans_matrix = [csr_matrix((all_p[i], all_ind[i], list_ptr[i]), shape=(self.num_states, self.num_states))
                             for i in range(self.num_actions)]
        self.all_states_actions = all_states_actions
        print('Calculating dynamics... finished')

    def all_actions(self):
        return np.array(range(self.num_class + 1))

    def set_optimal_policy(self, v, a):
        self.v_optimal = v
        self.a_optimal = a

    def best_action(self, s):
        assert self.a_optimal is not None, 'Optimal policy is not set'
        sid = s.get_id()
        index = self.id_to_index[sid]
        return self.a_optimal[index]

    def get_all_states(self, class_max=5):
        if self.num_class > class_max:
            all_x = self._get_all_states_low_mem()
        else:
            tmp = np.tile(np.arange(self.num_bed + 1), (self.num_class, 1))
            tmp = np.array(np.meshgrid(*tmp)).T.reshape(-1, self.num_class)
            all_x = tmp[tmp.sum(-1) <= self.num_bed]
        num_rows = len(all_x)
        col_x = np.tile(all_x, (self.num_class + 1, 1))

        all_y = np.arange(self.num_class + 1)
        col_y = np.tile(all_y, (num_rows, 1)).T.reshape(-1, 1)
        all_states = np.concatenate((col_x, col_y), axis=-1)
        return all_states

    def _get_all_states_low_mem(self, class_max=5):
        # meshgrid for 6 dim only
        tmp = np.tile(np.arange(self.num_bed + 1), (class_max, 1))
        tmp = np.array(np.meshgrid(*tmp)).T.reshape(-1, class_max)
        all_x = tmp[tmp.sum(-1) <= self.num_bed]
        num_rows = len(all_x)

        for i in range(self.num_class - class_max):
            tmp = np.tile(all_x, (self.num_bed + 1, 1))
            tmp2 = np.tile(np.arange(self.num_bed + 1), (len(all_x), 1)).T.reshape(-1, 1)
            tmp = np.concatenate([tmp, tmp2], axis=-1)
            all_x = tmp[tmp.sum(-1) <= self.num_bed]

        return all_x

    def _get_state_id(self, all_states):
        tmp = np.arange(all_states.shape[-1])
        return np.sum(all_states * ((self.num_bed + 1) ** tmp), axis=-1)

    def _get_id_to_index(self, array_state_id):
        id_to_index = {}
        for i, v in enumerate(array_state_id):
            id_to_index[v] = i
        return id_to_index

    def _get_all_next_states(self, s, a):
        current_state = s.copy()
        if a != State.ACTION_STAY:
            current_state[a - 1] -= 1
        if current_state[-1] != 0:
            current_state[current_state[-1] - 1] += 1
        tmp = np.tile(np.arange(self.max_leave + 1), (self.num_class, 1))
        tmp = np.array(np.meshgrid(*tmp)).T.reshape(-1, self.num_class)
        tmp = current_state[:-1] - tmp
        all_x = tmp[tmp.min(1) >= 0]
        num_rows = len(all_x)
        col_x = np.tile(all_x, (self.num_class + 1, 1))

        all_y = np.arange(self.num_class + 1)
        col_y = np.tile(all_y, (num_rows, 1)).T.reshape(-1, 1)
        all_next_states = np.concatenate((col_x, col_y), axis=-1)

        num_leave_per_class = current_state[:-1] - all_next_states[:, :-1]
        p_transition = np.ones(len(all_next_states))

        for i, n in enumerate(num_leave_per_class.T):
            p_transition *= self.distr_leave[i].pmf(n.reshape(-1, 1))[:, current_state[i]]

        p_transition *= self.p_arrive[all_next_states[:, -1]]
        p_transition /= p_transition.sum()

        return self._get_state_id(all_next_states), p_transition, self._get_reward(a)

    def _get_reward(self, a):
        return self.delta[a] * -1

    def get_all_rewards(self):
        return self.delta * -1

    def _get_index_from_state_ids(self, sid):
        f = itemgetter(*sid)
        return list(f(self.id_to_index))


class State:
    ACTION_STAY = 0

    def __init__(self, num_patients_per_class, arrival, num_bed):
        self.num_bed = num_bed
        self.num_patients_per_class = np.array(num_patients_per_class)
        self.arrival = arrival
        self.rep = np.append(self.num_patients_per_class, arrival)

    def all_actions(self):
        if self.num_patients_per_class.sum() + (self.arrival > 0) <= self.num_bed:
            actions = np.array([self.ACTION_STAY])
        else:
            actions = self.num_patients_per_class.nonzero()[0] + 1

        return actions

    def get_id(self):
        tmp = np.arange(len(self.num_patients_per_class) + 1)
        digit = self.num_bed + 1
        return np.sum(self.num_patients_per_class * (digit ** tmp[:-1])) + self.arrival * (digit ** tmp[-1])

    def get_rep(self):
        return self.rep

