import numpy as np


class MDP(object):
    """
    Class to represent the dynamics of a discrete time, discrete state MDP
    This class is for MDPs with possibly different number of actions at different states.
    """

    def __init__(self,
                 list_of_states_and_transitions: list,
                 initial_state_dist: np.ndarray,
                 reward=None):
        """
        :param list_of_states_and_transitions: List that contains the transition probability matrix of every state
        list_of_states_and_transitions[i] is a 2 element tuple for state i.
        list_of_states_and_transitions[i][0] is the 1d array of successor states of state i.
        list_of_states_and_transitions[i][1] is the transition probability matrix of state i.
        list_of_states_and_transitions[i][1][a,j] is the transition probability to j-th successor state when taking action a at state i
        :param initial_state_dist: A 1d array of initial state distribution.
        """
        valid_mdp = True
        self.list_of_states_and_transitions = list_of_states_and_transitions
        self.initial_state_dist = initial_state_dist
        """
        Properties:
        NS: Number of states
        NA: (Maximum) number of actions
        s0: Index of the initial state
        NNext: (Maximum) number of successor states
        NA_list = 
        """

        #Tolerance for numerical check operations
        tol = 1e-6

        # Check whether MDP contains any states
        self.NS = len(self.list_of_states_and_transitions)
        if self.NS < 1:
            valid_mdp = False
            assert valid_mdp, "Number of states is lower than 1."

        # Check whether the initial state is valid
        if any(self.initial_state_dist < 0) or (abs(np.sum(self.initial_state_dist) - 1) > tol):
            valid_mdp = False
            assert valid_mdp, "Initial state distribution is not valid."

        self.NA = int(1)
        self.NSA = int(0)
        self.NNext = int(1)
        self.NA_list = np.zeros((1, self.NS), dtype=int)

        # Check whether the transition probability function is well defined
        for state_index in range(len(self.list_of_states_and_transitions)):
            (succ_state_list, tran_mat) = self.list_of_states_and_transitions[state_index]
            if (type(succ_state_list) != np.ndarray):
                valid_mdp = False
                assert valid_mdp, ('List of successor states are not in the desired format for state ' + str(state_index))
            elif (not issubclass(succ_state_list.dtype.type, np.integer)) or (np.amax(succ_state_list) >= self.NS) or (np.amin(succ_state_list) < 0):
                valid_mdp = False
                assert valid_mdp, ('List of successor states are not in the desired format for state ' + str(state_index))
            else:
                num_of_succ_states = np.size(succ_state_list)
                if succ_state_list.shape == (1, num_of_succ_states):
                    if type(tran_mat) == np.ndarray:
                        (num_of_actions, num_of_outgoing_states) = tran_mat.shape
                        # Check for a size mismatch between the number of successor states
                        if (num_of_actions < 1) or (num_of_outgoing_states != num_of_succ_states):
                            valid_mdp = False
                            assert valid_mdp, ('Transition probability matrix and successor states mismatch for state '+ str(state_index))

                        # Check whether transition matrix is well-defined using probability functions
                        elif (tran_mat < 0).any() or (tran_mat > 1).any() or (abs(1-np.sum(tran_mat, axis=1)) > tol).any():
                            valid_mdp = False
                            assert valid_mdp, ('Transition probabilities are not valid for state ' + str(state_index))

                        else:
                            self.NA = max(self.NA, num_of_actions)
                            self.NNext = max(self.NNext, num_of_succ_states)
                            self.NSA += num_of_actions
                            self.NA_list[0, state_index] = num_of_actions
                    else:
                        valid_mdp = False
                        assert valid_mdp, ('Transition probability matrix is not in the desired format for state ' + str(state_index))

        # Check if reward function is valid
        if reward is not None:
            if self.is_reward_function_valid(reward):
                self.reward = reward
            else:
                valid_mdp = False
                assert valid_mdp, ('Reward function is not valid.')
        else:
            self.reward = None

        assert valid_mdp


    def build_matrix_representation(self):
        """
        Builds the matrix representation of MDP from the MDP object.
        :return: A tuple (tran_mat_mdp, num_of_available_actions, initial_state).
        tran_mat_mdp[s,a,q] is the transiton probability from state s to state q when action a is taken.
        num_of_available_actions[0,s] gives the number available actions at state s
        initial_state is the index of initial_state
        """
        initial_state_dist = self.initial_state_dist
        tran_mat_mdp = np.zeros((self.NS,self.NA,self.NS))
        num_of_available_actions = np.zeros((1,self.NS),int)
        for state_index in range(self.NS):
            (succ_state_list, tran_mat_state) = self.list_of_states_and_transitions[state_index]
            num_of_available_actions[0,state_index] = tran_mat_state.shape[0]
            for action_index in range(num_of_available_actions[0,state_index]):
                for succ_state_index in range(tran_mat_state.shape[1]):
                    tran_mat_mdp[state_index, action_index, succ_state_list[0,succ_state_index]] = tran_mat_state[action_index, succ_state_index]
        return (tran_mat_mdp, num_of_available_actions, initial_state_dist)


    def is_policy_valid(self, policy:list ):
        """
        Checks the validity of a given policy.

        :param policy: A list representing the policy.
        :return: A Boolean representing validity.
        """

        valid_policy = True

        #A tolerance value since the action probabilities may not add up to 1 due to numerical issues
        tol = 1e-10
        if len(policy) != self.NS:
            valid_policy = False

        #Action probabilities must add up to 1 for every state
        for state_index in range(len(self.list_of_states_and_transitions)):
            if self.NA_list[0, state_index] != policy[state_index].size:
                valid_policy = False
            elif (policy[state_index] < 0).any() or (policy[state_index] > 1).any() or (abs(1 - np.sum(policy[state_index], axis=1)) > tol).any():
                print('Action probabilities are not valid for state ', state_index)
                valid_policy = False

        return valid_policy


    def is_reward_function_valid(self, reward):
        """
        Checks the validity of a given policy. Reward/costs must be strictly positive for every state

        :param reward: A list representing the reward.
        :return: A Boolean representing validity.
        """
        valid_reward = True

        if len(reward) != self.NS:
            valid_reward = False

        for state_index in range(len(self.list_of_states_and_transitions)):
            if self.NA_list[0, state_index] != reward[state_index].size:
                valid_reward = False
            elif (reward[state_index] < 0).any():
                #print('Rewards/Costs are not valid for state ', state_index)
                #valid_reward = False
                pass

        return valid_reward


    def sample_a_transition(self, current_state, action):
        """
        Returns a random next state given the state and the action.
        :param current_state: The index of the current state
        :param action: The index of the selected action
        :return: A state that is randomly chosen from the successor states according to the distribution P(current
        _state, action, .)
        """
        probs = self.list_of_states_and_transitions[current_state][1][action, :]
        succ_states = self.list_of_states_and_transitions[current_state][0][0,:]
        next_state = np.random.choice(succ_states, 1, p=probs)[0]
        return next_state

    def find_unique_state_action_index(self, state, action):
        """
        Finds the unique state-action index by enumerating the actions of the previous states.
        :param state: Current state index
        :param action: Current action index among the actions of the state
        :return: A unique state-action index
        """
        state_action = 0
        for s in range(state):
            state_action += self.NA_list[0, s]
        state_action += action
        return state_action

    #TODO A method to remove unreachable states from the MDP
    def remove_unreachable_states(self):
        pass

    def find_reachable_states(self, initial_state_dist=None):
        """
        Finds the set of reachable states given an initial state distribution.

        :param initial_state_dist: A given initial state distribution
        :return: A set of states
        """

        # If the initial state is not specified, use the initial state distribution of the MDP.
        if initial_state_dist is None:
            initial_state_dist = self.initial_state_dist
        initial_states = np.nonzero(initial_state_dist)[0]

        #Finds the set of states using breadth-first search
        reachable_set = set()
        reachable_set.update(initial_states)
        states_to_be_checked = set()
        states_to_be_checked.update(initial_states)
        while len(states_to_be_checked) > 0:
            state = states_to_be_checked.pop()
            new_states_to_be_checked = set(self.list_of_states_and_transitions[state][0].flatten()) - reachable_set
            states_to_be_checked.update(new_states_to_be_checked)
            reachable_set.update(new_states_to_be_checked)
        return reachable_set
