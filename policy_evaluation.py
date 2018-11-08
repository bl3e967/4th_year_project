import numpy as np
from RL.lib.envs.gridworld import GridworldEnv

def policy_eval(policy, env, discount_factor = 1.0, theta = 0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.

    Args: 
        policy: [S,A] shaped matrix representing the policy
        env: OpenAI env. env.P represents the transitiion prob. of the environment
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done)
            env.nS is a number of states in the environment
            env.nA is a number of actions in the environment
        theta: the minimum error at which the iteration is terminated.
        discount_factor: gamma discount factor

    Returns: 
        Vector of length env.nS representing the value function.
    """
    # Initialise with a random (all 0) value function
    V = np.zeros(env.nS)

    while True:
        delta = 0
        # for each s in state space:
        for s in range(env.nS):
            v = 0
            # for each action in the policy pi:
            for a, a_prob in enumerate(policy[s]):
                # for each possible state and reward: 
                for prob, next_state, reward, done in env.P[s][a]:
                    v += a_prob*prob*(reward + discount_factor*V[next_state])

            delta = max(delta, np.abs(v - V[s]))
            V[s] = v 
        if delta < theta:
            break
    
    return np.array(V)


def policy_improvement(env, policy_eval_fn = policy_eval, discount_factor = 1.0):
    """
    Policy Improvement Algorithm. Iteratively evaluate and improrve a policy 
    until the optimal policy is found. 

    Args: 
        env: the OpenAI environment
        policy_eval_fn: Evaluation function that takes 3 arguments: 
            policy, env, discount_factor
        discount_factor: Gamma discount factor

    Returns: 
        A tuple (policy, V)
        Policy is the optimal policy, a matrix of shape [S,A] where each state
        contains a valid probability distribution over actions
        V is the value function for the optimal policy
    """
    def one_step_lookahead(s, V):
        """
        Helper function to calculat the value for all action in a given state.

        Args: 
            state: the state to consider (int)
            V: the value to use as an estimator, Vector of length env.nS 
        """
        A = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_state, reward, done in env.P[s][a]:
                A[a] += prob*(reward + discount_factor*V[next_state])
        return A

    # Start with a random policy
    policy = np.ones([env.nS, env.nA])/env.nA 

    while True: 
        # Evaluate the current policy
        V = policy_eval_fn(policy, env, discount_factor)

        # Set to false when the policy is changed
        policy_stable = True
        
        for s in range(env.nS):
            # choose the action which we take under the current policy
            chosen_a = np.argmax(policy[s])
            # Find the best action by one-step lookahead
            action_values = one_step_lookahead(s,V)
            best_a = np.argmax(action_values)

            # Greedily update the policy
            if chosen_a != best_a:
                policy_stable = False
            # update policy
            policy[s] = np.eye(env.nA)[best_a]
    
        if policy_stable:
            return policy, V