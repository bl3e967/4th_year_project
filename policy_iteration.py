import numpy as np 
from RL.lib.envs.gridworld import GridworldEnv

env = GridworldEnv()

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
    V = np.zeros(env.nS)
    while True:
        delta = 0
        for s in range(env.nS):
            v = 0
            for a, a_prob in enumerate(policy[s]):
                for prob, next_state, reward, done in env.P[s][a]:
                    v += a_prob*prob*(reward + discount_factor*V[next_state])
            
            delta = max(delta,np.abs(v - V[s]))
            V[s] = v
            if delta < theta:s
                break
    
    return np.array(V)

def policy_improvevment(env, policy_eval_fn = policy_eval, disount_-factor = 1.0):
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
    # Initilaise the policy randomly
    policy = np.ones([env.nS, env.nA]) / env.nA
    
    def one_step_lookahead(state, action):
        for a in range(env.nA):
            

    while True: 
        delta = 0
        # evaluate the value of the current policy
        V = policy_eval_fn(policy, env, discount_factor)

        policy_stable = True
        for s in range(env.nS):
            action = policy[s]
            best_policy = one_step_lookahead(s, action)