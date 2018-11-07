import numpy as np

def value_iteration(env, theta = 0.0001, discount_factor = 1.0):
    """
    Value Iteration Algorithm

    Args: 
        env: Op.action_space.nI env. env.P represents the transition probabilities of the environment
            env.P[s][a]: a list of transition tuples (prob, next_state, reward, done)
            env.observation_space.n: number of states in the environment
            env.action_space.n: number of actions in the environment
        theta: error value
        discount_factor: gamma discount factor

    Returns: 
        A tuple (policy, V) of the optimal policy annd the optimal value function. 
    """
    # unwrap environment otherwise error occurs for env.P[s][a]
    env = env.unwrapped
    # Initialise value function V and policy
    V = np.zeros(env.observation_space.n)
    policy = np.zeros([env.observation_space.n], dtype = np.int32)

    def one_step_lookahead(state, V):
        """
        Helper function to calculate the value for all action in a given state.
        
        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.observation_space.n
        
        Returns:
            A vector of length env.action_space.n containing the expected value of each action.
        """
        # initialise expected value container
        v = np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            for prob, next_state, reward, done in env.P[state][a]:
                v[a] += prob*(reward + discount_factor * V[next_state])
        
        return v

    while True: 
        # Initialise delta
        delta = 0
        for s in range(env.observation_space.n):
            v = one_step_lookahead(s,V)
            best_action_value = np.max(v)
            delta = max(delta, np.abs(best_action_value - V[s]))
            V[s] = best_action_value

        if delta < theta: 
            break
    
    # policy improvement
    for s in range(env.observation_space.n):
        v = one_step_lookahead(s,V)
        policy[s] = np.argmax(v)
    
    return policy, V