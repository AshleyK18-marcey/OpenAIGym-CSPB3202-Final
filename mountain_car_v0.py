import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

def q_train(episodes):
    env = gym.make("MountainCar-v0", render_mode="human")

    # Divide position and velocity so I have a tangible state
    pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 20)
    vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 20)

    # initialize Q table 
    q = np.zeros((len(pos_space), len(vel_space), env.action_space.n))

    alpha = 0.9
    gamma = 0.9

    epsilon = 1
    epsilon_decay = 2/1000
    rng = np.random.default_rng()
    
    rewards_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0]
        state_p = np.digitize(state[0], pos_space)
        state_v = np.digitize(state[1], vel_space)

        terminated = False
        rewards = 0

        while(not terminated and rewards>-1000):
            if rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state_p,state_v, :])
    
            new_state, reward, terminated, _, _ = env.step(action)
            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vel_space)
        
            q[state_p, state_v, action] = q[state_p, state_v, action] + alpha * (
                reward + gamma * np.max(q[new_state_p,new_state_v, :]) - q[state_p, state_v, action]
            )
    
            state = new_state
            state_p = new_state_p
            state_v = new_state_v
    
            rewards+=reward
        epsilon = max(epsilon - epsilon_decay, 0)
        
        rewards_episode[i] = rewards
    
    env.close()
    
    mean_rewards = np.zeros(episodes)
    for t in range(episodes):
        mean_rewards[t] = np.mean(rewards_episode[max(0,t-100):(t+1)])
    plt.plot(mean_rewards)
    plt.savefig(f'moun.png')

q_train(10)