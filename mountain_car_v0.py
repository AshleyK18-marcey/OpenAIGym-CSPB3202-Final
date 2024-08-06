import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def q_train(episodes, training=True, render=False):
    env = gym.make("MountainCar-v0", render_mode="human" if render else None)

    # Divide position and velocity so I have a tangible state
    pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 20)
    vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 20)

    # initialize Q table if training
    if(training): 
        q_values = np.zeros((len(pos_space), len(vel_space), env.action_space.n))
    else:
        file = open('mountain_car.pkl', 'rb')
        q_values = pickle.load(file)
        file.close()

    alpha = 0.9
    gamma = 0.9

    epsilon = 1
    epsilon_decay = 2/episodes
    rng = np.random.default_rng()
    
    rewards_episode = np.zeros(episodes)
    successful_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0]
        state_p = np.digitize(state[0], pos_space)
        state_v = np.digitize(state[1], vel_space)

        terminated = False
        rewards = 0

        while(not terminated and rewards>-1000):
            if i >= (episodes - 5) or i < 5:
                env.render()
            # Determine action using epsilon greedy strategy
            if training and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_values[state_p,state_v, :])
    
            new_state, reward, terminated, _, _ = env.step(action)
            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vel_space)

            if training:
                q_values[state_p, state_v, action] = q_values[state_p, state_v, action] + alpha * (
                    reward + gamma * np.max(q_values[new_state_p,new_state_v, :]) - q_values[state_p, state_v, action]
                )
            if new_state[0] >= 0.5:
                print('Success on episode {}'.format(i+1))
                successful_episode[i] = 1
            
            state = new_state
            state_p = new_state_p
            state_v = new_state_v
    
            rewards+=reward

        epsilon = max(epsilon - epsilon_decay, 0)
        
        rewards_episode[i] = rewards

    
    env.close()
    if training:
        file = open('mountain_car.pkl', 'wb')
        pickle.dump(q_values,file)
        file.close()
    
    mean_rewards = np.zeros(episodes)
    for t in range(episodes):
        mean_rewards[t] = np.mean(rewards_episode[max(0,t-100):(t+1)])
    plt.plot(mean_rewards)
    highlight_indices = np.where(successful_episode == 1)[0]
    plt.scatter(highlight_indices, mean_rewards[highlight_indices], color='red', zorder=5, label='Successful Episodes')
    plt.title('Average Rewards vs Episodes')
    plt.xlabel('Episodes')
    plt.ylabel('Average Rewards')
    plt.savefig(f'moun.png')

q_train(1000, True, False)