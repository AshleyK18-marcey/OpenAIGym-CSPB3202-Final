import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Q learning algorithm
# Parameters: 
# episodes - the number of 200 step episodes you want to train the agent for
# alpha - learning rate 
# gamma - Discount factor
# training - if true then the q table is reset and the agent is trained, 
# if false then the agent will use the saved q table to complete the control problem
# render - if true then pygame will render the env, if false then it will run without a visualization (saves time)
def q_train(episodes, alpha = 0.1, gamma = 0.99, training=True, render=False):
    # Initialize the mountain car environment
    env = gym.make("MountainCar-v0", render_mode="human" if render else None)

    # Create bins to discretize the continuous state space for position and velocity
    pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 20) # Observation space is -1.2 and 0.6
    vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 20) # Observation space is -0.07 and 0.07

    # Initialize Q table if training, otherwise load the pre-trained Q-table
    if(training): 
        q_table = np.zeros((len(pos_space), len(vel_space), env.action_space.n))
    else:
        file = open('mountain_car.pkl', 'rb')
        q_table = pickle.load(file)
        file.close()

    # Epsilon-greedy parameters
    epsilon = 1 # Initial exploration rate, 100% random actions at first
    epsilon_decay = 2/episodes # Epsilon decay rate
    rng = np.random.default_rng() # random number generator
    
    # Initialize arrays to track rewards and successful episodes
    rewards_episode = np.zeros(episodes)
    successful_episode = np.zeros(episodes)
    steps_per_episode = np.zeros(episodes)

    for i in range(episodes):
        # Reset the environment to start a new episode
        state = env.reset()[0]
        
        # Discretize the continuous state into indices for the Q-table
        state_p = np.digitize(state[0], pos_space)
        state_v = np.digitize(state[1], vel_space)

        terminated = False # If goal is reached then this will be true
        rewards = 0
        # capped the penalty of rewards to allow for neater plotting
        while(not terminated and rewards > -1000):

            # Determine action using epsilon greedy strategy
            if training and rng.random() < epsilon:
                # take random action (0 = left, 1= do nothing, 2= right)
                # exploration
                action = env.action_space.sample()
            else:
                # use what has been learned from the q table
                # exploitation
                action = np.argmax(q_table[state_p,state_v, :])

            # Take action
            new_state, reward, terminated, _, _ = env.step(action)
            steps_per_episode[i] += 1
            
            # Discretize the new state
            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vel_space)

            if training:
                # Update the Q table
                q_table[state_p, state_v, action] = q_table[state_p, state_v, action] + alpha * (
                    reward + gamma * np.max(q_table[new_state_p,new_state_v, :]) - q_table[state_p, state_v, action]
                )
            
            # Check if the car has reached the goal
            if new_state[0] >= 0.5:
                print('Success on episode {}'.format(i+1))
                print('Success after {} steps'.format(steps_per_episode[i]))
                successful_episode[i] = 1
            
            # Update state and rewards for the next step
            state = new_state
            state_p = new_state_p
            state_v = new_state_v
            rewards+=reward

        # Decay epsilon to reduce exploration over time
        epsilon = max(epsilon - epsilon_decay, 0)
        # Track total rewards for the episode
        rewards_episode[i] = rewards

    env.close()
    
    # Save the Q table after training to a binary file
    if training:
        file = open('mountain_car.pkl', 'wb')
        pickle.dump(q_table,file)
        file.close()
    
    # Plot the average rewards over a window of 100 episodes
    mean_rewards = np.zeros(episodes)
    for t in range(episodes):
        mean_rewards[t] = np.mean(rewards_episode[max(0,t-100):(t+1)])
    plt.plot(mean_rewards)
    
    # Highlight the successful episodes on the plot
    highlight_indices = np.where(successful_episode == 1)[0]
    plt.scatter(highlight_indices, mean_rewards[highlight_indices], color='green', label='Successful Episodes', linewidths=0.1)
    
    plt.title('Average Rewards vs Episodes')
    plt.xlabel('Episodes')
    plt.ylabel('Average Rewards')
    plt.savefig(f'moun.png')

# Call the Q-Learning algorithm
q_train(10000, 0.1, 0.99, True, False)