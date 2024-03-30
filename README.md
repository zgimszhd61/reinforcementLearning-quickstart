Here is a quickstart guide to reinforcement learning with complete code examples:

## Introduction to Reinforcement Learning

Reinforcement learning (RL) is a type of machine learning where an agent learns to make decisions by interacting with an environment. The agent receives rewards or penalties for its actions, with the goal of maximizing cumulative rewards over time.

The core elements are:

- **Environment**: The world the agent operates in 
- **State**: The current situation the agent is in
- **Action**: What the agent can do 
- **Reward**: The feedback signal that guides the agent's learning

The agent learns a policy π, which maps states to actions, in order to maximize the expected cumulative reward.

## Q-Learning Example

We'll use the classic CartPole problem from OpenAI Gym to illustrate Q-learning, one of the most popular RL algorithms.

```python
import gym
import numpy as np

# Create the CartPole environment
env = gym.make('CartPole-v1')

# Initialize Q-table with zeros
Q = np.zeros([env.observation_space.shape[0], env.action_space.n])

# Hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.99 # Discount factor
eps = 1.0    # Initial exploration probability

# Training loop
for episode in range(1000):
    # Reset environment 
    state = env.reset()
    done = False
    
    while not done:
        # Explore or exploit
        if np.random.uniform() < eps:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(Q[state])        # Exploit
        
        # Take action and get next state and reward
        next_state, reward, done, _ = env.step(action)
        
        # Update Q-table
        Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
        
        state = next_state
        
    eps *= 0.995  # Decrease exploration over time
    
# Test trained agent
state = env.reset()
done = False
total_reward = 0

while not done:
    action = np.argmax(Q[state])
    state, reward, done, _ = env.step(action)
    total_reward += reward
    env.render()

print(f"Total reward: {total_reward}") [1][2][4]
```

This implements the Q-learning update rule:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \big(r + \gamma \max_{a'} Q(s', a') - Q(s, a)\big)$$

Where:
- `s` is the current state
- `a` is the action taken 
- `r` is the reward received
- `s'` is the new state after taking action `a`
- `\alpha` is the learning rate
- `\gamma` is the discount factor for future rewards

The agent explores randomly at first, but over time learns to exploit the optimal policy encoded in the Q-table.

## Deep Q-Learning

For more complex environments, we can use deep neural networks to approximate the Q-function instead of tables. This is called Deep Q-Networks (DQN).

We'll use the Keras-RL library to build a DQN agent for the CartPole environment:

```python
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

# Build DQN model
model = Sequential()
model.add(Dense(24, input_dim=4, activation='relu')) 
model.add(Dense(24, activation='relu'))
model.add(Dense(2, activation='linear'))
model.compile(loss='mse', optimizer=Adam())

# Define DQN agent
policy = EpsGreedyQPolicy()
memory = SequentialMemory(limit=50000, window_length=1)
dqn = DQNAgent(model=model, policy=policy, memory=memory, nb_actions=2)

# Train the agent
dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)

# Test the trained agent
scores = dqn.test(env, nb_episodes=100, visualize=True)
print(f'Average score over 100 episodes: {np.mean(scores.rewards)}')
```

This builds a simple feedforward neural network to approximate the Q-function, and uses the DQN algorithm to train the weights through experience replay and target network updates.

The Keras-RL library provides a convenient interface for building RL agents using Keras models. [6]

## Summary

This covers the basics of reinforcement learning and Q-learning using Python code examples. Reinforcement learning is a powerful paradigm for solving complex decision-making problems by learning from experience. Simple Q-learning can be extended to deep reinforcement learning using neural networks for more sophisticated environments.

Citations:
[1] https://github.com/vmayoral/basic_reinforcement_learning
[2] https://blog.paperspace.com/getting-started-with-reinforcement-learning/
[3] https://neptune.ai/blog/best-reinforcement-learning-tutorials-examples-projects-and-courses
[4] https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/
[5] https://towardsdatascience.com/reinforcement-learning-with-openai-d445c2c687d2
[6] https://www.youtube.com/watch?v=cO5g5qLrLSo
[7] https://www.youtube.com/watch?v=Mut_u40Sqz4
[8] https://github.com/ZihaoZhouSCUT/Quick-Start-in-Reinforcement-Learning-Algorithm