
import time
import gymnasium as gym
import numpy as np
import random

# Step 1: Create the Taxi environment
env = gym.make("Taxi-v4")

# Step 2: Create a Q-table filled with zeros
# Rows represent states and columns represent actions
Q = np.zeros((env.observation_space.n, env.action_space.n))

# Step 3: Set the learning parameters
alpha, gamma, epsilon = 0.8, 0.95, 1.0

# Store the reward of each episode
rewards = []

print("Training started...")

# Step 4: Train the taxi for 20,000 episodes
for ep in range(20000):

    # Start a new episode
    state, info = env.reset()
    total = 0

    # Allow a maximum of 200 steps in each episode
    for step in range(200):

        # Step 5: Choose an action
        # Sometimes choose a random action (exploration)
        # Otherwise choose the best action from the Q-table (exploitation)
        action = (env.action_space.sample() if random.random() < epsilon
                  else np.argmax(Q[state]))

        # Step 6: Perform the action and get the result
        ns, reward, terminated, truncated, info = env.step(action)

        # Step 7: Update the Q-table using the reward received
        Q[state, action] += alpha * (
            reward + gamma * np.max(Q[ns]) * (not terminated)
            - Q[state, action])

        # Move to the next state
        state = ns

        # Add the reward to the episode's total reward
        total += reward

        # Stop if the episode is finished
        if terminated or truncated:
            break

    # Save the total reward of this episode
    rewards.append(total)

    # Gradually reduce random actions as the taxi learns
    epsilon = max(0.01, epsilon * 0.9995)

    # Display progress after every 2,000 episodes
    if (ep + 1) % 2000 == 0:
        avg_recent = np.mean(rewards[-1000:])
        print(f"Episode {ep+1}/20000 | Average Reward (last 1000): {avg_recent:.2f}")

# Display the average reward after training
print("\nTraining complete! Average reward:", np.mean(rewards[-1000:]))

# Step 8: Run a visual demonstration of the trained taxi
print("\n--- Running a Visual Demo ---")

demo_env = gym.make("Taxi-v4", render_mode="ansi")

# Start the demonstration
state, info = demo_env.reset()
terminated, truncated = False, False

# Display the initial taxi environment
print("Initial Grid State:")
print(demo_env.render())

# Step 9: Let the trained taxi complete its task
while not (terminated or truncated):

    # Choose the best action learned during training
    action = np.argmax(Q[state])

    # Perform the action and get the next state
    state, reward, terminated, truncated, info = demo_env.step(action)

    # Display the updated taxi environment
    print(demo_env.render())

    # Pause briefly so each step can be seen
    time.sleep(0.4)

# Reinforcement Learning (RL) is a type of machine learning in which an
# agent learns to make decisions by interacting with an environment.
# The agent receives rewards or penalties based on its actions and
# learns a strategy to maximize the total cumulative reward.
#
# The Taxi environment is a predefined environment in Gymnasium where
# a taxi must pick up a passenger and drop them at the correct destination.
#
# Agent: The taxi that learns to perform the task.
# Environment: The Taxi grid world in which the agent operates.
# State: The current position of the taxi, passenger and destination.
# Action: An operation performed by the taxi, such as moving or picking
# up or dropping off a passenger.
# Reward: Feedback received after performing an action. Correct actions
# receive positive rewards, while incorrect actions receive penalties.
#
# Q-Learning is a model-free, off-policy Reinforcement Learning algorithm
# that learns the optimal action-value function (Q-values).
# A Q-table stores the expected future rewards for every state-action pair.
#
# Learning Rate (alpha) controls how much new information influences
# the updated Q-value. A higher value gives more importance to new learning.
#
# Discount Factor (gamma) determines the importance of future rewards
# compared to immediate rewards. Its value ranges from 0 to 1.
#
# Epsilon (epsilon) controls the exploration-exploitation trade-off.
# Exploration means choosing random actions to discover new possibilities.
# Exploitation means choosing the best-known action using the Q-table.
#
# Epsilon-Greedy Strategy selects a random action with probability epsilon
# and the best-known action with probability (1 - epsilon).
#
# Q-Learning Update Rule updates the Q-value using the current reward
# and the maximum estimated future Q-value of the next state.
#
# Epsilon Decay gradually reduces random exploration as training progresses,
# allowing the agent to increasingly use its learned knowledge.
#
# Episodes represent complete attempts by the agent to finish the task.
# Each episode consists of multiple steps until termination or truncation.
#
# Training Performance is evaluated using the average reward over recent
# episodes. Higher average rewards generally indicate improved learning.
#
# After training, the agent uses the learned Q-table to select the best
# action in each state and demonstrates the learned taxi behaviour.