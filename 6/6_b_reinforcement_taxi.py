import time
import gymnasium as gym
import numpy as np
import random

# Note: 'Taxi-v3' is the standard Gymnasium ID
env = gym.make("Taxi-v4")
Q = np.zeros((env.observation_space.n, env.action_space.n))

alpha, gamma, epsilon = 0.8, 0.95, 1.0
rewards = []

print("Training started...")
for ep in range(20000):
    state, info = env.reset()
    total = 0

    for step in range(200):
        action = (env.action_space.sample() if random.random() < epsilon
                  else np.argmax(Q[state]))

        ns, reward, terminated, truncated, info = env.step(action)
        Q[state, action] += alpha * (
            reward + gamma * np.max(Q[ns]) * (not terminated)
            - Q[state, action])

        state = ns
        total += reward

        if terminated or truncated:
            break

    rewards.append(total)
    epsilon = max(0.01, epsilon * 0.9995)

# --- DISPLAY LOGIC: Print training progress every 2,000 episodes ---
    if (ep + 1) % 2000 == 0:
        avg_recent = np.mean(rewards[-1000:])
        print(f"Episode {ep+1}/20000 | Average Reward (last 1000): {avg_recent:.2f}")

print("\nTraining complete! Average reward:", np.mean(rewards[-1000:]))

# --- DISPLAY LOGIC: Visual Demo of the Trained Agent ---
print("\n--- Running a Visual Demo ---")
demo_env = gym.make("Taxi-v4", render_mode="ansi")
state, info = demo_env.reset()
terminated, truncated = False, False

print("Initial Grid State:")
print(demo_env.render())

while not (terminated or truncated):
    # Pure exploitation: pick the best learned action (no random exploration)
    action = np.argmax(Q[state])
    state, reward, terminated, truncated, info = demo_env.step(action)
    
    print(demo_env.render())
    time.sleep(0.4) # Brief pause so you can watch each step in the terminal