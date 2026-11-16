
import numpy as np
import random
import matplotlib.pyplot as plt

# Create maze (0 = Path, 1 = Wall)
maze = np.array([
    [0, 0, 0, 1, 0],
    [0, 1, 0, 1, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0],
    [1, 1, 0, 0, 0]
])

start = (0, 0)
goal = (4, 4)

# Actions: Up, Right, Down, Left
actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
arrows = ['↑', '→', '↓', '←']

# Initialize Q-table
Q = np.zeros((25, 4))

alpha, gamma, epsilon = 0.8, 0.95, 1.0
rewards = []

# Convert position into state number
def get_state(pos):
    return pos[0] * 5 + pos[1]

# Train the agent
for episode in range(5000):
    pos = start
    total = 0

    for step in range(100):
        state = get_state(pos)

        # Choose random or best action
        if random.random() < epsilon:
            action = random.randint(0, 3)
        else:
            action = np.argmax(Q[state])

        move = actions[action]
        new_pos = (pos[0] + move[0], pos[1] + move[1])

        # Give reward or penalty
        if (new_pos[0] < 0 or new_pos[0] >= 5 or
            new_pos[1] < 0 or new_pos[1] >= 5 or
            maze[new_pos] == 1):
            reward = -5
            new_pos = pos
        elif new_pos == goal:
            reward = 50
        else:
            reward = -1

        # Update Q-table
        next_state = get_state(new_pos)
        Q[state, action] += alpha * (
            reward + gamma * np.max(Q[next_state])
            * (new_pos != goal) - Q[state, action]
        )

        pos = new_pos
        total += reward

        if pos == goal:
            break

    rewards.append(total)
    epsilon = max(0.01, epsilon * 0.995)

print("Training completed!")

# Plot training rewards
avg = [np.mean(rewards[i:i+100])
       for i in range(0, 5000, 100)]

plt.plot(range(100, 5001, 100), avg)
plt.xlabel("Episodes")
plt.ylabel("Average Reward")
plt.title("Q-Learning Training")
plt.grid()
plt.show()

# Display learned policy
print("\nLearned Policy:")
for r in range(5):
    for c in range(5):
        if (r, c) == start:
            print("S", end="  ")
        elif (r, c) == goal:
            print("G", end="  ")
        elif maze[r, c] == 1:
            print("#", end="  ")
        else:
            print(arrows[np.argmax(Q[get_state((r, c))])], end="  ")
    print()

# Find the learned path
pos = start
path = [pos]
visited = {pos}

for step in range(25):
    action = np.argmax(Q[get_state(pos)])
    move = actions[action]
    new_pos = (pos[0] + move[0], pos[1] + move[1])

    if (new_pos[0] < 0 or new_pos[0] >= 5 or
        new_pos[1] < 0 or new_pos[1] >= 5 or
        maze[new_pos] == 1 or new_pos in visited):
        break

    pos = new_pos
    path.append(pos)
    visited.add(pos)

    if pos == goal:
        break

print("\nSample Path:", path)

# Visualize maze and path
plt.imshow(maze, cmap="binary")
for r, c in path:
    plt.scatter(c, r, color="red")

plt.text(0, 0, "S", ha="center", va="center")
plt.text(4, 4, "G", ha="center", va="center")
plt.title("Learned Maze Path")
plt.grid()
plt.show()
