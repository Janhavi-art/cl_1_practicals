
import numpy as np
import random

# Step 1: Tic-Tac-Toe environment
class TicTacToe:
    def reset(self):
        self.board = [0] * 9

    def moves(self):
        return [i for i in range(9) if self.board[i] == 0]

    def win(self, p):
        lines = [(0,1,2), (3,4,5), (6,7,8),
                 (0,3,6), (1,4,7), (2,5,8),
                 (0,4,8), (2,4,6)]
        return any(all(self.board[i] == p for i in line)
                   for line in lines)


# Step 2: Q-learning agent
class Agent:
    def __init__(self):
        self.Q = np.zeros((3**9, 9))
        self.alpha = 0.2
        self.gamma = 0.9
        self.epsilon = 1.0

    def state(self, board):
        return sum((3**i) * (v+1) for i, v in enumerate(board))

    def choose(self, board, moves):
        s = self.state(board)
        if random.random() < self.epsilon:
            return random.choice(moves)
        return moves[np.argmax(self.Q[s, moves])]

    def update(self, s, a, reward, next_s=None):
        future = 0 if next_s is None else np.max(self.Q[next_s])
        self.Q[s, a] += self.alpha * (
            reward + self.gamma * future - self.Q[s, a])


# Step 3: Train the agent against a random opponent
def train(agent, episodes=20000):
    env = TicTacToe()

    for _ in range(episodes):
        env.reset()
        pending = None

        while True:
            # Agent plays as X
            s = agent.state(env.board)
            if pending:
                ps, pa = pending
                agent.update(ps, pa, 0, s)
                pending = None

            a = agent.choose(env.board, env.moves())
            env.board[a] = 1

            if env.win(1):
                agent.update(s, a, 1)
                break

            if not env.moves():
                agent.update(s, a, 0)
                break

            pending = (s, a)

            # Random opponent plays as O
            b = random.choice(env.moves())
            env.board[b] = -1

            if env.win(-1):
                agent.update(s, a, -1)
                break

            if not env.moves():
                agent.update(s, a, 0)
                break

        agent.epsilon = max(0.05, agent.epsilon * 0.9997)


# Step 4: Test the trained agent
def test(agent, episodes=100):
    env = TicTacToe()
    results = [0, 0, 0]  # Wins, losses, draws
    agent.epsilon = 0

    for _ in range(episodes):
        env.reset()

        while True:
            a = agent.choose(env.board, env.moves())
            env.board[a] = 1

            if env.win(1):
                results[0] += 1
                break

            if not env.moves():
                results[2] += 1
                break

            b = random.choice(env.moves())
            env.board[b] = -1

            if env.win(-1):
                results[1] += 1
                break

            if not env.moves():
                results[2] += 1
                break

    return results


# Step 5: Run training and testing
agent = Agent()

print("Training started...")
train(agent)
print("Training completed!")

result = test(agent)

print("Wins:", result[0])
print("Losses:", result[1])
print("Draws:", result[2])
