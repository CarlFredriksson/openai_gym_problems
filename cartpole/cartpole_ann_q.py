import random
import gym
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

ENV_NAME = "CartPole-v1"
NUM_EPISODES = 100
GAMMA = 0.95
LEARNING_RATE = 0.05
EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995

class ANNQAgent():
    def __init__(self, state_space_size, action_space_size):
        self.action_space_size = action_space_size
        self.exploration_rate = EXPLORATION_MAX

        self.model = Sequential()
        self.model.add(Dense(24, input_shape=(state_space_size,), activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(action_space_size, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

    def select_action(self, state):
        # Select action epsilon-greedily
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space_size)
        Q_values = self.model.predict(state)
        return np.argmax(Q_values[0])

    def learn(self, state, action, reward, state_next, done):
        # Update model parameters
        Q_update = reward if done else (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
        Q_values = self.model.predict(state)
        Q_values[0][action] = Q_update
        self.model.fit(state, Q_values, verbose=0)

    def update_exploration_rate(self):
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

def cartpole():
    # Create environment and agent
    env = gym.make(ENV_NAME)
    state_space_size = env.observation_space.shape[0]
    action_space_size = env.action_space.n
    agent = ANNQAgent(state_space_size, action_space_size)

    # Simulate episodes
    scores = np.zeros(NUM_EPISODES)
    for episode in range(NUM_EPISODES):
        state = env.reset()
        state = np.expand_dims(state, axis=0)
        done = False
        while not done:
            #env.render()
            action = agent.select_action(state)
            state_next, reward, done, info = env.step(action)
            scores[episode] += reward
            state_next = np.expand_dims(state_next, axis=0)
            agent.learn(state, action, reward, state_next, done)
            agent.update_exploration_rate()
            state = state_next
        print("Episode: {}, Score: {}".format(str(episode), str(scores[episode])))

    # Plot score
    plt.plot(np.arange(NUM_EPISODES), scores)
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.show()

if __name__ == "__main__":
    cartpole()
