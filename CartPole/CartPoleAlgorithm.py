import gym
import numpy as np
import random
import json
import os
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim


class DQN(nn.Module):
    """
    Deep Q-Network (DQN) model with three fully connected layers.
    This network approximates the Q-value function.
    """
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            #! Input to first hidden layer with ReLU activation
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            #! Second hidden layer with ReLU activation
            nn.Linear(256, 256),
            nn.ReLU(),
            #! Output layer providing Q-values for all possible actions
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        """
        Forward pass to calculate Q-values.

        Args:
            x (torch.Tensor): Input state.

        Returns:
            torch.Tensor: Q-values for each action.
        """
        return self.fc(x)


class ReplayBuffer:
    """
    Replay buffer for storing and sampling transitions.
    Used to decorrelate consecutive experiences and stabilize training.
    """
    def __init__(self, capacity):
        #! Initialize a fixed-size deque to store transitions
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Add a transition to the buffer.

        Args:
            state (numpy.ndarray): Current state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (numpy.ndarray): Next state.
            done (bool): Whether the episode ended.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Sample a batch of transitions from the buffer.

        Args:
            batch_size (int): Number of samples to retrieve.

        Returns:
            list: A batch of transitions.
        """
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        return batch

    def __len__(self):
        #! Return the current size of the buffer
        return len(self.buffer)


def custom_reward(state, done):
    """
    Custom reward function to encourage the agent to balance the pole.

    Args:
        state (numpy.ndarray): Current state of the environment.
        done (bool): Whether the episode ended.

    Returns:
        float: Computed reward.
    """
    #! Decompose state into components
    position, velocity, angle, angular_velocity = state
    if done:
        #! Large penalty for falling
        return -100
    #! Reward for maintaining small angle and staying near the center
    reward = 1.0 - (abs(angle) / 0.2095)
    reward += 0.5 * (1.0 - (abs(position) / 2.4))
    return reward


def train_dqn():
    """
    Train a Deep Q-Network (DQN) on the CartPole-v1 environment.
    """
    #! Initialize the environment and retrieve state/action dimensions
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    #! Create folder to save the model and logs
    save_dir = "CartPole"
    os.makedirs(save_dir, exist_ok=True)

    #! Initialize DQN models (policy and target networks)
    policy_net = DQN(state_dim, action_dim)
    target_net = DQN(state_dim, action_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()  #! Set target network to evaluation mode

    #! Create the replay buffer
    replay_buffer = ReplayBuffer(10000)
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)

    #! Hyperparameters
    gamma = 0.99  #! Discount factor
    epsilon = 1.0  #! Initial exploration rate
    epsilon_decay = 0.995  #! Decay factor for exploration
    epsilon_min = 0.01  #! Minimum exploration rate
    batch_size = 64  #! Size of training batches
    target_update = 10  #! Frequency of target network updates
    num_episodes = 2  #! Total number of training episodes
    gradient_clip = 1.0  #! Gradient clipping value

    #! Metrics for logging progress
    rewards = []
    losses = []
    epsilon_values = []

    for episode in range(num_episodes):
        #! Reset the environment at the start of each episode
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            #! Epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()  #! Explore (random action)
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    action = policy_net(state_tensor).argmax().item()  #! Exploit (best action)

            #! Take the chosen action and observe the outcome
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            reward = custom_reward(next_state, done)  #! Apply custom reward function
            total_reward += reward

            #! Store the transition in the replay buffer
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state

            #! Train the policy network if sufficient data is available
            if len(replay_buffer) > batch_size:
                batch = replay_buffer.sample(batch_size)
                states, actions, rewards_batch, next_states, dones = zip(*batch)

                #! Convert batch data to tensors
                states = torch.FloatTensor(np.array(states))
                actions = torch.LongTensor(actions).unsqueeze(1)
                rewards_batch = torch.FloatTensor(rewards_batch)
                next_states = torch.FloatTensor(next_states)
                dones = torch.BoolTensor(dones)

                #! Compute current Q-values and target Q-values
                q_values = policy_net(states).gather(1, actions).squeeze()
                next_actions = policy_net(next_states).argmax(dim=1, keepdim=True)
                next_q_values = target_net(next_states).gather(1, next_actions).squeeze()
                next_q_values[dones] = 0.0  #! Zero out Q-values for terminal states
                target = rewards_batch + gamma * next_q_values

                #! Compute and apply the loss
                loss = nn.MSELoss()(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), gradient_clip)
                optimizer.step()
                losses.append(loss.item())

        #! Update the target network periodically
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        #! Decay the exploration rate and adjust the learning rate
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        lr_scheduler.step()

        #! Log metrics for this episode
        rewards.append(total_reward)
        epsilon_values.append(epsilon)

        #! Print progress every 100 episodes
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards[-100:])
            print(f"Episode {episode + 1}/{num_episodes}, Avg Reward: {avg_reward:.2f}, Epsilon: {epsilon:.3f}")

    #! Close the environment after training
    env.close()

    #! Save the trained model and logs
    torch.save(policy_net.state_dict(), os.path.join("ModelData", "cartpole_dqn_optimized.pth"))
    training_logs = {"rewards": rewards, "losses": losses, "epsilon": epsilon_values}
    with open(os.path.join("ModelData", "training_logs.json"), "w") as f:
        json.dump(training_logs, f)
    print("Training complete. Model and logs saved.")


if __name__ == "__main__":
    #! Execute the training function
    train_dqn()