import gym
import os
import torch
import torch.nn as nn
from time import time as t
import time


class DQN(nn.Module):
    """
    Deep Q-Network (DQN) architecture for solving reinforcement learning tasks.
    This model uses a feedforward neural network to approximate the Q-function.
    """
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            #! First hidden layer with ReLU activation
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            #! Second hidden layer with ReLU activation
            nn.Linear(256, 256),
            nn.ReLU(),
            #! Output layer predicting Q-values for each action
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        """
        Forward pass through the neural network.

        Args:
            x (torch.Tensor): Input state representation.

        Returns:
            torch.Tensor: Q-values for each possible action.
        """
        return self.fc(x)


def evaluate_model(model_folder="CartPole", model_name="cartpole_dqn_optimized.pth", render_delay=0.02):
    """
    Evaluate a pre-trained DQN model on the CartPole-v1 environment.

    Args:
        model_folder (str): Folder containing the saved model file.
        model_name (str): Name of the saved model file.
        render_delay (float): Delay between rendering frames (in seconds).
    """
    #! Construct the full path to the model file
    model_path = os.path.join(model_folder, model_name)

    #! Check if the model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found. Please provide a valid path.")
        return

    #! Initialize the CartPole environment
    try:
        env = gym.make('CartPole-v1', render_mode="human")
    except Exception as e:
        print(f"Error initializing environment: {e}")
        return

    #! Determine state and action dimensions from the environment
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    #! Load the pre-trained model
    policy_net = DQN(state_dim, action_dim)
    try:
        policy_net.load_state_dict(torch.load(model_path))
        print(f"Model loaded successfully from '{model_path}'.")
    except Exception as e:
        print(f"Error loading model: {e}")
        env.close()
        return

    #! Reset the environment to start evaluation
    state, _ = env.reset()
    done = False
    total_steps = 0
    total_reward = 0

    #! Start a timer to track evaluation duration
    start = t()

    print("Starting evaluation...")
    try:
        while not done:
            total_steps += 1

            #! Convert the state to a tensor for the model
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                #! Predict the best action using the model
                action = policy_net(state_tensor).argmax().item()

            #! Perform the action in the environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            #! Render the environment for visualization
            env.render()
            time.sleep(render_delay)

            #! Update the state to the next state
            state = next_state

    except KeyboardInterrupt:
        #! Handle user interruption gracefully
        print("\nEvaluation interrupted by user.")

    #! Stop the timer and display evaluation results
    end = t()
    print("Evaluation completed.")
    print(f"Total steps balanced: {total_steps}")
    print(f"Total reward accumulated: {total_reward:.2f}")
    print(f"Total balance time: {end - start:.2f} seconds")

    #! Close the environment after evaluation
    env.close()


if __name__ == "__main__":
    #! Specify the folder and model name, then evaluate
    evaluate_model(model_folder="ModelData", model_name="cartpole_dqn_optimized.pth", render_delay=0.02)