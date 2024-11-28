import json
import os
import matplotlib.pyplot as plt
import numpy as np


def load_data(file_path):
    """
    Load training data from a JSON file.

    Args:
        file_path (str): Path to the JSON file containing training logs.
    
    Returns:
        dict: Dictionary containing training data.
    """
    #! Check if the specified file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: File '{file_path}' not found. Please ensure it exists in the 'CartPole' folder.")
    
    #! Open and load the JSON file
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def plot_rewards(rewards, title="Rewards per Episode"):
    """
    Plot total rewards per episode.

    Args:
        rewards (list): List of total rewards per episode.
        title (str): Title of the plot.
    """
    #! Create a new figure for the rewards plot
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, label="Total Reward")
    plt.xlabel("Episode")  #! Label for the x-axis
    plt.ylabel("Total Reward")  #! Label for the y-axis
    plt.title(title)  #! Title of the plot
    plt.legend()  #! Display legend
    plt.grid()  #! Add grid lines for better readability
    plt.show()  #! Render the plot


def plot_loss(losses, title="Loss Over Time"):
    """
    Plot loss values during training.

    Args:
        losses (list): List of loss values.
        title (str): Title of the plot.
    """
    #! Check if loss data is available
    if not losses:
        print("Warning: No loss data available to plot.")
        return
    
    #! Create a new figure for the loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label="Loss", color="orange")
    plt.xlabel("Training Step")  #! Label for the x-axis
    plt.ylabel("Loss")  #! Label for the y-axis
    plt.title(title)  #! Title of the plot
    plt.legend()  #! Display legend
    plt.grid()  #! Add grid lines for better readability
    plt.show()  #! Render the plot


def plot_epsilon(epsilon_values, title="Epsilon Decay Over Time"):
    """
    Plot the epsilon decay curve.

    Args:
        epsilon_values (list): List of epsilon values during training.
        title (str): Title of the plot.
    """
    #! Check if epsilon data is available
    if not epsilon_values:
        print("Warning: No epsilon data available to plot.")
        return
    
    #! Create a new figure for the epsilon decay plot
    plt.figure(figsize=(10, 6))
    plt.plot(epsilon_values, label="Epsilon Value", color="green")
    plt.xlabel("Episode")  #! Label for the x-axis
    plt.ylabel("Epsilon")  #! Label for the y-axis
    plt.title(title)  #! Title of the plot
    plt.legend()  #! Display legend
    plt.grid()  #! Add grid lines for better readability
    plt.show()  #! Render the plot


def main():
    """
    Main function to load training data and plot performance metrics.
    """
    #! Specify the folder and paths to your training log data
    folder = "ModelData"
    log_file = os.path.join(folder, "training_logs.json")

    try:
        #! Load the training logs
        data = load_data(log_file)
        print(f"Training data loaded successfully from '{log_file}'.")

        #! Plot metrics: Rewards, Loss, and Epsilon
        plot_rewards(data.get("rewards", []), title="Total Rewards Per Episode")
        plot_loss(data.get("losses", []), title="Training Loss Over Time")
        plot_epsilon(data.get("epsilon", []), title="Epsilon Decay During Training")

    except FileNotFoundError as e:
        #! Handle cases where the training logs file is missing
        print(e)


if __name__ == "__main__":
    #! Execute the main function
    main()