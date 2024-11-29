
<div align="center">

# 🌟 **CartPole: OpenAI Gym Reinforcement Learning** 🌟  
### *Master the CartPole-v1 environment with Deep Q-Learning and Visualization*

![Build Passing](https://img.shields.io/badge/build-passing-success?style=flat-square)
![Views](https://hits.dwyl.com/alienx5499/CartPole.svg)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat-square)](https://github.com/alienx5499/CartPole/blob/main/CONTRIBUTING.md)
[![License: MIT](https://custom-icon-badges.herokuapp.com/github/license/alienx5499/CartPole?logo=law&logoColor=white)](https://github.com/alienx5499/CartPole/blob/main/LICENSE)
![Security](https://snyk.io/test/github/dwyl/hapi-auth-jwt2/badge.svg?targetFile=package.json)
![⭐ GitHub stars](https://img.shields.io/github/stars/alienx5499/CartPole?style=social)
![🍴 GitHub forks](https://img.shields.io/github/forks/alienx5499/CartPole?style=social)
![Commits](https://badgen.net/github/commits/alienx5499/cartpole)
![🐛 GitHub issues](https://img.shields.io/github/issues/alienx5499/CartPole)
![📂 GitHub pull requests](https://img.shields.io/github/issues-pr/alienx5499/cartpole)
![💾 GitHub code size](https://img.shields.io/github/languages/code-size/alienx5499/cartpole)

🔗 **[Visit the Live Demo](#-screenshots)** | 📑 **[Explore Documentation](#)**

</div>

---

## **🎢 What is CartPole?**

CartPole is a classic reinforcement learning environment from OpenAI Gym where the goal is to balance a pole on a moving cart. This repository includes:
- **CartPole.py**: A script for rendering and evaluating a trained DQN agent.
- **CartPoleAlgorithm.py**: Deep Q-Learning implementation to train the agent on the CartPole-v1 environment.
- **Visualize_q_table.py**: Tools for analyzing and visualizing training metrics like rewards, losses, and epsilon decay.
- **ModelData/**: A folder containing:
  - Pre-trained DQN model (`cartpole_dqn_optimized.pth`).
  - Training logs (`training_logs.json`) for insights.

> *"Master the art of balancing with Deep Q-Learning!"*

---

## **📚 Table of Contents**
1. [✨ Features](#-features)
2. [🛠️ Tech Stack](#️-tech-stack)
3. [📸 Screenshots](#-screenshots)
4. [⚙️ Setup Instructions](#️-setup-instructions)
5. [⚠️ Resource Warning](#-resource-warning)
6. [🎯 Target Audience](#-target-audience)
7. [🤝 Contributing](#-contributing)
8. [📜 License](#-license)

---

## **✨ Features**  
- 🤖 **Deep Q-Learning**: Train a neural network to approximate the Q-value function for optimal policy learning.
- 📊 **Training Logs**: Analyze rewards, epsilon decay, and losses to understand the learning process.
- 📈 **Visualization Tools**: View training progress with clear and detailed graphs.
- 🖥️ **Pre-Trained Model**: Quickly test and render the pre-trained model for instant results.
- 💻 **Modular Codebase**: Separate scripts for training, evaluation, and visualization.

---

## **🛠️ Tech Stack**

### 🌐 **Python Technologies**
- **Reinforcement Learning**: OpenAI Gym
- **Deep Learning**: PyTorch
- **Visualization**: Matplotlib, NumPy
- **Code Management**: JSON for logs, Torch for saving models

### 🛠️ **Scripts and Files**
- **CartPole/**: Folder containing:
  - **`CartPole.py`**: Script for rendering the trained agent and observing its performance.
  - **`CartPoleAlgorithm.py`**: Core DQN training implementation.
  - **`Visualize_q_table.py`**: Tools to visualize training metrics and analyze learning progress.
- **ModelData/**: Folder containing:
  - **`cartpole_dqn_optimized.pth`**: Pre-trained DQN model.
  - **`training_logs.json`**: Saved training metrics for detailed analysis.

---

## **📸 Screenshots**
Here are visualizations showcasing the training process and results:

1. **Total Rewards Per Episode**  
   Visualizes the total rewards collected by the agent over episodes, showing trends and improvement over time.  
   ![Total Rewards Per Episode](https://github.com/user-attachments/assets/7d56aff9-3550-486f-a9b4-fcfb56540167)

2. **Training Loss Over Time**  
   Displays the loss curve over the course of training, providing insights into model convergence.  
   ![Training Loss Over Time](https://github.com/user-attachments/assets/5769ea2c-7a94-41b7-86fc-bad19987dabd)

3. **Epsilon Decay During Training**  
   Highlights how the epsilon value decreases during training, balancing exploration and exploitation.  
   ![Epsilon Decay During Training](https://github.com/user-attachments/assets/cc1687dc-cf37-4ba5-a4fb-74837270706b)

5. **CartPole Agent in Action**  
   Watch the trained agent perform in the CartPole-v1 environment as it successfully balances the pole.  
   ![CartPole Agent in Action](https://github.com/user-attachments/assets/b24d4e5b-2f5b-4237-8c3a-584f89045f30)

---

## **⚙️ Setup Instructions**

1. **Clone the Repository**
   ```bash
   git clone https://github.com/alienx5499/CartPole.git
   ```
2. **Navigate to the Project Directory**
   ```bash
   cd CartPole
   ```
3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run Training Script**
   ```bash
   python CartPoleAlgorithm.py
   ```
5. **Visualize Training Metrics**
   ```bash
   python Visualize_q_table.py
   ```
6. **Render the Trained Agent**
   ```bash
   python CartPole.py
   ```

---

## **⚠️ Resource Warning**

This project includes a pre-trained model trained for **10,000 episodes**, which is **resource-intensive** and can be **CPU/GPU consuming**. 

If you have **limited hardware resources**, we recommend:
- Training with fewer episodes (e.g., **2000–5000 episodes**) to achieve reasonable performance.
- Ensure your system can handle training workloads to prevent hardware strain.

**Alternative**: If you prefer not to train the model, you can use the pre-trained model and logs provided in the **`ModelData/`** folder, which contains data for **10,000 episodes**.

---

## **🎯 Target Audience**

1. **Reinforcement Learning Enthusiasts**: Dive deep into Deep Q-Learning and OpenAI Gym.
2. **AI Researchers**: Analyze and experiment with the classic CartPole environment.
3. **Students and Educators**: Use as a learning tool for understanding reinforcement learning.
4. **Developers**: Expand the repository with new features or algorithms.

---

## **🤝 Contributing**

We ❤️ open source! Contributions are welcome to make this project even better.  
1. Fork the repository.  
2. Create your feature branch.  
   ```bash
   git checkout -b feature/new-feature
   ```
3. Commit your changes.  
   ```bash
   git commit -m "Add a new feature"
   ```
4. Push to the branch and open a pull request.

> Refer to our [CONTRIBUTING.md](CONTRIBUTING.md) for detailed contribution guidelines.

---

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f31f/512.webp" width="35" height="30"> Awesome Contributors

<div align="center">
	<h3>Thank you for contributing to our repository</h3><br>
	<p align="center">
		<a href="https://github.com/alienx5499/CartPole/contributors">
			<img src="https://contrib.rocks/image?repo=alienx5499/CartPole" width="90" height="45" />
		</a>
	</p>
</div>

---

## **📜 License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

<div align="center">

### 📬 **Feedback & Suggestions**
*We value your input! Share your thoughts through [GitHub Issues](https://github.com/alienx5499/CartPole/issues).*


🔗 **[Visit the Live Demo](#-screenshots)** | 📑 **[Explore Documentation](#)** 

---


💡 *Let's master the CartPole challenge together!*

</div>
