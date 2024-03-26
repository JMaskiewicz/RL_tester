# Reinforcement Learning for Stock Market

## Overview

This project implements a different agent designed for algorithmic trading. The model utilizes multiple innovations in reinforcement learning (RL) to address the complex dynamics of financial markets, focusing on maximizing returns through strategic trade executions. 

## Features

- **Multiple Asset Classes:** The models are tested across different asset classes, including but not limited to indices like SP500/NASDQ, FOREX and cryptocurrencies, to evaluate their performance and adaptability.
- **Diverse Models:** Incorporates a range of reinforcement learning models, such as double deep Q network (DDQN), and Proximal Policy Optimization (PPO), to compare their effectiveness in different financial scenarios.
- **Diverse based Neural Netowrks:** The project uses LSTM, transformers, and standard neural networks to capture different learning patterns and temporal dependencies
- **Move forward optimization:** This project adopts advanced strategies for improving the efficiency and effectiveness of the trading agent's learning process. Through the careful integration of state-of-the-art optimization techniques, it ensures that the agent not only learns faster but also makes more accurate and profitable decisions in the dynamic financial markets.
- **Trading Environment:** Utilizes a custom Gym environment to simulate real-world trading dynamics. It processes market data, manages transactions, and computes rewards based on trading actions, incorporating elements like leverage and transaction costs.
- **Own implementation of RL parts:**  For example features a unique implementation of the Temporal Difference (TD) method and Generalized Advantage Estimation (GAE) for optimizing policy and value functions, improving agent's learning accuracy and stability.
- **Performance Analysis:** Each model's performance is thoroughly analyzed through various metrics like return on investment (ROI), Sharpe Ratio, and drawdowns to gauge their potential in real-world applications.
- **Visualization:** Includes visualization tools to graphically represent the models' performance, asset price movements, and decision-making processes.


## Results

### Transformer-based Proximal Policy Optimization (TPPO) Agent for Algorithmic Trading:

### Double Deep Q Network (DDQN) Agent for Algorithmic Trading:


## Getting Started

To begin with the RL_tester, follow these setup instructions.

### Prerequisites

- Python 3.10 or higher is required.

### Installation

1. **Clone the Repository**: Start by cloning the repository to your local machine to access the project files. Open your terminal or command prompt and run:

    ```bash
    git clone https://github.com/JMaskiewicz/RL_tester/tree/master
    cd RL_tester
    ```

2. **Install Dependencies**: After setting up the virtual environment and cloning the repository, you need to install the required dependencies. Execute:

    ```bash
    pip install -r requirements.txt
    ```

    This will install all necessary libraries, including PyTorch, NumPy, and Pandas.

3. **Run the Agent**: With the dependencies installed, you are ready to run the agent. Run the script:

    ```bash
    python PPO_T_alternative_rewards_1D_SPX.py
    ```

