# Asynchronous Advantage Actor-Critic (A3C) for Kung-Fu Master

## Overview

This project implements the Asynchronous Advantage Actor-Critic (A3C) algorithm to train an agent to play the game "Kung-Fu Master". A3C is a state-of-the-art deep reinforcement learning algorithm that leverages multiple parallel environments to rapidly train agents.

## System Components

### Model Architecture
The model in this project uses a neural network with both actor and critic heads:
- **Actor Head**: Determines which action to take based on the current state.
- **Critic Head**: Estimates the value function of the current state to assist in training the actor.

### Asynchronous Training
- Multiple agent instances (workers) explore different parts of the environment simultaneously.
- Each worker updates global model parameters asynchronously, which accelerates the learning process.

## Functionality

### Environment Setup
The agent interacts with the "Kung-Fu Master" game environment, receiving pixel data as input and executing actions that affect the game state. The rewards are based on the agent's performance in the game, such as defeating enemies and avoiding attacks.

### Training Process
- **Parallel Execution**: Workers independently explore the environment and accumulate gradients.
- **Global Updates**: After a set number of steps, workers push these gradients to update the global model and sync their local models with the global parameters.

### Reward and Policy Optimization
- The actor head outputs a probability distribution over actions. Actions are sampled from this distribution to encourage exploration.
- The critic provides a baseline to help calculate the advantage, which measures how much better an action is compared to the average.

## Implementation Details

### Network Setup
The network processes input frames from the game using convolutional layers, followed by fully connected layers that split into actor and critic paths.

### Asynchronous Gradient Descent
- Workers compute gradients based on their experiences and asynchronously update a shared model, which is key to the algorithm's efficiency and effectiveness.

## Visualization of Training
Training progress is visualized by plotting the reward curves and other metrics, helping to monitor improvements and diagnose training issues.

## Usage Instructions

To utilize this project:
1. Install necessary Python libraries and dependencies.
2. Run the notebook to initiate the training process with multiple workers.
3. Observe the training progress through plotted metrics and output logs.
4. Change the high score limit or change other training parameters to see how well the Agent can do!
