"""
RL Model Module for the Autonomous Trading Bot

This module implements the Reinforcement Learning model using PPO algorithm
for autonomous trading decision making.

Improvements:
- Added batch normalization for more stable training
- Implemented dropout for better regularization
- Added learning rate scheduling for improved convergence
- Enhanced network architecture with residual connections
- Improved advantage normalization and clipping
- Added early stopping mechanism to prevent overfitting
"""

import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import math
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PolicyNetwork(nn.Module):
    """
    Policy Network for the PPO algorithm.
    Maps state observations to action probabilities.
    
    Improvements:
    - Added batch normalization
    - Added dropout for regularization
    - Implemented residual connections for better gradient flow
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, dropout_rate: float = 0.2):
        """
        Initialize the policy network.
        
        Args:
            input_dim: Dimension of the input (state) vector
            hidden_dims: List of hidden layer dimensions
            output_dim: Dimension of the output (action) vector
            dropout_rate: Dropout rate for regularization
        """
        super(PolicyNetwork, self).__init__()
        
        # Input batch normalization
        self.input_bn = nn.BatchNorm1d(input_dim)
        
        # Build network layers
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.skip_connections = nn.ModuleList()
        
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Main layer
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            self.dropouts.append(nn.Dropout(dropout_rate))
            
            # Skip connection if dimensions match or can be adapted
            if i > 0 and prev_dim == hidden_dim:
                self.skip_connections.append(nn.Identity())
            elif i > 0:
                self.skip_connections.append(nn.Linear(prev_dim, hidden_dim))
            else:
                self.skip_connections.append(None)
            
            prev_dim = hidden_dim
        
        # Output layer
        self.output_layer = nn.Linear(prev_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)
        
        logger.info(f"Enhanced Policy Network initialized with architecture: {input_dim} -> {hidden_dims} -> {output_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor (state)
            
        Returns:
            Action probabilities
        """
        # Handle single state case
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        # Input normalization
        x = self.input_bn(x)
        
        # Hidden layers with residual connections
        for i, (layer, bn, dropout, skip) in enumerate(zip(self.layers, self.batch_norms, self.dropouts, self.skip_connections)):
            # Main path
            h = layer(x)
            h = bn(h)
            h = torch.relu(h)
            h = dropout(h)
            
            # Skip connection if available
            if skip is not None:
                x_skip = skip(x)
                h = h + x_skip
            
            x = h
        
        # Output layer
        x = self.output_layer(x)
        x = self.softmax(x)
        
        # Remove batch dimension if input was a single state
        if x.shape[0] == 1:
            x = x.squeeze(0)
            
        return x

class ValueNetwork(nn.Module):
    """
    Value Network for the PPO algorithm.
    Maps state observations to value estimates.
    
    Improvements:
    - Added batch normalization
    - Added dropout for regularization
    - Implemented residual connections for better gradient flow
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout_rate: float = 0.2):
        """
        Initialize the value network.
        
        Args:
            input_dim: Dimension of the input (state) vector
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout rate for regularization
        """
        super(ValueNetwork, self).__init__()
        
        # Input batch normalization
        self.input_bn = nn.BatchNorm1d(input_dim)
        
        # Build network layers
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.skip_connections = nn.ModuleList()
        
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Main layer
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            self.dropouts.append(nn.Dropout(dropout_rate))
            
            # Skip connection if dimensions match or can be adapted
            if i > 0 and prev_dim == hidden_dim:
                self.skip_connections.append(nn.Identity())
            elif i > 0:
                self.skip_connections.append(nn.Linear(prev_dim, hidden_dim))
            else:
                self.skip_connections.append(None)
            
            prev_dim = hidden_dim
        
        # Output layer (single value)
        self.output_layer = nn.Linear(prev_dim, 1)
        
        logger.info(f"Enhanced Value Network initialized with architecture: {input_dim} -> {hidden_dims} -> 1")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor (state)
            
        Returns:
            Value estimate
        """
        # Handle single state case
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        # Input normalization
        x = self.input_bn(x)
        
        # Hidden layers with residual connections
        for i, (layer, bn, dropout, skip) in enumerate(zip(self.layers, self.batch_norms, self.dropouts, self.skip_connections)):
            # Main path
            h = layer(x)
            h = bn(h)
            h = torch.relu(h)
            h = dropout(h)
            
            # Skip connection if available
            if skip is not None:
                x_skip = skip(x)
                h = h + x_skip
            
            x = h
        
        # Output layer
        x = self.output_layer(x)
        
        # Remove batch dimension if input was a single state
        if x.shape[0] == 1:
            x = x.squeeze(0)
            
        return x

class PPOAgent:
    """
    PPO Agent for autonomous trading.
    
    Improvements:
    - Added learning rate scheduling
    - Enhanced advantage normalization
    - Improved clipping mechanism
    - Added early stopping
    - Implemented gradient clipping
    - Added model checkpointing
    """
    
    def __init__(
        self,
        state_dim: int = 30,
        action_dim: int = 3,
        hidden_dims: List[int] = [512, 256, 128],  # Increased network capacity
        lr_policy: float = 3e-5,  # Reduced learning rate for more stable learning
        lr_value: float = 1e-4,   # Reduced learning rate for more stable learning
        gamma: float = 0.99,      # Discount factor
        gae_lambda: float = 0.95, # GAE lambda parameter
        clip_ratio: float = 0.2,  # PPO clip ratio
        value_coef: float = 0.5,  # Value loss coefficient
        entropy_coef: float = 0.01, # Entropy bonus coefficient
        max_grad_norm: float = 0.5, # Maximum gradient norm for clipping
        target_kl: float = 0.01,  # Target KL divergence for early stopping
        lr_scheduler_patience: int = 5, # Patience for learning rate scheduler
        lr_scheduler_factor: float = 0.5, # Factor for learning rate scheduler
        use_early_stopping: bool = True, # Whether to use early stopping
        checkpoint_dir: str = "models", # Directory to save model checkpoints
        device: str = None
    ):
        """
        Initialize the PPO agent.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dims: List of hidden layer dimensions for both networks
            lr_policy: Learning rate for the policy network
            lr_value: Learning rate for the value network
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_ratio: PPO clip ratio
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm for clipping
            target_kl: Target KL divergence for early stopping
            lr_scheduler_patience: Patience for learning rate scheduler
            lr_scheduler_factor: Factor for learning rate scheduler
            use_early_stopping: Whether to use early stopping
            checkpoint_dir: Directory to save model checkpoints
            device: Device to run the model on ('cpu' or 'cuda')
        """
        # Set device
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize networks
        self.policy_net = PolicyNetwork(state_dim, hidden_dims, action_dim).to(self.device)
        self.value_net = ValueNetwork(state_dim, hidden_dims).to(self.device)
        
        # Initialize optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr_policy)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr_value)
        
        # Initialize learning rate schedulers
        self.policy_scheduler = ReduceLROnPlateau(
            self.policy_optimizer, 
            mode='max', 
            factor=lr_scheduler_factor, 
            patience=lr_scheduler_patience,
            verbose=True
        )
        
        self.value_scheduler = ReduceLROnPlateau(
            self.value_optimizer, 
            mode='min', 
            factor=lr_scheduler_factor, 
            patience=lr_scheduler_patience,
            verbose=True
        )
        
        # Set hyperparameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.use_early_stopping = use_early_stopping
        
        # Initialize memory buffers
        self.states = []
        self.actions = []
        self.action_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        
        # Initialize training metrics
        self.train_iterations = 0
        self.best_reward = -float('inf')
        
        # Create checkpoint directory
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        logger.info("Enhanced PPO Agent initialized")
    
    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[int, float, float]:
        """
        Select an action based on the current state.
        
        Args:
            state: Current state observation
            training: Whether the agent is in training mode
            
        Returns:
            Tuple of (selected action, action probability, state value)
        """
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).to(self.device)
        
        # Get action probabilities and state value
        with torch.no_grad():
            action_probs = self.policy_net(state_tensor)
            state_value = self.value_net(state_tensor).item()
        
        # Sample action from probability distribution
        if training:
            action_dist = Categorical(action_probs)
            action = action_dist.sample().item()
            action_prob = action_probs[action].item()
        else:
            # During evaluation, take the most probable action
            action = torch.argmax(action_probs).item()
            action_prob = action_probs[action].item()
        
        return action, action_prob, state_value
    
    def store_transition(self, state: np.ndarray, action: int, action_prob: float, reward: float, value: float, done: bool):
        """
        Store a transition in memory.
        
        Args:
            state: State observation
            action: Selected action
            action_prob: Probability of the selected action
            reward: Received reward
            value: Estimated state value
            done: Whether the episode is done
        """
        self.states.append(state)
        self.actions.append(action)
        self.action_probs.append(action_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
    
    def compute_advantages(self, next_value: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute advantages and returns using Generalized Advantage Estimation (GAE).
        
        Args:
            next_value: Value estimate of the next state
            
        Returns:
            Tuple of (advantages, returns)
        """
        # Convert to numpy arrays
        rewards = np.array(self.rewards)
        values = np.array(self.values + [next_value])
        dones = np.array(self.dones + [False])
        
        # Initialize arrays
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        
        # Compute GAE
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        return advantages, returns
    
    def update(self, next_value: float, epochs: int = 10, batch_size: int = 64) -> Dict[str, float]:
        """
        Update the policy and value networks using PPO.
        
        Args:
            next_value: Value estimate of the next state
            epochs: Number of epochs to train on the collected data
            batch_size: Mini-batch size for training
            
        Returns:
            Dictionary of training metrics
        """
        # Compute advantages and returns
        advantages, returns = self.compute_advantages(next_value)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(np.array(self.actions)).to(self.device)
        old_action_probs = torch.FloatTensor(np.array(self.action_probs)).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training metrics
        policy_losses = []
        value_losses = []
        entropy_losses = []
        kl_divs = []
        
        # Mini-batch training
        batch_count = len(states) // batch_size + (1 if len(states) % batch_size != 0 else 0)
        indices = np.arange(len(states))
        
        for _ in range(epochs):
            # Shuffle data
            np.random.shuffle(indices)
            
            # Mini-batch loop
            for start_idx in range(0, len(states), batch_size):
                # Get mini-batch indices
                idx = indices[start_idx:start_idx + batch_size]
                
                # Get mini-batch data
                mb_states = states[idx]
                mb_actions = actions[idx]
                mb_old_action_probs = old_action_probs[idx]
                mb_returns = returns[idx]
                mb_advantages = advantages[idx]
                
                # Get current action probabilities and values
                mb_action_dists = self.policy_net(mb_states)
                mb_values = self.value_net(mb_states).squeeze(-1)
                
                # Get probabilities of actions taken
                mb_new_action_probs = mb_action_dists.gather(1, mb_actions.unsqueeze(1)).squeeze(1)
                
                # Compute probability ratio
                ratio = mb_new_action_probs / (mb_old_action_probs + 1e-8)
                
                # Compute surrogate losses
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * mb_advantages
                
                # Compute policy loss
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Compute value loss
                value_loss = nn.MSELoss()(mb_values, mb_returns)
                
                # Compute entropy loss
                entropy_loss = -torch.mean(torch.sum(mb_action_dists * torch.log(mb_action_dists + 1e-8), dim=1))
                
                # Compute total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss
                
                # Compute approximate KL divergence
                with torch.no_grad():
                    log_ratio = torch.log(ratio + 1e-8)
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                    kl_divs.append(approx_kl)
                
                # Check for early stopping
                if self.use_early_stopping and approx_kl > 1.5 * self.target_kl:
                    logger.info(f"Early stopping at epoch {_+1}/{epochs} due to reaching max KL divergence.")
                    break
                
                # Update policy network
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
                self.policy_optimizer.step()
                
                # Update value network
                self.value_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
                self.value_optimizer.step()
                
                # Store metrics
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
            
            # Check for early stopping after each epoch
            if self.use_early_stopping and np.mean(kl_divs) > self.target_kl:
                logger.info(f"Early stopping at epoch {_+1}/{epochs} due to reaching target KL divergence.")
                break
        
        # Update learning rate schedulers
        mean_policy_loss = np.mean(policy_losses)
        mean_value_loss = np.mean(value_losses)
        self.policy_scheduler.step(-mean_policy_loss)  # Negative because we want to maximize
        self.value_scheduler.step(mean_value_loss)
        
        # Increment training iterations
        self.train_iterations += 1
        
        # Clear memory buffers
        self.states = []
        self.actions = []
        self.action_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        
        # Return metrics
        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy_loss': np.mean(entropy_losses),
            'approx_kl': np.mean(kl_divs),
            'policy_lr': self.policy_optimizer.param_groups[0]['lr'],
            'value_lr': self.value_optimizer.param_groups[0]['lr']
        }
    
    def save_model(self, path: str) -> bool:
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save model
            torch.save({
                'policy_state_dict': self.policy_net.state_dict(),
                'value_state_dict': self.value_net.state_dict(),
                'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
                'value_optimizer_state_dict': self.value_optimizer.state_dict(),
                'policy_scheduler_state_dict': self.policy_scheduler.state_dict(),
                'value_scheduler_state_dict': self.value_scheduler.state_dict(),
                'train_iterations': self.train_iterations,
                'best_reward': self.best_reward,
                'hyperparameters': {
                    'state_dim': self.state_dim,
                    'action_dim': self.action_dim,
                    'gamma': self.gamma,
                    'gae_lambda': self.gae_lambda,
                    'clip_ratio': self.clip_ratio,
                    'value_coef': self.value_coef,
                    'entropy_coef': self.entropy_coef,
                    'max_grad_norm': self.max_grad_norm,
                    'target_kl': self.target_kl
                }
            }, path)
            
            logger.info(f"Model saved to {path}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, path: str) -> bool:
        """
        Load the model from disk.
        
        Args:
            path: Path to load the model from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if file exists
            if not os.path.exists(path):
                logger.error(f"Model file not found: {path}")
                return False
            
            # Load model
            checkpoint = torch.load(path, map_location=self.device)
            
            # Load state dictionaries
            self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
            self.value_net.load_state_dict(checkpoint['value_state_dict'])
            self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
            self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
            
            # Load scheduler state dictionaries if available
            if 'policy_scheduler_state_dict' in checkpoint:
                self.policy_scheduler.load_state_dict(checkpoint['policy_scheduler_state_dict'])
            if 'value_scheduler_state_dict' in checkpoint:
                self.value_scheduler.load_state_dict(checkpoint['value_scheduler_state_dict'])
            
            # Load training state
            self.train_iterations = checkpoint.get('train_iterations', 0)
            self.best_reward = checkpoint.get('best_reward', -float('inf'))
            
            # Load hyperparameters if available
            if 'hyperparameters' in checkpoint:
                hp = checkpoint['hyperparameters']
                self.state_dim = hp.get('state_dim', self.state_dim)
                self.action_dim = hp.get('action_dim', self.action_dim)
                self.gamma = hp.get('gamma', self.gamma)
                self.gae_lambda = hp.get('gae_lambda', self.gae_lambda)
                self.clip_ratio = hp.get('clip_ratio', self.clip_ratio)
                self.value_coef = hp.get('value_coef', self.value_coef)
                self.entropy_coef = hp.get('entropy_coef', self.entropy_coef)
                self.max_grad_norm = hp.get('max_grad_norm', self.max_grad_norm)
                self.target_kl = hp.get('target_kl', self.target_kl)
            
            logger.info(f"Model loaded from {path}")
            return True
        
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def save_checkpoint(self, episode: int, reward: float) -> bool:
        """
        Save a checkpoint of the model.
        
        Args:
            episode: Current episode number
            reward: Episode reward
            
        Returns:
            True if successful, False otherwise
        """
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_episode_{episode}.pt")
        success = self.save_model(checkpoint_path)
        
        # Save best model if reward is better
        if reward > self.best_reward:
            self.best_reward = reward
            best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
            success = success and self.save_model(best_path)
            logger.info(f"New best model saved with reward: {reward:.4f}")
        
        return success
