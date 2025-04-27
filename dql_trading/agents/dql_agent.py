import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class QNetwork(nn.Module):
    """
    Neural network for Deep Q-Learning
    """
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.model(x)


class DQLAgent:
    """
    Deep Q-Learning Agent with experience replay and target network
    """
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=1e-4, epsilon=1.0,
                 epsilon_min=0.01, epsilon_decay=0.995, buffer_size=50000, batch_size=64):
        # Environment dimensions
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Hyperparameters
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate (initial)
        self.epsilon_min = epsilon_min  # Minimum exploration rate
        self.epsilon_decay = epsilon_decay  # Epsilon decay rate
        self.batch_size = batch_size
        
        # Neural networks and optimizer
        self.q_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        
        # Experience replay buffer
        self.memory = deque(maxlen=buffer_size)
        
        # Initialize target network with same weights
        self.update_target()
    
    def select_action(self, state, test=False):
        """
        Select action using epsilon-greedy policy
        
        Parameters:
        -----------
        state : array-like
            Current state
        test : bool
            Whether to use exploration (False) or be deterministic (True)
            
        Returns:
        --------
        int
            Selected action
        """
        # In test mode, we don't do exploration
        if test or np.random.rand() > self.epsilon:
            # Ensure state is correctly shaped for the network
            if isinstance(state, np.ndarray):
                if state.ndim == 1:
                    state = state.reshape(1, -1)
            
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state)
            with torch.no_grad():
                q_values = self.q_net(state_tensor)
            return torch.argmax(q_values).item()
        else:
            # Exploration - random action
            return random.randint(0, self.action_dim - 1)
    
    def update_target(self):
        """
        Update target network with current Q-network weights
        """
        self.target_net.load_state_dict(self.q_net.state_dict())
    
    def store_transition(self, state, action, reward, next_state, done):
        """
        Store transition in replay memory
        
        Parameters:
        -----------
        state : array-like
            Current state
        action : int
            Action taken
        reward : float
            Reward received
        next_state : array-like
            Next state
        done : bool
            Whether episode is done
        """
        # Ensure states are 1D arrays
        if isinstance(state, np.ndarray) and state.ndim > 1:
            state = state.flatten()
        if isinstance(next_state, np.ndarray) and next_state.ndim > 1:
            next_state = next_state.flatten()
            
        # Store transition in memory
        self.memory.append((state, action, reward, next_state, done))
    
    def train(self):
        """
        Train the agent using a random batch from replay memory
        
        Returns:
        --------
        float or None
            Loss value if training occurred, None otherwise
        """
        # Skip if not enough samples
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample random batch
        batch = random.sample(self.memory, self.batch_size)
        
        # Convert to numpy arrays
        states = np.array([t[0] for t in batch])
        actions = np.array([t[1] for t in batch])
        rewards = np.array([t[2] for t in batch])
        next_states = np.array([t[3] for t in batch])
        dones = np.array([t[4] for t in batch], dtype=np.float32)
        
        # Convert to tensors
        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)
        actions = torch.LongTensor(actions)
        
        # Compute current Q values
        curr_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # Compute target Q values using target network
        next_q = self.target_net(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss and perform optimization step
        loss = self.loss_fn(curr_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        
        # Return loss value for tracking
        return loss.item()
    
    def save_model(self, path="dql_model.pth"):
        """
        Save model weights
        """
        torch.save(self.q_net.state_dict(), path)
    
    def load_model(self, path="dql_model.pth"):
        """
        Load model weights
        """
        self.q_net.load_state_dict(torch.load(path))
        self.update_target() 