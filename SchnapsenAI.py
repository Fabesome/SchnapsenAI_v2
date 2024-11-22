import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os
import pickle
from collections import deque
from card import Card  # Import Card from card.py instead of main.py
import random

class SchnapsenAI: 
    def __init__(self):
        # Define input shape based on state representation
        input_shape = 160  # Verify this matches your state representation
        
        # Create model with correct output size (20 cards + close_deck action = 21 total)
        inputs = layers.Input(shape=(input_shape,))
        x = layers.Dense(256, activation='relu')(inputs)
        x = layers.Dense(128, activation='relu')(x)
        outputs = layers.Dense(21, activation='softmax')(x)  # 20 cards + close_deck
        
        self.model = models.Model(inputs=inputs, outputs=outputs)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.model.compile(optimizer=self.optimizer, loss='categorical_crossentropy')
        
        # Initialize memory buffer
        self.memory = deque(maxlen=10000)

    def get_state(self, game, player, hand, led_card=None):
        # Ensure state vector matches input_shape
        state = np.zeros(160)  # Should match input_shape above
        # ... rest of state encoding ...
        return state

    def calculate_reward(self, game, player, action_type, trick_points):
        """Calculate reward for an action with enhanced trump and high-card strategy"""
        reward = trick_points / 66.0  # Base reward
        
        if action_type == 'play_card':
            # Extra reward for winning with trump
            if game.trick_history and game.trick_history[-1][2] == player:  # If player won the trick
                winning_card = game.trick_history[-1][0] if player == 1 else game.trick_history[-1][1]
                if winning_card.suit == game.trump_suit:
                    reward += 0.2  # Bonus for effective trump use
                
            # Penalty for giving away high-point cards
            if game.trick_history and game.trick_history[-1][2] != player:  # If player lost the trick
                lost_card = game.trick_history[-1][0] if player == 1 else game.trick_history[-1][1]
                if game.points[lost_card.rank] >= 10:  # 10s and Aces
                    reward -= 0.3  # Penalty for losing high-value cards
        
        return reward

    def save_model(self, filename):
        """Save the model and memory buffer"""
        # Change file extension from .h5 to .keras
        filename = filename.replace('.h5', '.keras')
        model_dir = os.path.dirname(filename)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # Save model in new Keras format
        self.model.save(filename)
        
        # Save memory buffer
        memory_file = filename.replace('.keras', '_memory.pkl')
        with open(memory_file, 'wb') as f:
            pickle.dump(self.memory, f)

    def load_model(self, filename):
        """Load model and memory buffer"""
        # Update file extension
        filename = filename.replace('.h5', '.keras')
        
        # Load model
        self.model = tf.keras.models.load_model(filename)
        
        # Load memory buffer if it exists
        memory_file = filename.replace('.keras', '_memory.pkl')
        if os.path.exists(memory_file):
            with open(memory_file, 'rb') as f:
                self.memory = pickle.load(f)

    def choose_action(self, state, possible_actions, game):
        """Choose an action using the neural network"""
        # Convert state to tensor
        state_tensor = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
        
        # Get action probabilities from model
        action_probs = self.model.predict(state_tensor, verbose=0)[0]
        
        # Create mask for legal actions (20 cards + close_deck = 21)
        legal_actions_mask = np.zeros(21)
        
        # Add card actions to mask
        for action in possible_actions:
            if isinstance(action, Card):
                card_index = self.card_to_index(action)
                legal_actions_mask[card_index] = 1
                
        # Add close_deck action if legal
        if game.can_close_deck(game.current_player):
            legal_actions_mask[20] = 1  # Index 20 is close_deck
        
        # Apply mask and renormalize
        masked_probs = action_probs * legal_actions_mask
        if np.sum(masked_probs) > 0:
            masked_probs = masked_probs / np.sum(masked_probs)
        else:
            # If all probabilities are zero, use uniform distribution over legal actions
            masked_probs = legal_actions_mask / np.sum(legal_actions_mask)
        
        # Choose action based on probabilities
        action_index = np.random.choice(len(masked_probs), p=masked_probs)
        
        # Convert index back to action
        if action_index == 20:
            return ('close_deck', None)
        else:
            return ('play_card', self.index_to_card(action_index))
            
    def card_to_index(self, card):
        """Convert a card to its index in the action space"""
        suit_index = {'Hearts': 0, 'Diamonds': 1, 'Clubs': 2, 'Spades': 3}
        rank_index = {'Jack': 0, 'Queen': 1, 'King': 2, '10': 3, 'Ace': 4}
        return suit_index[card.suit] * 5 + rank_index[card.rank]
        
    def index_to_card(self, index):
        """Convert an index back to a card"""
        suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
        ranks = ['Jack', 'Queen', 'King', '10', 'Ace']
        suit_index = index // 5
        rank_index = index % 5
        return Card(suits[suit_index], ranks[rank_index])

    def learn(self, state, action, reward, next_state):
        """Update the model based on experience"""
        # Store experience in memory
        self.memory.append((state, action, reward, next_state))
        
        # Start learning when we have enough samples
        if len(self.memory) < 32:  # batch size of 32
            return
        
        # Sample random batch from memory
        batch = random.sample(self.memory, 32)
        
        states = []
        targets = []
        
        for state, action, reward, next_state in batch:
            # Convert state to tensor
            state_tensor = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
            next_state_tensor = tf.convert_to_tensor(next_state.reshape(1, -1), dtype=tf.float32)
            
            # Get current Q values
            target = self.model.predict(state_tensor, verbose=0)[0]
            
            # Get next Q values
            next_q_values = self.model.predict(next_state_tensor, verbose=0)[0]
            
            # Get action index
            if isinstance(action[1], Card):
                action_idx = self.card_to_index(action[1])
            else:  # close_deck action
                action_idx = 20
            
            # Update Q value for the taken action
            target[action_idx] = reward + 0.95 * np.max(next_q_values)  # gamma = 0.95
            
            states.append(state)
            targets.append(target)
        
        # Convert to numpy arrays
        states = np.array(states)
        targets = np.array(targets)
        
        # Train the model
        self.model.fit(states, targets, epochs=1, verbose=0)
