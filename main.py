import random
import numpy as np
from collections import defaultdict, deque
import tensorflow as tf
from tensorflow.keras import layers, models
import os
import json
import pickle
from tqdm import tqdm
import threading
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import time
from card import Card
from SchnapsenAI import SchnapsenAI


# Schnapsen game class
class SchnapsenGame:
    def __init__(self):
        self.suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
        self.ranks = ['Jack', 'Queen', 'King', '10', 'Ace']
        self.points = {'Jack': 2, 'Queen': 3, 'King': 4, '10': 10, 'Ace': 11}
        self.deck = self.create_deck()
        self.trump_suit = None
        self.player1_hand = []
        self.player2_hand = []
        self.player1_points = 0
        self.player2_points = 0
        self.trump_marriage_bonus = 40
        self.regular_marriage_bonus = 20
        self.game_points1 = 0  # Points for winning sub-rounds (up to 7)
        self.game_points2 = 0
        self.deck_closed = False
        self.deck_closer = None  # Player who closed the deck
        self.talon = []  # Remaining cards in deck
        self.played_cards = set()  # Track all played cards
        self.trick_history = []    # Track all tricks
        self.marriage_history = [] # Track marriage declarations
        self.first_trick_leader = None
        self.tricks_won = {1: 0, 2: 0}  # Track tricks won by each player
        self.current_player = 1
        
    def create_deck(self):
        return [Card(suit, rank) for suit in self.suits for rank in self.ranks]
    
    def deal_cards(self):
        """Deal initial cards and set trump suit"""
        random.shuffle(self.deck)
        self.player1_hand = self.deck[:5]
        self.player2_hand = self.deck[5:10]
        self.talon = self.deck[10:]  # Set talon first
        self.trump_suit = self.talon[0].suit  # Use first card in talon for trump suit
        
    def play_card(self, player, card):
        """Play a card from player's hand"""
        # Find the matching card in the player's hand
        if player == 1:
            matching_cards = [c for c in self.player1_hand 
                             if c.suit == card.suit and c.rank == card.rank]
            if matching_cards:
                self.player1_hand.remove(matching_cards[0])
            else:
                raise ValueError(f"Card {card} not in Player 1's hand: {self.player1_hand}")
        else:
            matching_cards = [c for c in self.player2_hand 
                             if c.suit == card.suit and c.rank == card.rank]
            if matching_cards:
                self.player2_hand.remove(matching_cards[0])
            else:
                raise ValueError(f"Card {card} not in Player 2's hand: {self.player2_hand}")
            
        self.played_cards.add(card)
        
        # If this is the first card of the first trick
        if not self.trick_history and not self.first_trick_leader:
            self.first_trick_leader = player
            
    def calculate_trick_winner(self, card1, card2):
        if card1.suit == card2.suit:
            return 1 if self.points[card1.rank] > self.points[card2.rank] else 2
        elif card2.suit == self.trump_suit:
            return 2
        else:
            return 1
    
    def check_marriage(self, hand, suit):
        has_king = any(card.suit == suit and card.rank == 'King' for card in hand)
        has_queen = any(card.suit == suit and card.rank == 'Queen' for card in hand)
        
        if has_king and has_queen:
            return self.trump_marriage_bonus if suit == self.trump_suit else self.regular_marriage_bonus
        return 0
    
    def calculate_trick_points(self, card1, card2):
        # Calculate points from the cards in the trick
        return self.points[card1.rank] + self.points[card2.rank]
    
    def reset_sub_round(self):
        self.deck = self.create_deck()
        self.player1_hand = []
        self.player2_hand = []
        self.player1_points = 0
        self.player2_points = 0
        self.deck_closed = False
        self.deck_closer = None
        
        # Deal cards and set talon only once
        random.shuffle(self.deck)
        self.player1_hand = self.deck[:5]
        self.player2_hand = self.deck[5:10]
        self.talon = self.deck[10:]  # Set talon
        self.trump_suit = self.talon[0].suit  # Set trump suit based on first talon card
        
        self.played_cards.clear()
        self.trick_history.clear()
        self.marriage_history.clear()
        self.first_trick_leader = None
        self.tricks_won = {1: 0, 2: 0}
        
    def calculate_game_points(self, winner_points, loser_points):
        if loser_points == 0:
            return 3  # Opponent didn't win any tricks
        elif loser_points < 33:
            return 2  # Opponent won tricks but scored less than 33
        else:
            return 1  # Opponent scored 33 or more
    
    def can_close_deck(self, player):
        # Can only close if deck is still open and player has at least one trick
        player_points = self.player1_points if player == 1 else self.player2_points
        return not self.deck_closed and player_points > 0
        
    def close_deck(self, player):
        if self.can_close_deck(player):
            self.deck_closed = True
            self.deck_closer = player
            return True
        return False
    
    def get_legal_moves(self, player, hand, led_card=None):
        if not self.deck_closed or led_card is None:
            return hand  # Can play any card if deck is open or leading
            
        # If deck is closed, must follow suit if possible
        matching_suits = [card for card in hand if card.suit == led_card.suit]
        if matching_suits:
            # Must beat the led card if possible
            winning_cards = [card for card in matching_suits 
                           if self.points[card.rank] > self.points[led_card.rank]]
            return winning_cards if winning_cards else matching_suits
            
        # If can't follow suit, must trump if possible
        if led_card.suit != self.trump_suit:
            trump_cards = [card for card in hand if card.suit == self.trump_suit]
            if trump_cards:
                return trump_cards
                
        # If can't follow suit or trump, can play any card
        return hand
        
    def draw_cards(self):
        if not self.deck_closed and self.talon:
            self.player1_hand.append(self.talon.pop(0))
            self.player2_hand.append(self.talon.pop(0))

    def exchange_trump_jack(self, player, hand):
        """Allow player to exchange trump jack with face-up trump card"""
        if (not self.deck_closed and len(self.talon) > 0 and 
            any(card.suit == self.trump_suit and card.rank == 'Jack' for card in hand)):
            # Find trump jack in hand
            trump_jack = next(card for card in hand 
                            if card.suit == self.trump_suit and card.rank == 'Jack')
            # Get face-up trump card
            face_up_trump = self.talon[-1]
            # Exchange cards
            if player == 1:
                self.player1_hand.remove(trump_jack)
                self.player1_hand.append(face_up_trump)
            else:
                self.player2_hand.remove(trump_jack)
                self.player2_hand.append(face_up_trump)
            self.talon[-1] = trump_jack
            return True
        return False

    def handle_last_trick(self, winner):
        """Handle last trick when deck is exhausted"""
        if not self.deck_closed and len(self.talon) == 0:
            if winner == 1:
                self.player1_points = 66
            else:
                self.player2_points = 66

    def _get_played_cards(self, game):
        """Return set of all cards that have been played."""
        return self.played_cards
        
    def _was_marriage_declared(self):
        """Check if a marriage was declared in the current trick."""
        if not self.marriage_history:
            return False
        return self.marriage_history[-1][0] == len(self.trick_history)
        
    def _has_no_tricks(self, player):
        """Return True if player hasn't won any tricks yet."""
        return self.tricks_won[player] == 0
        
    def _calculate_closer_points(self):
        """Calculate potential points for deck closer"""
        if not self.deck_closed:
            return 0
        
        non_closer = 2 if self.deck_closer == 1 else 1
        non_closer_points = self.player2_points if self.deck_closer == 1 else self.player1_points
        
        if non_closer_points == 0:
            return 3
        elif non_closer_points < 33:
            return 2
        else:
            return 1

    def _calculate_non_closer_points(self):
        """Calculate potential points for non-closer"""
        if not self.deck_closed:
            return 0
        
        non_closer = 2 if self.deck_closer == 1 else 1
        if self.tricks_won[non_closer] == 0:
            return 3
        else:
            return 2
            
    def _led_first_trick(self, player):
        """Return True if player led the first trick of the current deal."""
        return self.first_trick_leader == player
        
    def _get_known_opponent_cards(self, game, opponent_hand):
        """Return binary vector of known opponent cards"""
        known_cards = [0] * 22  # 11 cards * 2 players
        
        # Cards known from marriages
        for trick_num, player, suit in self.marriage_history:
            if player != self.current_player:
                # Mark both Queen and King as known
                queen_idx = self._get_card_index(suit, 'Queen')
                king_idx = self._get_card_index(suit, 'King')
                known_cards[queen_idx] = 1
                known_cards[king_idx] = 1
        
        # Trump Jack exchanges
        if hasattr(self, 'trump_exchange_happened') and self.trump_exchange_happened:
            jack_idx = self._get_card_index(self.trump_suit, 'Jack')
            known_cards[jack_idx + 11] = 1  # Offset for second player
        
        # Last face-up trump card
        if self.talon and len(self.talon) == 1:
            last_card = self.talon[0]
            card_idx = self._get_card_index(last_card.suit, last_card.rank)
            known_cards[card_idx] = 1
            
        return known_cards
        
    def _get_last_n_tricks(self, n):
        """Return the last n tricks played."""
        return self.trick_history[-n:] if self.trick_history else []
        
    def _get_card_index(self, suit, rank):
        """Helper method to get index of a card in the binary representation."""
        suit_idx = self.suits.index(suit)
        rank_idx = self.ranks.index(rank)
        return suit_idx * 5 + rank_idx
        
    def declare_marriage(self, player, suit):
        """Declare a marriage and update tracking."""
        if self.check_marriage(self.player1_hand if player == 1 else self.player2_hand, suit):
            self.marriage_history.append((len(self.trick_history), player, suit))
            return True
        return False
        
    def complete_trick(self, card1, card2, winner):
        """Complete a trick and update tracking."""
        self.trick_history.append((card1, card2, winner))
        self.tricks_won[winner] += 1
        
    def get_state_vector(self, game, player, hand, opponent_played=None):
        """Create state vector for the current game state"""
        state = []
        
        # 1. Played cards (20 binary)
        state.extend([1 if any(c.suit == suit and c.rank == rank for c in self.played_cards) else 0 
                     for suit in self.suits for rank in self.ranks])
        
        # 2. Hand cards (20 binary)
        state.extend([1 if any(c.suit == suit and c.rank == rank for c in hand) else 0 
                     for suit in self.suits for rank in self.ranks])
        
        # 3. Action possibilities (20 binary)
        legal_moves = self.get_legal_moves(player, hand, opponent_played)
        state.extend([1 if any(c.suit == suit and c.rank == rank for c in legal_moves) else 0 
                     for suit in self.suits for rank in self.ranks])
        
        # 4. Marriage declared (1 binary)
        state.append(1 if self.marriage_history and 
                    self.marriage_history[-1][0] == len(self.trick_history) else 0)
        
        # 5. Last card value (5 binary)
        if self.talon:
            last_card = self.talon[-1]
            state.extend([1 if last_card.rank == rank else 0 for rank in self.ranks])
        else:
            state.extend([0] * 5)
        
        # 6. Trick points normalized (2 float)
        player_points = self.player1_points if player == 1 else self.player2_points
        opponent_points = self.player2_points if player == 1 else self.player1_points
        state.extend([player_points / 66, opponent_points / 66])
        
        # 7. No tricks (1 binary)
        state.append(1 if self.tricks_won[player] == 0 else 0)
        
        # 8. Game points normalized (2 float)
        state.extend([self.game_points1 / 7, self.game_points2 / 7])
        
        # 9. Stock closed (1 binary)
        state.append(1 if self.deck_closed else 0)
        
        # 10. Stock closed payoff (2 float)
        if self.deck_closed:
            if self.deck_closer == player:
                state.extend([self._calculate_closer_points() / 3, 0])
            else:
                state.extend([0, self._calculate_non_closer_points() / 3])
        else:
            state.extend([0, 0])
        
        # 11. Cards left normalized (1 float)
        state.append(len(self.talon) / 10)
        
        # 12. Led first trick (1 binary)
        state.append(1 if self.first_trick_leader == player else 0)
        
        # 13. Known opponent cards (22 binary)
        opponent_hand = self.player2_hand if player == 1 else self.player1_hand
        known_cards = self._get_known_opponent_cards(game, opponent_hand)  # Pass game as argument
        state.extend(known_cards)
        
        # 14. Last 2 tricks (42 binary)
        last_tricks = self.trick_history[-2:] if len(self.trick_history) >= 2 else []
        trick_vector = []
        for _ in range(2):  # Two tricks
            if last_tricks:
                trick = last_tricks.pop(0)
                # 10 bits per trick (simplified representation)
                for card in [trick[0], trick[1]]:
                    # Use fewer bits per card (e.g., 4 bits suit, 5 bits rank)
                    suit_idx = self.suits.index(card.suit)
                    rank_idx = self.ranks.index(card.rank)
                    trick_vector.extend([suit_idx/3, rank_idx/4])  # Normalized indices
                # 1 bit for winner
                trick_vector.append(1 if trick[2] == player else 0)
            else:
                trick_vector.extend([0] * 21)  # (10 + 1) features per empty trick
        state.extend(trick_vector)  # Now adds 42 features total (21 * 2)
        
        # 15. First card for q2 (20 binary)
        if opponent_played:
            state.extend([1 if (opponent_played.suit == suit and opponent_played.rank == rank) else 0 
                         for suit in self.suits for rank in self.ranks])
        else:
            state.extend([0] * 20)
        
        return np.array(state, dtype=np.float32)

def process_batch(experiences, ai1, ai2):
    """Process a batch of experiences for both AIs"""
    # Split experiences by player
    ai1_experiences = [(s, a, r, ns) for s, a, r, ns, player in experiences if player == 1]
    ai2_experiences = [(s, a, r, ns) for s, a, r, ns, player in experiences if player == 2]
    
    # Train each AI on their respective experiences
    if ai1_experiences:
        ai1._train_on_batch(ai1_experiences)
    if ai2_experiences:
        ai2._train_on_batch(ai2_experiences)

def train_ai(episodes=10000, save_interval=1000):
    # Remove plot_interval parameter and plotting-related code
    
    # 1. GPU Setup
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Using GPU: {gpus[0]}")
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU available, using CPU")
    
    # 2. Initialize models and game
    device = '/GPU:0' if gpus else '/CPU:0'
    with tf.device(device):
        ai1 = SchnapsenAI()
        ai2 = SchnapsenAI()
    
    game = SchnapsenGame()
    
    # Create directory for saving
    save_dir = "training_checkpoints"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # Dictionary to store training history
    training_history = {
        'player1_wins': [],
        'player2_wins': [],
        'episode_rewards': [],
        'deck_closing_success_rate': []
    }
    
    # Create progress bar
    pbar = tqdm(total=episodes, desc="Training Progress")
    
    # Set up thread pool
    num_threads = multiprocessing.cpu_count()
    executor = ThreadPoolExecutor(max_workers=num_threads)
    
    # Buffer for collecting experiences
    experience_buffer = []
    BATCH_SIZE = 32
    
    for episode in range(episodes):
        episode_experiences = []
        
        while game.game_points1 < 7 and game.game_points2 < 7:
            game.reset_sub_round()
            current_leader = 1
            # Initialize sub-round points here
            sub_round_points1 = 0
            sub_round_points2 = 0
            
            while len(game.player1_hand) > 0 and len(game.player2_hand) > 0:
                # First player's turn
                game.current_player = current_leader  # Add this line
                first_ai = ai1 if current_leader == 1 else ai2
                first_hand = game.player1_hand if current_leader == 1 else game.player2_hand
                state1 = first_ai.get_state(game, current_leader, first_hand)
                possible_actions1 = game.get_legal_moves(current_leader, first_hand)
                action1 = first_ai.choose_action(state1, possible_actions1, game)
                
                if action1[0] == 'close_deck':
                    game.close_deck(current_leader)
                    continue
                else:
                    card1 = action1[1]
                    game.play_card(current_leader, card1)
                
                # Second player's turn
                second_player = 2 if current_leader == 1 else 1
                second_ai = ai2 if current_leader == 1 else ai1
                second_hand = game.player2_hand if current_leader == 1 else game.player1_hand
                state2 = second_ai.get_state(game, second_player, second_hand, card1)
                possible_actions2 = game.get_legal_moves(second_player, second_hand, card1)
                action2 = second_ai.choose_action(state2, possible_actions2, game)
                
                if action2[0] == 'close_deck':
                    game.close_deck(second_player)
                    continue
                else:
                    card2 = action2[1]
                    game.play_card(second_player, card2)
                
                # Determine winner and update trick points
                winner = game.calculate_trick_winner(card1, card2)
                trick_points = game.calculate_trick_points(card1, card2)
                
                # Update points based on actual player numbers, not current_leader
                if (winner == 1 and current_leader == 1) or (winner == 2 and current_leader == 2):
                    sub_round_points1 += trick_points
                else:
                    sub_round_points2 += trick_points
                
                # Calculate rewards
                first_reward = first_ai.calculate_reward(game, current_leader, action1[0], 
                    trick_points if winner == 1 else 0)
                second_reward = second_ai.calculate_reward(game, second_player, action2[0], 
                    trick_points if winner == 2 else 0)
                
                # Get new states
                new_state1 = first_ai.get_state(game, current_leader, first_hand)
                new_state2 = second_ai.get_state(game, second_player, second_hand)
                
                # Update Q-values
                first_ai.learn(state1, action1, first_reward, new_state1)
                second_ai.learn(state2, action2, second_reward, new_state2)
                
                # Draw cards if deck is open
                if not game.deck_closed:
                    game.draw_cards()
                
                # Update who leads the next trick based on who won this trick
                current_leader = 1 if ((winner == 1 and current_leader == 1) or 
                                     (winner == 2 and current_leader == 2)) else 2
                
                # Check for game-ending conditions
                if game.deck_closed:
                    closer_points = sub_round_points1 if game.deck_closer == 1 else sub_round_points2
                    if closer_points >= 66:
                        break
                    elif len(game.player1_hand) == 0:
                        if game.deck_closer == 1:
                            game.game_points2 += 2
                        else:
                            game.game_points1 += 2
                        break
                elif sub_round_points1 >= 66 or sub_round_points2 >= 66:
                    break
            
            # Determine sub-round winner and award game points
            if sub_round_points1 >= 66:
                game_points = game.calculate_game_points(sub_round_points1, sub_round_points2)
                game.game_points1 += game_points
            elif sub_round_points2 >= 66:
                game_points = game.calculate_game_points(sub_round_points2, sub_round_points1)
                game.game_points2 += game_points
        
        # Determine game winner
        game_winner = 1 if game.game_points1 >= 7 else 2
        sub_round_winner = 1 if sub_round_points1 >= 66 else 2
        
        # Update training history directly (instead of using stats.update)
        training_history['player1_wins'].append(1 if game_winner == 1 else 0)
        training_history['player2_wins'].append(1 if game_winner == 2 else 0)
        training_history['episode_rewards'].append({
            'player1': first_reward,
            'player2': second_reward
        })
        training_history['deck_closing_success_rate'].append(
            1 if game.deck_closed and (
                (game.deck_closer == 1 and sub_round_points1 >= 66) or
                (game.deck_closer == 2 and sub_round_points2 >= 66)
            ) else 0
        )
        
        # Add episode experiences to buffer
        episode_experiences.append((state1, action1, first_reward, new_state1, 1))
        episode_experiences.append((state2, action2, second_reward, new_state2, 2))
        
        # Process experiences in batches using thread pool
        if len(experience_buffer) >= BATCH_SIZE:
            # Split buffer into batches
            batches = [experience_buffer[i:i + BATCH_SIZE] 
                      for i in range(0, len(experience_buffer), BATCH_SIZE)]
            
            # Process batches in parallel
            futures = [executor.submit(process_batch, batch, ai1, ai2) 
                      for batch in batches]
            
            # Wait for all batches to complete
            for future in futures:
                future.result()
            
            # Clear buffer
            experience_buffer = []
        
        # Update progress bar
        pbar.update(1)
        
        # Save checkpoints and print progress
        if (episode + 1) % save_interval == 0:
            checkpoint_dir = os.path.join(save_dir, f'checkpoint_{episode+1}')
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Save models with .keras extension
            ai1.save_model(os.path.join(checkpoint_dir, 'model_player1.keras'))
            ai2.save_model(os.path.join(checkpoint_dir, 'model_player2.keras'))
            
            # Save training history
            with open(os.path.join(checkpoint_dir, 'training_history.json'), 'w') as f:
                json.dump(training_history, f)
            
            # Calculate stats
            stats_dict = {
                'win_rate_p1': np.mean(training_history['player1_wins'][-save_interval:]),
                'win_rate_p2': np.mean(training_history['player2_wins'][-save_interval:]),
                'avg_reward_p1': np.mean([r['player1'] for r in training_history['episode_rewards'][-save_interval:]]),
                'avg_reward_p2': np.mean([r['player2'] for r in training_history['episode_rewards'][-save_interval:]]),
                'deck_closing_success': np.mean(training_history['deck_closing_success_rate'][-save_interval:])
            }
            
            # Update progress bar description with stats
            pbar.set_description(
                f"P1 Win: {stats_dict['win_rate_p1']:.2f} "
                f"P2 Win: {stats_dict['win_rate_p2']:.2f} "
                f"Deck Close: {stats_dict['deck_closing_success']:.2f}"
            )
            
            # Save stats
            with open(os.path.join(checkpoint_dir, 'stats.json'), 'w') as f:
                json.dump(stats_dict, f)
    
    # Clean up
    executor.shutdown()
    pbar.close()

class SchnapsenVisualizer:
    def __init__(self):
        self.card_symbols = {
            'Hearts': '♥',
            'Diamonds': '♦',
            'Clubs': '♣',
            'Spades': '♠'
        }
        # Add color codes
        self.colors = {
            'Hearts': '\033[91m',    # Light red
            'Diamonds': '\033[91m',  # Light red
            'Clubs': '\033[90m',     # Dark gray
            'Spades': '\033[90m',    # Dark gray
            'reset': '\033[0m',      # Reset color
            'green': '\033[92m',     # Green for points
            'yellow': '\033[93m',    # Yellow for trump
            'blue': '\033[94m',      # Blue for headers
            'purple': '\033[95m',    # Purple for special events
            'cyan': '\033[96m'       # Cyan for general info
        }
        
    def display_game_state(self, game, hide_hands=False):
        """Display current game state in terminal"""
        os.system('cls' if os.name == 'nt' else 'clear')  # Clear screen
        
        # Display game points
        print(f"\n{self.colors['blue']}Game Points:{self.colors['reset']} "
              f"{self.colors['green']}Player 1: {game.game_points1}{self.colors['reset']} | "
              f"{self.colors['green']}Player 2: {game.game_points2}{self.colors['reset']}")
        print(f"{self.colors['blue']}Trick Points:{self.colors['reset']} "
              f"{self.colors['green']}Player 1: {game.player1_points}{self.colors['reset']} | "
              f"{self.colors['green']}Player 2: {game.player2_points}{self.colors['reset']}")
        print(f"{self.colors['blue']}Current tricks won:{self.colors['reset']} "
              f"{self.colors['green']}Player 1: {game.tricks_won[1]}{self.colors['reset']} | "
              f"{self.colors['green']}Player 2: {game.tricks_won[2]}{self.colors['reset']}")
        
        # Display trump suit
        print(f"\n{self.colors['yellow']}Trump Suit: {self.colors[game.trump_suit]}"
              f"{self.card_symbols[game.trump_suit]} {game.trump_suit}{self.colors['reset']}")
        
        # Display deck status
        deck_status = f"{self.colors['purple']}Closed{self.colors['reset']}" if game.deck_closed else f"{self.colors['cyan']}Open{self.colors['reset']}"
        print(f"Deck {deck_status}")
        if game.deck_closed:
            print(f"{self.colors['purple']}Deck closed by Player {game.deck_closer}{self.colors['reset']}")
        
        # Display hands
        if not hide_hands:
            print(f"\n{self.colors['blue']}Player 1 hand:{self.colors['reset']}")
            self._display_cards(game.player1_hand)
            print(f"\n{self.colors['blue']}Player 2 hand:{self.colors['reset']}")
            self._display_cards(game.player2_hand)
        
        # Display talon
        if game.talon:
            print(f"\n{self.colors['cyan']}Top card in talon: {self._card_to_string(game.talon[-1])}{self.colors['reset']}")
            print(f"{self.colors['cyan']}Cards in talon: {len(game.talon)}{self.colors['reset']}")
        
        # Display trick history
        if game.trick_history:
            print(f"\n{self.colors['blue']}Last trick:{self.colors['reset']}")
            last_trick = game.trick_history[-1]
            print(f"Player 1: {self._card_to_string(last_trick[0])}")
            print(f"Player 2: {self._card_to_string(last_trick[1])}")
            print(f"{self.colors['purple']}Winner: Player {last_trick[2]}{self.colors['reset']}")
        
        # Display marriage history
        if game.marriage_history:
            print(f"\n{self.colors['blue']}Marriages declared:{self.colors['reset']}")
            for trick_num, player, suit in game.marriage_history:
                print(f"{self.colors['purple']}Player {player}: {suit} marriage at trick {trick_num}{self.colors['reset']}")
                
        print(f"\n{self.colors['cyan']}" + "="*50 + f"{self.colors['reset']}")
        
    def _card_to_string(self, card):
        """Convert card to string with colored symbol"""
        return f"{self.colors[card.suit]}{card.rank} {self.card_symbols[card.suit]}{self.colors['reset']}"
        
    def _display_cards(self, cards):
        """Display a list of cards"""
        card_strings = [self._card_to_string(card) for card in cards]
        print(" | ".join(card_strings))

def play_visual_game(ai1, ai2, delay_ms=1000):
    """Play a single game with visualization"""
    game = SchnapsenGame()
    visualizer = SchnapsenVisualizer()
    
    while game.game_points1 < 7 and game.game_points2 < 7:
        game.reset_sub_round()
        current_leader = 1
        
        while len(game.player1_hand) > 0 and len(game.player2_hand) > 0:
            visualizer.display_game_state(game)
            time.sleep(delay_ms/1000)
            
            # First player's turn
            game.current_player = current_leader
            first_ai = ai1 if current_leader == 1 else ai2
            first_hand = game.player1_hand if current_leader == 1 else game.player2_hand
            state1 = first_ai.get_state(game, current_leader, first_hand)
            possible_actions1 = game.get_legal_moves(current_leader, first_hand)
            action1 = first_ai.choose_action(state1, possible_actions1, game)
            
            if action1[0] == 'close_deck':
                game.close_deck(current_leader)
                print(f"\n{visualizer.colors['purple']}Player {current_leader} closes the deck!{visualizer.colors['reset']}")
                time.sleep(delay_ms/1000)
                continue
            else:
                card1 = action1[1]
                game.play_card(current_leader, card1)
                print(f"\n{visualizer.colors['cyan']}Player {current_leader} plays {card1}{visualizer.colors['reset']}")
                time.sleep(delay_ms/1000)
            
            # Second player's turn
            second_player = 2 if current_leader == 1 else 1
            second_ai = ai2 if current_leader == 1 else ai1
            second_hand = game.player2_hand if current_leader == 1 else game.player1_hand
            state2 = second_ai.get_state(game, second_player, second_hand, card1)
            possible_actions2 = game.get_legal_moves(second_player, second_hand, card1)
            action2 = second_ai.choose_action(state2, possible_actions2, game)
            
            if action2[0] == 'close_deck':
                game.close_deck(second_player)
                print(f"\n{visualizer.colors['purple']}Player {second_player} closes the deck!{visualizer.colors['reset']}")
                time.sleep(delay_ms/1000)
                continue
            else:
                card2 = action2[1]
                game.play_card(second_player, card2)
                print(f"\n{visualizer.colors['cyan']}Player {second_player} plays {card2}{visualizer.colors['reset']}")
                time.sleep(delay_ms/1000)
            
            # Determine winner and update points
            winner = game.calculate_trick_winner(card1, card2)
            trick_points = game.calculate_trick_points(card1, card2)
            print(f"\n{visualizer.colors['purple']}Player {winner} wins the trick! "
                  f"(+{trick_points} points){visualizer.colors['reset']}")
            time.sleep(delay_ms/1000)
            
            if winner == 1:
                game.player1_points += trick_points
            else:
                game.player2_points += trick_points
            
            game.complete_trick(card1, card2, winner)
            
            if not game.deck_closed:
                game.draw_cards()
            
            current_leader = winner
            
            if game.player1_points >= 66 or game.player2_points >= 66:
                break
    
    # Display final result
    visualizer.display_game_state(game)
    winner = 1 if game.game_points1 >= 7 else 2
    print(f"\nGame over! Player {winner} wins!")
    return winner

# Add to imports at top:
import os
import time

# Add to main:
if __name__ == "__main__":
    # Configuration
    EPISODES = 200000
    SAVE_INTERVAL = 1000

    # Train the models
    train_ai(episodes=EPISODES, save_interval=SAVE_INTERVAL)

    '''
    
    # After training, load and visualize
    ai1 = SchnapsenAI()
    ai2 = SchnapsenAI()
    
    # Load the final checkpoint
    ai1.load_model(f"training_checkpoints/checkpoint_{EPISODES}/model_player1.keras")
    ai2.load_model(f"training_checkpoints/checkpoint_{EPISODES}/model_player2.keras")
    
    # Play visual game
    
    play_visual_game(ai1, ai2, delay_ms=5000)
    '''

    # After loading the model
    #print(f"Model input shape: {ai1.model.input_shape}")
    #print(f"State vector shape: {ai1.get_state(game, 1, game.player1_hand).shape}")


