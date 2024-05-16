import numpy as np

class Buy_and_hold_Agent:
    def __init__(self, action_size=3):
        self.action_size = action_size
        self.generation = 'Buy and Hold'

    def get_action_probabilities(self, observation, current_position):
        action_probs = np.zeros(self.action_size)
        action_probs[2] = 1.0
        return action_probs

class Sell_and_hold_Agent:
    def __init__(self, action_size=3):
        self.action_size = action_size
        self.generation = 'Sell and Hold'

    def get_action_probabilities(self, observation, current_position):
        action_probs = np.zeros(self.action_size)
        action_probs[0] = 1.0
        return action_probs

