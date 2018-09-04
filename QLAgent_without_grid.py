# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 09:51:43 2018

@author: Bravin
"""

import numpy as np
from ple.games.flappybird import FlappyBird
from ple import PLE
import time

class QLAgent():
    def __init__(self,flappy_actions):
        
        #intializing state variables index size
        horizontal_distance_next = 350
        horizontal_distance_next_next = 700
        vertical_distance_lower = 1024
        
        #initializing the q-matrix with 4d array.4 of them are to hold the state values one to hold the action value
        self.Q_values = np.zeros((horizontal_distance_next,horizontal_distance_next_next,vertical_distance_lower,2))
        self.flappy_actions = flappy_actions
        self._alpha = 0.1
        self._gamma = 1
        
        #function to get the state of the agent and store it in a state array
    def get_current_state(self,flappy_observations):
        game_state = np.zeros((3,), dtype=int)
        game_state[0] = flappy_observations['next_pipe_dist_to_player']
        game_state[1] = flappy_observations['next_next_pipe_dist_to_player']
        #game_state[1] = (flappy_observations['next_pipe_top_y'] - flappy_observations['player_y'] + 512) // self.grid_size
        game_state[2] = (flappy_observations['next_pipe_bottom_y'] - flappy_observations['player_y'])
        #game_state[1] = flappy_observations['next_pipe_bottom_y']
        
        return game_state
        
    #definfing function to perform an action and retun values based on the returned rewards
    def perform_action(self,p,action):
        reward = p.act(self.flappy_actions[action])
        if reward >= 0:
            return 1
        else:
            return -1000
    
    
    def get_action(self,game_state):
        flappy_jump = self.Q_values[game_state[0],game_state[1],game_state[2],0]
        print (flappy_jump)
        flappy_not_jump = self.Q_values[game_state[0],game_state[1],game_state[2],1]
        print (flappy_not_jump)
        
        if flappy_jump > flappy_not_jump:
            return 0
        else:
            return 1
        
    def update_Q_values(self,game_current_state,game_next_state,reward,action):
        self.Q_values[game_current_state[0], game_current_state[1], game_current_state[2], action] = (1-self._alpha) * self.Q_values[game_current_state[0], 
                     game_current_state[1], game_current_state[2], action] + self._alpha * (reward + self._gamma * np.max(self.Q_values[game_next_state[0],
                                       game_next_state[1],game_next_state[2]]))
                         
            
if __name__ == "__main__":
    game = FlappyBird()
    p = PLE(game, fps=30, display_screen=True)
    agent = QLAgent(flappy_actions=p.getActionSet())

    p.init()
    
    game_current_state = agent.get_current_state(game.getGameState())
    number_of_episods = 0
    maximum_score = 0
    
    while True:
        action = agent.get_action(game_current_state)
        current_score = p.score()
        maximum_score = max(current_score, maximum_score)
        
        reward = agent.perform_action(p, action)
        
        game_next_state = agent.get_current_state(game.getGameState())
        
        agent.update_Q_values(game_current_state, game_next_state, reward, action)
        game_current_state = game_next_state
        
        time.sleep(0.01)
        
        if p.game_over():
            number_of_episods += 1
            print('Episode number: %s, Current episode score score: %s, Maximun score: %s' % (number_of_episods,
                                                                                              current_score, 
                                                                                              maximum_score))
            p.reset_game()