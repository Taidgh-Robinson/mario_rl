from framestack import FrameStack
import torch
import os
import pickle
from model import DQN
from memory import ReplayMemory

class DQNHolder():
    def __init__(self, env, should_reset, game_name, joypad_space):
        self.game_name = game_name
        self.policy_net = DQN(len(joypad_space))
        self.target_net = DQN(len(joypad_space))
        self.step_count = 0 
        self.currently_selected_action = None
        self.current_episode_is_terminated = False
        self.memory = ReplayMemory(100)
        self.episode_scores = [] 
        self.framestack = FrameStack(env, 6)
        if should_reset:
            self.framestack.reset()
        
        self.copy_policy_weights_to_target()

    def copy_policy_weights_to_target(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()

        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]
                
        self.target_net.load_state_dict(target_net_state_dict)

    def step(self):
        if (self.step_count % 4 == 0):
            #TODO: SELECT ACTION FROM AGENT 
            action = 0
            self.currently_selected_action = action
        current_state = self.framestack.get_stack()
        observation, reward, terminated, truncated, _ = self.framestack.step(self.currently_selected_action)
        done = terminated or truncated
        
        if terminated:
            next_state = None
            self.current_episode_is_terminated = True
        else:
            next_state = torch.from_numpy(self.framestack.get_stack())

        self.memory.push(current_state, self.currently_selected_action, next_state, reward)

        self.step_count += 1

    def save_training_information(self, is_done):
        path = 'models/' +self.game_name+"/"+ str(self.step_count) + '/'
        os.makedirs(path, exist_ok=True)
        torch.save(self.policy_net.state_dict(), path+'policy_net.pth')
        torch.save(self.target_net.state_dict(), path+'target_net.pth')

        with open(path+'memory.pkl', 'wb') as f:
            pickle.dump(self.memory, f)
        with open(path+'framestack.pkl', 'wb') as f:
            pickle.dump(self.framestack, f)
        with open(path+'count.pkl', 'wb') as f:
            pickle.dump(self.step_count, f)
        with open(path+'episode_durations.pkl', 'wb') as f:
            pickle.dump(self.episode_scores, f)
        with open(path+'is_done.pkl', 'wb') as f:
            pickle.dump(is_done, f)
