from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
import torch
from framestack import FrameStack
from model import DQN
from dqn_holder import DQNHolder

env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

dqnH = DQNHolder(env, True, 'Mario', SIMPLE_MOVEMENT)


done = True
""" for step in range(1000):
    action = fs.env.action_space.sample()
    obs, reward, terminated, truncated, info = fs.step(action)
    done = terminated or truncated

    if done:
       state = env.reset()
 """

dqnH.run_episode_to_failure()
print(dqnH.memory)
print(len(dqnH.memory))

env.close()
