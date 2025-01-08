from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
import torch
from framestack import FrameStack
from model import DQN

env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

fs = FrameStack(env, 6)
fs.reset()

done = True
""" for step in range(1000):
    action = fs.env.action_space.sample()
    obs, reward, terminated, truncated, info = fs.step(action)
    done = terminated or truncated

    if done:
       state = env.reset()
 """

for i in range(10):
    fs.step(1)

data = fs.get_stack()
reshaped = data.reshape(18, 240, 256)
tens = torch.from_numpy(reshaped)
d = DQN(7)
output = d(tens/255)

print(output.shape)
env.close()
