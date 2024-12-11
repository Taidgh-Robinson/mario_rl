from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
from framestack import FrameStack

env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

fs = FrameStack(env, 6)
fs.reset()

done = True
for step in range(1000):
    action = fs.env.action_space.sample()
    obs, reward, terminated, truncated, info = fs.step(action)
    done = terminated or truncated

    if done:
       state = env.reset()

env.close()
