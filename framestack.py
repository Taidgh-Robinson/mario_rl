import numpy as np
from collections import deque

class FrameStack:
    def __init__(self, env, queue_length, frame_height=240, frame_width=256):
        self.env = env  
        self.queue_length = queue_length
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.frames = deque(maxlen=queue_length)

    def reset(self):
        """Initialize the frame stack by filling it with np.arrays of 0s, as they do in the paper"""
        state, info = self.env.reset()
        #fill the deque with 0s
        for _ in range(self.queue_length):
            self.frames.append(np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8))
        
        #Then add the first frame of the game
        self.frames.append(state)

        #Return the state for ease of access, could just do framestack.env.state but this makes it easier to read
        return state

    def step(self, action):
        """Update the frame stack when an action is taken."""
        observation, reward, terminated, truncated, _ = self.env.step(action)
        self.frames.append(observation)
        return (observation, reward, terminated, truncated, _)
        
    def get_stack(self):
        """Return the stacked frames."""
        # Convert deque to a numpy array
        return np.stack(self.frames, axis=0)