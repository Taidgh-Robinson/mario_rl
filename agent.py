import torch
import random 
from variables import EPS_END_STEP_COUNT, EPS_START, EPS_END, device

class Agent(): 

    def __init__(self):
        pass

    def select_action_linearly(self,policy_net, steps_done, state, joypad_space):
        sample = random.random()
        if(steps_done <= EPS_END_STEP_COUNT):
            eps_threshold = EPS_START + ((steps_done / (EPS_END_STEP_COUNT - 1)) * (EPS_END - EPS_START))
        else: 
            eps_threshold = EPS_END

        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max1 result is index of where max element was
                # found, so we pick action with the larger expected reward.
                simon_says = policy_net(state)
                if(len(simon_says.shape) == 1):
                    return simon_says.max(0).indices.view(1, 1)
                if(len(simon_says.shape) == 2):
                    return simon_says.max(1).indices.view(1, 1)
        else:
            choice = random.choice(joypad_space)
            return torch.tensor([[joypad_space.index(choice)]], device=device, dtype=torch.long)
