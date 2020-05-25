# -*- coding: utf-8 -*-
from collections import deque
import random
import cv2
import torch

from robotron.robotron import Robotron


class Env():
    GAMEBOX = [116, 309, 608, 974]

    def __init__(self, args):
        self.device = args.device
        self.env = Robotron(fps=0)
        self.actions = 64
        self.lives = 0  # Life counter (used in DeepMind training)
        self.life_termination = False  # Used to check if resetting only from loss of life
        self.window = args.history_length  # Number of frames to concatenate
        self.state_buffer = deque([], maxlen=args.history_length)
        self.training = True  # Consistent with model training mode

    def _get_state(self):
        state = self.env.get_state()
        return torch.tensor(state, dtype=torch.float32, device=self.device).div_(255)

    def _reset_buffer(self):
        for _ in range(self.window):
            self.state_buffer.append(torch.zeros(84, 84, device=self.device))

    def reset(self):
        info = None

        if self.life_termination:
            self.life_termination = False  # Reset flag
            _, reward, terminal, info = self.env.step(0)
        else:
            # Reset internals
            self._reset_buffer()
            self.env.reset()
            # Perform up to 30 random no-ops before starting
            for _ in range(random.randrange(30)):
                _, reward, terminal, info = self.env.step(0)  # Assumes raw action 0 is always no-op
                if terminal:
                    self.env.reset()
        # Process and return "initial" state
        observation = self._get_state()
        self.state_buffer.append(observation)
        if info is not None:
            self.lives = info['lives']
        return torch.stack(list(self.state_buffer), 0)

    def step(self, action):
        # Repeat action 4 times, max pool over last 2 frames
        frame_buffer = torch.zeros(2, 84, 84, device=self.device)
        reward, done, info, lives = 0, False, None, 3
        for t in range(4):
            move = action // 8
            shoot = action % 8
            action = ((move + 1) * 9) + shoot + 1
            image, new_reward, terminal, info = self.env.step(action)
            reward += new_reward
            lives = info['lives']
            if t == 2:
                frame_buffer[0] = self._get_state()
            elif t == 3:
                frame_buffer[1] = self._get_state()
            done = terminal or info['level'] > 1
            if done:
                break
        observation = frame_buffer.max(0)[0]
        self.state_buffer.append(observation)
        # Detect loss of life as terminal in training mode
        if self.training:
            if lives < self.lives and lives > 0:  # Lives > 0 for Q*bert
                self.life_termination = not done  # Only set flag when not truly done
                done = True
            self.lives = lives
        # Return state, reward, done
        return torch.stack(list(self.state_buffer), 0), reward, done

    # Uses loss of life as terminal signal
    def train(self):
        self.training = True

    # Uses standard terminal signal
    def eval(self):
        self.training = False

    def action_space(self):
        return self.actions

    def render(self):
        pass

    def close(self):
        cv2.destroyAllWindows()
