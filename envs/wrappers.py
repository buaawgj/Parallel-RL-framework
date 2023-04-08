# This code is mainly excerpted from openai baseline code.
# https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
import cv2
import pygame
import numpy as np
from collections import deque
import gym
from gym import spaces
from abc import ABC, abstractmethod
from pygame import gfxdraw
from multiprocessing import Process, Pipe

from .monitor import Monitor
from .pendulum import PendulumEnv


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


class ClippedRewardsWrapper(gym.RewardWrapper):
    def reward(self, reward):
        """Change all the positive rewards to 1, negative to -1 and keep zero."""
        return np.sign(reward)


class DiscreteActions(gym.ActionWrapper):
    def __init__(self, env, disc_actions):
        super().__init__(env)
        self.disc_actions = disc_actions
        self._action_space = spaces.Discrete(len(disc_actions))
    
    def action(self, act):
        return self.disc_actions[act]


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env=None):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        super(EpisodicLifeEnv, self).__init__(env)
        self.lives = 0
        self.was_real_done = True
        self.was_real_reset = False

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert somtimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset()
            self.was_real_reset = True
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
            self.was_real_reset = False
        self.lives = self.env.unwrapped.ale.lives()
        return obs
    
    
class EpisodicPendulumEnv(gym.Wrapper):
    def __init__(self, env=None, episode_length=200):
        """
        Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        self.env = env
        self.episode_length = episode_length
        self.lives = 0
        self.was_real_done = True
        self.was_real_reset = False

    def step(self, action):
        obs, reward, done, info, _ = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        if self.episode_length < self.lives and self.episode_length > 0:
            # for Qbert somtimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives += 1
        return obs, reward, done, info, _

    def reset(self):
        """
        Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs, _ = self.env.reset()
            self.was_real_reset = True
            self.lives = 0
        else:
            # no-op step to advance from terminal/lost life state
            obs, _ = self.env.reset()
            self.was_real_reset = True
            self.lives = 0
        return obs, {}


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env=None, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        super(NoopResetEnv, self).__init__(env)
        self.noop_max = noop_max
        self.override_num_noops = None
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset()
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = np.random.randint(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(0)
            if done:
                obs = self.env.reset()
        return obs
   
    
class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break

        max_frame = np.max(np.stack(self._obs_buffer), axis=0)

        return max_frame, total_reward, done, info

    def reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs, {}


class FireResetEnv(gym.Wrapper):
    def __init__(self, env=None):
        """For environments where the user need to press FIRE for the game to start."""
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs
    
    
class GenerateFrame42(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(GenerateFrame42, self).__init__(env)
        self.screen_dim = 42
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_dim, self.screen_dim, 1))
        
        pygame.init()
        
    def observation(self, obs):
        im = self.get_image_data()
        return obs 
    
    def get_image_data(self):
        self.screen = pygame.Surface((self.screen_dim, self.screen_dim))
        
        self.surf = pygame.Surface((self.screen_dim, self.screen_dim))
        self.surf.fill((255, 255, 255))
        
        bound = 2.2
        scale = self.screen_dim / (bound * 2)
        offset = self.screen_dim // 2

        rod_length = 1 * scale
        rod_width = 0.2 * scale
        l, r, t, b = 0, rod_length, rod_width / 2, -rod_width / 2
        coords = [(l, b), (l, t), (r, t), (r, b)]
        transformed_coords = []
        for c in coords:
            c = pygame.math.Vector2(c).rotate_rad(self.unwrapped.state[0] + np.pi / 2)
            c = (c[0] + offset, c[1] + offset)
            transformed_coords.append(c)
        gfxdraw.aapolygon(self.surf, transformed_coords, (204, 77, 77))
        gfxdraw.filled_polygon(self.surf, transformed_coords, (204, 77, 77))

        gfxdraw.aacircle(self.surf, offset, offset, int(rod_width / 2), (204, 77, 77))
        gfxdraw.filled_circle(
            self.surf, offset, offset, int(rod_width / 2), (204, 77, 77)
        )

        rod_end = (rod_length, 0)
        rod_end = pygame.math.Vector2(rod_end).rotate_rad(self.unwrapped.state[0] + np.pi / 2)
        rod_end = (int(rod_end[0] + offset), int(rod_end[1] + offset))
        gfxdraw.aacircle(
            self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
        )
        gfxdraw.filled_circle(
            self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
        )

        # drawing axle
        gfxdraw.aacircle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))
        gfxdraw.filled_circle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
        )
    
    def _get_obs(self):
        return [self.get_image_data(), self.unwrapped._get_obs()]
    
    def reset(self):
        self.env.reset()
        return self._get_obs(), {}
    
    def step(self, u):
        obs, costs, done, info, _ = self.env.step(u)
        return self._get_obs(), costs, done, info, _


class ProcessFrame42(gym.ObservationWrapper):
    def __init__(self, env=None, screen_dim=42):
        super(ProcessFrame42, self).__init__(env)
        self.screen_dim = screen_dim
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_dim, self.screen_dim, 1))

    def observation(self, obs):
        return self.process(obs)

    def process(self, ob):
        # if frame.size == 84 * 160 * 3:
        #     img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        # elif frame.size == 250 * 160 * 3:
        #     img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        # else:
        #     assert False, "Unknown resolution."
        img = ob[0].astype(np.float32)
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        img = np.reshape(img, [self.screen_dim, self.screen_dim, 1])
        ob[0] = img.astype(np.uint8)
        return ob


class ImageToPyTorch(gym.ObservationWrapper):
    """
    Change image shape to CWH
    """
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1])
        )

    def observation(self, observation):
        return [np.swapaxes(observation[0], 2, 0), observation[1]]


class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not belive how complex the previous solution was."""
        self._frames = frames

    def __array__(self, dtype=None):
        out = np.concatenate(self._frames, axis=0)
        if dtype is not None:
            out = out.astype(dtype)
        return out
    
    
class LazyArraies(object):
    def __init__(self, states):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not belive how complex the previous solution was."""
        self._states = states

    def __array__(self, dtype=None):
        out = np.concatenate(self._states, axis=0)
        if dtype is not None:
            out = out.astype(dtype)
        return out


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        self.states = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0]*k, shp[1], shp[2]))

    def reset(self):
        ob, _ = self.env.reset()
        
        for _ in range(self.k):
            self.frames.append(ob[0])
            self.states.append(ob[1])
        return self._get_obs(), {}

    def step(self, action):
        ob, reward, done, info, _ = self.env.step(action)
        
        self.frames.append(ob[0])
        self.states.append(ob[1])
        return self._get_obs(), reward, done, info, _

    def _get_obs(self):
        assert len(self.frames) == self.k and len(self.states) == self.k
        return [LazyFrames(list(self.frames)).__array__(), LazyArraies(self.states).__array__()]


def wrap(env):
    """Apply a common set of wrappers for Atari games."""
    assert 'NoFrameskip' in env.spec.id
    env = EpisodicLifeEnv(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = FrameStack(env, 4)
    return env

def wrap_cover(env_name):
    def wrap_():
        """Apply a common set of wrappers for Atari games."""
        print("NAME: ", env_name)
        env = gym.make(env_name)
        env = Monitor(env, './')
        assert 'NoFrameskip' in env.spec.id
        env = EpisodicLifeEnv(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        if 'FIRE' in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ProcessFrame84(env)
        env = ImageToPyTorch(env)
        env = FrameStack(env, 4)
        env = ClippedRewardsWrapper(env)
        return env
    return wrap_

def wrap_cover_pendulun(env_name, disc_actions, render_mode=None, episode_length=200):
    def wrap_():
        """Apply a common set of wrappers for Atari games."""
        print("NAME: ", env_name)
        env = PendulumEnv(g=9.81, render_mode=render_mode)
        env = GenerateFrame42(env)
        env = ProcessFrame42(env)
        env = ImageToPyTorch(env)
        env = EpisodicPendulumEnv(env, episode_length=episode_length)
        env = FrameStack(env, 4)
        env = DiscreteActions(env, disc_actions)
        env = Monitor(env, './')
        return env
    return wrap_

def wrap_cover_pendulun_test(env_name, disc_actions, render_mode=None, episode_length=200):
    def wrap_():
        """Apply a common set of wrappers for Atari games."""
        print("NAME: ", env_name)
        env = PendulumEnv(g=9.81, render_mode=render_mode)
        env = GenerateFrame42(env)
        env = ProcessFrame42(env)
        env = ImageToPyTorch(env)
        env = EpisodicPendulumEnv(env, episode_length=episode_length)
        env = FrameStack(env, 4)
        env = DiscreteActions(env, disc_actions)
        return env
    return wrap_

def wrap_cover_pendulun_conti(env_name, render_mode=None, episode_length=200):
    def wrap_():
        """Apply a common set of wrappers for Atari games."""
        print("NAME: ", env_name)
        env = PendulumEnv(g=9.81, render_mode=render_mode)
        env = GenerateFrame42(env)
        env = ProcessFrame42(env)
        env = ImageToPyTorch(env)
        env = EpisodicPendulumEnv(env, episode_length=episode_length)
        env = FrameStack(env, 4)
        env = Monitor(env, './')
        return env
    return wrap_