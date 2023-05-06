import sys
import pygame
import gym
import numpy as np
from pygame import gfxdraw

import wrappers
import pendulum


def fun(x, y):
    return [x, y]


class DiscreteToBoxWrapper(gym.ObservationWrapper):
        def __init__(self, env):
            super().__init__(env)
            assert isinstance(env.observation_space, gym.spaces.Discrete), \
                "Should only be used to wrap Discrete envs."
            self.n = self.observation_space.n
            self.observation_space = gym.spaces.Box(0, 1, (self.n,))
        
        def observation(self, obs):
            print("Reward range: ", self.env.reward_range)
            new_obs = np.zeros(self.n)
            new_obs[obs] = 1
            return new_obs
        
        def reset(self):
            self.env.reset()
            print("SUCCESSFULLY RESET BY ACTIONWRAPPER")
            

class ClipReward(gym.RewardWrapper):
    def __init__(self, env, min_reward, max_reward):
        super().__init__(env)
        self.min_reward = min_reward
        self.max_reward = max_reward
        self._reward_range = (min_reward, max_reward)
    
    def reward(self, reward):
        return np.clip(reward, self.min_reward, self.max_reward)
    
    def reset(self):
        self.env.reset()
        print("SUCCESSFULLY RESET BY REWARDWRAPPER")


if __name__ == '__main__':
    # env_1 = ClipReward(DiscreteToBoxWrapper(gym.make("FrozenLake-v1")), -0.5, 0.5)
    # print("Reward range after being wrapped twice: ", env_1.reward_range)
    # env_1.reset()
    # for t in range(10):
    #     a_t = env_1.action_space.sample()
    #     s_t, r_t, done, info, logprob = env_1.step(a_t)
    #     print(s_t)
        
    # print("unwrapped env reset: ")
    # env_1.unwrapped.reset()
    # print(env_1.unwrapped.observation_space)
    # env = gym.make("FrozenLake-v1")
    # print(env.observation_space)
    
    pendulum_env = wrappers.FrameStack(wrappers.GenerateFrame84(gym.make("Pendulum-v1", g=9.81)), 1)
    # pendulum_env = wrappers.GenerateFrame84(gym.make("Pendulum-v1", g=9.81))
    # pendulum_env = wrappers.GenerateFrame84(pendulum.PendulumEnv())
    # pendulum_env = gym.make("Pendulum-v1", g=9.81)
    obs, _ = pendulum_env.reset()
    print("obs: ", obs)
    # print(np.where(obs != 255))
    
    # pygame.init()
    # #设置主屏窗口 ；设置全屏格式：flags=pygame.FULLSCREEN
    # screen = pygame.display.set_mode((84,84))
    # #设置窗口标题
    # pygame.display.set_caption('Surface对象')

    # screen.fill('white')

    # #创建一个 50*50 的图像,并优化显示
    # face = pygame.Surface((50,50))

    # #填充粉红色
    # face.fill(color='pink')
    
    # offset = 10
    # rod_end = (25, 0)
    # rod_width = 10
    # rod_end = pygame.math.Vector2(rod_end).rotate_rad(np.pi / 2)
    # rod_end = (int(rod_end[0] + offset), int(rod_end[1] + offset))
    # gfxdraw.aacircle(
    #     face, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
    # )
    # gfxdraw.filled_circle(
    #     face, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
    # )
    
    # while True:
    #     # 循环获取事件，监听事件
    #     for event in pygame.event.get():
    #         # 判断用户是否点了关闭按钮
    #         if event.type == pygame.QUIT:
    #             #卸载所有模块
    #             pygame.quit()
    #             #终止程序
    #             sys.exit()

    #     # 将绘制的图像添加到主屏幕上，(100,100)是位置坐标，显示屏的左上角为坐标系的(0,0)原点
    #     screen.blit(face, (100, 100))
    #     pygame.display.flip() #更新屏幕内容
    
    x = fun(1, 2)
    print("x: ", x)