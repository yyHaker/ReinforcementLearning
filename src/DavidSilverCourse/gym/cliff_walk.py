# -*- coding: utf-8 -*-
from gridworld import GridWorldEnv
from gym import spaces

env = GridWorldEnv(n_width=12,
                   n_height=4,
                   u_size=60,
                   default_reward=-1,
                   default_type=0,
                   windy=False)
env.action_space = spaces.Discrete(4)  # set action space num
env.start = (0, 0)
env.ends = [(11, 0)]
# set cliff
for i in range(10):
    env.rewards.append((i+1, 0, -100))   # set special reward
    env.ends.append((i+1, 0))   # set cliff all terminal states
# make set take effect
env.refresh_setting()
# 环境初始化
env.reset()
# UI show
env.render()
# input("press any key to continue")
for _ in range(20000):
    env.render()
    a = env.action_space.sample()
    state, reward, isdone, info = env.step(a)
    print("{0}, {1}, {2}, {3}".format(a, reward, isdone, info))

print("env closed")