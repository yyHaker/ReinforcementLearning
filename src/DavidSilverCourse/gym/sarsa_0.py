# -*- coding: utf-8 -*-
"""
realize the Sarsa(0) algorithm.
"""
from random import random
from gym import Env
import gym
from gridworld import *  # import the grid environment

class Agent():
    def __init__(self, env: Env):
        self.env = env  # 个体持有环境的引用
        self.Q = {}   # maintain state-action value function table
        self.state = None   # 个体当前的观测
        # initiate the start state
        self._initAgent()

    def performPolicy(self, s, episode_num, use_epsilon):  # 执行一个策略
        """
        perform the policy according to epsilon-greedy policy
        :param s: state
        :param episode_num: episode num
        :param use_epsilon: if use epsilon
        :return:
        """
        epsilon = 1.00 / (episode_num + 1)  # decay the epsilon
        # print("Q value:", self.Q)
        Q_s = self.Q[s]
        str_act = "unknown"
        random_value = random()
        if use_epsilon and random_value < epsilon:
            action = self.env.action_space.sample()
        else:
            str_act = max(Q_s, key=Q_s.get)
            action = int(str_act)
        return action

    def act(self, a):  # 执行一个行为
        return self.env.step(a)

    def learning(self, gamma, alpha, max_episode_num):
        """学习过程(核心)
        :param gamma:
        :param alpha:
        :param max_episode_num:
        :return:
        """
        total_time, time_in_episode, num_episode = 0, 0, 0
        while num_episode < max_episode_num:
            # 环境初始化
            self.state = self.env.reset()
            # 获取个体对于观测的命名
            s0 = self._get_state_name(self.state)
            # show UI
            self.env.render()
            a0 = self.performPolicy(s0, num_episode, use_epsilon=True)

            time_in_episode = 0
            is_done = False  # if stop
            while not is_done:   # w.r.t a  episode
                # execute action
                s1, r1, is_done, infor = self.act(a0)
                # update UI
                self.env.render()
                # get the state name
                s1 = self._get_state_name(s1)
                # make sure s1 exist
                self._assert_state_in_Q(s1, randomized=True)
                # get A'
                a1 = self.performPolicy(s1, num_episode, use_epsilon=True)
                # update Q
                old_value = self._get_Q(s0, a0)
                td_target = r1 + gamma * self._get_Q(s1, a1)
                new_value = old_value + alpha * (td_target - old_value)
                self._set_Q(s0, a0, new_value)

                # change current state
                s0, a0 = s1, a1
                time_in_episode += 1

                # show the last episode infor
                if num_episode == max_episode_num:
                    print("t: {0: >2} : s:{1}, a: {2: 2}, s1: {3}".format(
                        time_in_episode, s0, a0, s1
                    ))
            # show infor of every episode
            print("Episode {0} takes {1} steps.".format(num_episode, time_in_episode))
            num_episode += 1
        return

    def _get_state_name(self, state):
        """将观测转化为字典的键"""
        return str(state)

    def _is_state_in_Q(self, s):
        """判断s的Q值是否存在"""
        return self.Q.get(s) is not None

    def _init_state_value(self, s_name, randomized=True):
        """初始化某状态的Q值"""
        if not self._is_state_in_Q(s_name):
            self.Q[s_name] = {}
            # w.r.t all actions
            for action in range(self.env.action_space.n):
                default_v = random() / 10 if randomized is True else 0.0
                self.Q[s_name][action] = default_v

    def _assert_state_in_Q(self, s, randomized=True):
        """确保某状态Q值存在, 如果不在就初始化"""
        if not self._is_state_in_Q(s):
            self._init_state_value(s, randomized)

    def _get_Q(self, s, a):
        """get Q(s, a)"""
        self._assert_state_in_Q(s, randomized=True)
        return self.Q[s][a]

    def _set_Q(self, s, a, value):
        """set Q(s, a)"""
        self._assert_state_in_Q(s, randomized=True)
        self.Q[s][a] = value

    def _initAgent(self):
        """initiation"""
        self.state = self.env.reset()
        s_name = self._get_state_name(self.state)
        self._assert_state_in_Q(s_name, randomized=True)

def main():
    env = SimpleGridWorld()
    agent = Agent(env)
    print("Learning.........")
    agent.learning(gamma=0.9, alpha=0.1, max_episode_num=800)


if __name__ == "__main__":
    main()







