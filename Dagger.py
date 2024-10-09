from typing import List, Tuple, Union
import sys, io
from abc import abstractmethod
from copy import deepcopy
from base64 import standard_b64encode

import gym
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.linear_model import SGDClassifier

from utils import table, prompt_getkey, _Drawer_plt, _Drawer_kitty


# the wrap is mainly for speed up the game
# the agent will act every num_fpstep frames instead of one frame
class Env:
    def __init__(self, env_name: str, num_fpstep: int):
        """
        |args|
            env_name: str, the name of the used gym environment,
                default to `MontezumaRevengeNoFrameskip-v0`
            num_fpstep: int, a positive integer, the agent acts every num_fpstep frames,
        """
        self.bare_env = gym.make(env_name)
        self.num_fpstep, self.observation_space, self.action_space = (
            num_fpstep,
            self.bare_env.observation_space,
            self.bare_env.action_space,
        )
        self.dummy_obs, self.dummy_info = (
            self.bare_env.reset(),
            dict(lives=0, episode_frame_number=0, frame_number=0),
        )
        self.act2meaning, self.act_dec = (
            {
                0: "NOOP",
                1: "FIRE",
                2: "UP",
                3: "RIGHT",
                4: "LEFT",
                5: "DOWN",
                11: "RIGHTFIRE",
                12: "LEFTFIRE",
            },
            [0, 1, 2, 3, 4, 5, 11, 12],
        )

    def __getattr__(self, name: str):
        return getattr(self.bare_env, name)

    def step(self, action: int):
        """
        obs: numpy.ndarray
        self.bare_env.step():
            |return|
                observation (np.ndarray): agent's observation of the current environment
                reward (float) : amount of reward returned after previous action
                done (bool): whether the episode has ended, in which case further step() calls will return undefined results
                info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        reward_sum, obs_next, reward, done, info = (
            0,
            self.dummy_obs,
            0.0,
            True,
            self.dummy_info,
        )
        for _ in range(self.num_fpstep):
            obs_next, reward, done, info = self.bare_env.step(action)
            reward_sum += reward
            if done:
                obs_next = self.bare_env.reset()
                break
        return obs_next, reward_sum, done, info

    def reset(self):
        return self.bare_env.reset()

    def copy(self):
        return deepcopy(self.bare_env)


class Agent:
    def __init__(self, env: Env):
        self.env = env
        # observation_shape is the shape of the observation
        # here is (210,160,3)=(height, weight, channels)
        self.num_action, self.observation_shape = (
            env.action_space.n,
            env.observation_space.shape,
        )
        num_cls = len(env.act_dec)
        self.initialize_sample, self.num_cls = (
            ([env.dummy_obs.flatten() for _ in range(num_cls)], list(range(num_cls))),
            num_cls,
        )

    @abstractmethod
    def _select_action(self, obs: Union[np.ndarray, list]) -> Union[int, List[int]]:
        return 0

    @abstractmethod
    def update(self, data: List[np.ndarray], labels: List[Tuple[Union[int, str]]]):
        pass

    def select_action(
        self, obs, epsilon: Union[float, None] = None
    ) -> Union[int, List[int]]:
        if epsilon is None:
            # choose an action according to our model
            return self._select_action(obs)
        if np.random.rand() < epsilon:
            # we choose a random action
            return self.env.action_space.sample()
        else:
            # we choose a special action according to our model
            return self._select_action(obs)


# here is an example of creating your own Agent
class DummyAgent(Agent):
    def __init__(self, env: Env):
        super(DummyAgent, self).__init__(env)

    # train your model with labeled data
    def update(self, data, labels):
        pass

    # select actions by your model
    def _select_action(self, obs):
        return 0


class SvmAgent(Agent):
    def __init__(self, env: Env):
        super(SvmAgent, self).__init__(env)
        self.model, self.X, self.Y = svm.SVC(), [], []
        # initialize model
        self.model.fit(*self.initialize_sample)

    def _select_action(self, obs: Union[np.ndarray, list]) -> Union[int, List[int]]:
        if isinstance(obs, np.ndarray):
            return self.model.predict([obs.flatten()])[0]
        return self.model.predict([ob.flatten() for ob in obs])

    def update(self, data: List[np.ndarray], labels: List[Tuple[Union[int, str]]]):
        data, labels = [d.flatten() for d in data], [la[0] for la in labels]
        self.X += data
        self.Y += labels
        print(self.X, self.Y)
        self.model.fit(self.X, self.Y)


class SgdAgent(Agent):
    def __init__(self, env: Env):
        super(SgdAgent, self).__init__(env)
        self.model, self.X, self.Y = SGDClassifier(loss="hinge", penalty="l2"), [], []
        # initialize model
        self.model.fit(*self.initialize_sample)

    def _select_action(self, obs: Union[np.ndarray, list]) -> Union[int, List[int]]:
        if isinstance(obs, np.ndarray):
            return self.model.predict([obs.flatten()])[0]
        return self.model.predict([ob.flatten() for ob in obs])

    def update(self, data: List[np.ndarray], labels: List[Tuple[Union[int, str]]]):
        data, labels = [d.flatten() for d in data], [la[0] for la in labels]
        self.model.partial_fit(data, labels)


class Expert:
    def __init__(self, agent: Agent, env: Env, draw_method: str = "plt"):
        if draw_method == "kitty":
            self._drawer = _Drawer_kitty()
        elif draw_method == "plt":
            self._drawer = _Drawer_plt()
        self.agent, self.env = agent, env
        self.key2act = {
            "/": 0,
            "\x1b[A": 2,
            "\x1b[B": 5,
            "\x1b[C": 3,
            "\x1b[D": 4,
            ",": 12,
            ".": 11,
            " ": 1,
        }

    def draw(self, obs: np.ndarray):
        self._drawer.draw(obs)

    def label(self, obs: np.ndarray):
        self.draw(obs)
        label = prompt_getkey("please label the data:")
        while label not in self.key2act.keys():
            label = prompt_getkey("please re-label the data:")
        return self.key2act[label]

    def tell_meaning4act(self, action: int):
        return self.env.act2meaning.get(action, "I don't know")

    def __getattr__(self, name: str):
        return getattr(self._drawer, name)
