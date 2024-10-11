from typing import Any, List, Tuple, Union
import sys, io, logging
from abc import abstractmethod
from copy import deepcopy
from base64 import standard_b64encode

import gym
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from utils import (
    save_result,
    table,
    prompt_getkey,
    _Drawer_plt,
    _Drawer_kitty,
    load_data,
)


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
        self.act2meaning, self.act_dec, self.act_enc = (
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
            [0, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 6, 7],
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
            ([env.dummy_obs for _ in range(num_cls)], [0 for _ in range(num_cls)]),
            num_cls,
        )

    @abstractmethod
    def _select_action(
        self, obs: Union[np.ndarray, list]
    ) -> Union[int, List[int], np.ndarray]:
        return 0

    @abstractmethod
    def update(self, data: List[np.ndarray], labels: List[int]):
        pass

    def _act_dec(self, act) -> Union[List[int], int]:
        if isinstance(act, np.ndarray):
            return [self.env.act_dec[a] for a in act]
        return self.env.act_dec[act]

    def _act_enc(self, act) -> Union[List[int], int]:
        if isinstance(act, list) or isinstance(act, np.ndarray):
            return [self.env.act_enc[a] for a in act]
        return self.env.act_enc[act]

    def select_action(
        self, obs: Union[np.ndarray, list], epsilon: Union[float, None] = None
    ) -> Union[int, List[int]]:
        if epsilon is None or np.random.rand() >= epsilon:
            # choose an action according to our model
            return self._act_dec(self._select_action(obs))
        else:
            # we choose a random action
            return self.env.action_space.sample()


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
    def __init__(self, env: Env, init_dir: str = ""):
        super(SvmAgent, self).__init__(env)
        self.model, self.X, self.Y = svm.SVC(), [], []
        # initialize model
        if init_dir != "":
            X, Y = load_data(init_dir)
            X, Y = [x.flatten() for x in X], self._act_enc(Y)
        else:
            X, Y = (
                [d.flatten() for d in self.initialize_sample[0]],
                self.initialize_sample[1],
            )
        logging.info("start to initialize the model")
        self.model.fit(X, Y)
        logging.info("initialization finished")

    def _select_action(
        self, obs: Union[np.ndarray, list]
    ) -> Union[int, List[int], np.ndarray]:
        if isinstance(obs, np.ndarray):
            return self.model.predict([obs.flatten()])[0]
        return self.model.predict([ob.flatten() for ob in obs])

    def update(self, data: List[np.ndarray], labels: List[int]):
        self.X += data
        self.Y += labels
        save_result(data, labels, table(save_dir="imgs"))
        input("ready to update the agent, press any key to continue.")
        self.X, self.Y = load_data("imgs")
        self.model.fit([x.flatten() for x in self.X], self._act_enc(self.Y))


class SgdAgent(Agent):
    def __init__(self, env: Env, init_dir: str = ""):
        super(SgdAgent, self).__init__(env)
        self.model, self.X, self.Y = (
            SGDClassifier(loss="hinge", penalty="l2", max_iter=999),
            [],
            [],
        )
        # initialize model
        if init_dir != "":
            X, Y = load_data(init_dir)
            X, Y = [x.flatten() for x in X], self._act_enc(Y)
        else:
            X, Y = (
                [d.flatten() for d in self.initialize_sample[0]],
                self.initialize_sample[1],
            )
        logging.info("start to initialize the model")
        self.model.fit(X, Y)
        logging.info("initialization finished")

    def _select_action(
        self, obs: Union[np.ndarray, list]
    ) -> Union[int, List[int], np.ndarray]:
        if isinstance(obs, np.ndarray):
            return self.model.predict([obs.flatten()])[0]
        return self.model.predict([ob.flatten() for ob in obs])

    def update(self, data: List[np.ndarray], labels: List[int]):
        _data, _labels = [d.flatten() for d in data], self._act_enc(labels)
        self.model.partial_fit(_data, _labels)


class DtAgent(Agent):
    def __init__(self, env: Env, init_dir: str = ""):
        super(DtAgent, self).__init__(env)
        self.model, self.X, self.Y = (
            DecisionTreeClassifier(),
            [],
            [],
        )
        # initialize model
        if init_dir != "":
            X, Y = load_data(init_dir)
            X, Y = [x.flatten() for x in X], self._act_enc(Y)
        else:
            X, Y = (
                [d.flatten() for d in self.initialize_sample[0]],
                self.initialize_sample[1],
            )
        logging.info("start to initialize the model")
        self.model.fit(X, Y)
        logging.info("initialization finished")

    def _select_action(
        self, obs: Union[np.ndarray, list]
    ) -> Union[int, List[int], np.ndarray]:
        if isinstance(obs, np.ndarray):
            return self.model.predict([obs.flatten()])[0]
        return self.model.predict([ob.flatten() for ob in obs])

    def update(self, data: List[np.ndarray], labels: List[int]):
        self.X += data
        self.Y += labels
        save_result(data, labels, table(save_dir="imgs"))
        input("ready to update the agent, press any key to continue.")
        self.X, self.Y = load_data("imgs")
        self.model.fit([x.flatten() for x in self.X], self._act_enc(self.Y))


class RfAgent(Agent):
    def __init__(self, env: Env, init_dir: str = ""):
        super(RfAgent, self).__init__(env)
        self.model, self.X, self.Y = (
            RandomForestClassifier(),
            [],
            [],
        )
        # initialize model
        if init_dir != "":
            X, Y = load_data(init_dir)
            X, Y = [x.flatten() for x in X], self._act_enc(Y)
        else:
            X, Y = (
                [d.flatten() for d in self.initialize_sample[0]],
                self.initialize_sample[1],
            )
        logging.info("start to initialize the model")
        self.model.fit(X, Y)
        logging.info("initialization finished")

    def _select_action(
        self, obs: Union[np.ndarray, list]
    ) -> Union[int, List[int], np.ndarray]:
        if isinstance(obs, np.ndarray):
            return self.model.predict([obs.flatten()])[0]
        return self.model.predict([ob.flatten() for ob in obs])

    def update(self, data: List[np.ndarray], labels: List[int]):
        self.X += data
        self.Y += labels
        save_result(data, labels, table(save_dir="imgs"))
        input("ready to update the agent, press any key to continue.")
        self.X, self.Y = load_data("imgs")
        self.model.fit([x.flatten() for x in self.X], self._act_enc(self.Y))


class Expert:
    def __init__(self, agent: Agent, env: Env, plt, draw_method: str = "plt"):
        if draw_method == "kitty":
            self._drawer = _Drawer_kitty(plt)
        elif draw_method == "plt":
            self._drawer = _Drawer_plt(plt)
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
            "m": -1,
            "n": -2,
        }

    def draw(self, obs: np.ndarray):
        self._drawer.draw(obs)

    def label(
        self,
        obs: np.ndarray,
        prompt="Arrow key to control the agent, space to fire, `,` and `.` to fire left and right, `/` to noop, `m` to reject the sample, `n` to end this epoch. Please label the data:",
    ):
        self.draw(obs)
        label = prompt_getkey(prompt)
        while label not in self.key2act.keys():
            label = prompt_getkey("please re-label the data:")
        return self.key2act[label]

    def tell_meaning4act(self, action: int):
        return self.env.act2meaning.get(action, "I don't know")

    def __getattr__(self, name: str):
        return getattr(self._drawer, name)
