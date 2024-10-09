import os, time
from typing import List
import logging

import numpy as np
from PIL import Image

from utils import table, save_result
from Dagger import Env, Agent, DummyAgent, Expert

_c = table()


def play_game(env: Env, conf: table):
    logging.info("start to play the game")
    obs = env.reset()
    player = Expert(DummyAgent(env), env, conf.draw_method)
    react = player.label
    while True:
        action = react(obs)
        obs_next, reward, done, _ = env.step(action)
        obs = obs_next
        obs = env.reset() if done else obs
    del player


def label_data(expert: Expert, data: List[np.ndarray], conf: table):
    logging.info("start to label the data")
    os.system("cls" if os.name == "nt" else "clear")
    labels = []
    for obs in data:
        action = expert.label(obs)
        labels.append((action, expert.tell_meaning4act(action)))
    del expert
    logging.info("labelling finished")
    return labels


def train_once(epoch, env: Env, agent: Agent, conf):
    logging.info(f"start to train on epoch: {epoch}")
    obs = env.reset()
    data = []
    for _ in range(conf.T):
        action = agent.select_action(obs, epsilon=conf.epsilon)
        obs_next, reward, done, _ = env.step(action)
        obs = obs_next
        data.append(obs)
        # if the episode has terminated, we need to reset the environment.
        obs = env.reset() if done else obs
    expert = Expert(agent, env, conf.draw_method)
    labels = label_data(expert, data, conf)
    if conf.save_img:
        save_result(data, labels, conf)
    agent.update(data, labels)
    logging.debug(f"epoch: {epoch} finished")
    return data, labels, len(labels)


def val(epoch, query_cnt, env, agent, conf):
    logging.info(f"start to val the agent on epoch {epoch}")
    start = _c.start
    total_steps = (epoch + 1) * conf.T
    obs = env.reset()
    rewards = []
    total_reward = 0
    # evaluate your model by testing in the environment
    for _ in range(conf.test_T):
        action = agent.select_action(obs)
        # you can render to get visual results
        # envs.render()
        obs_next, reward, done, _ = env.step(action)
        total_reward += reward
        obs = obs_next
        if done:
            rewards.append(total_reward)
            total_reward, obs = 0, env.reset()

    end = time.time()
    rwd_mean, rwd_max, rwd_min = (
        np.mean(rewards) if len(rewards) != 0 else 0,
        np.max(rewards) if len(rewards) != 0 else 0,
        np.min(rewards) if len(rewards) != 0 else 0,
    )
    logging.info(
        f"TIME {time.strftime('%Hh %Mm %Ss', time.gmtime(time.time() - start))} "
        + f"Updates {epoch}, num timesteps {total_steps}, FPS {int(total_steps / (end - start))} \n "
        + f"query {query_cnt}, avrage/min/max reward {rwd_mean:.1f}/{rwd_min:.1f}/{rwd_max:.1f}"
    )

    return (
        total_steps,
        rwd_mean,
        rwd_min,
        rwd_max,
    )


def train(epochs, env, agent, conf):
    query_cnt, record = 0, table(steps=[0], mean=[0], min=[0], max=[0], query=[0])
    data_set, label_set = [], []
    for epoch in range(epochs):
        data, labels, cur_query_cnt = train_once(epoch, env, agent, conf)
        data_set += data
        label_set += labels
        query_cnt += cur_query_cnt
        if (epoch + 1) % conf.log_interval == 0:
            res = val(epoch, query_cnt, env.copy(), agent, conf)
            for k, v in zip(record.keys(), res + (query_cnt,)):
                record[k].append(v)
    return data_set, label_set, record
