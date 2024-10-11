import os, time
from typing import List
import logging

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from utils import table, save_result, plot
from Dagger import Env, Agent, DummyAgent, Expert

_c = table()


def play_game(env: Env, conf: table):
    logging.info("start to play the game")
    obs = env.reset()
    player = Expert(DummyAgent(env), env, _c.plt, conf.draw_method)
    react, mean = player.label, player.tell_meaning4act
    data, labels = [], []
    # while True:
    for i in range(0xFFFFFFFFFF):
        action = react(obs, "use keyboard to control the agent:")
        data.append(obs)
        labels.append(action)
        obs_next, reward, done, _ = env.step(action)
        obs = obs_next
        obs = env.reset() if done else obs
        if i % conf.play_save_interval == 0 and conf.save_img:
            save_result(data, labels, conf)
            data, labels = [], []
            if i % conf.play_save_interval == 0:
                obs = env.reset()
    del player


def label_data(expert: Expert, data: List[np.ndarray], conf: table):
    logging.info("start to label the data")
    os.system("cls" if os.name == "nt" else "clear")
    labels = []
    for obs in data:
        action = expert.label(obs)
        labels.append(action)
    del expert
    logging.info("labelling finished")
    return labels


def train_once(epoch, env: Env, agent: Agent, conf):
    logging.info(f"start to train on epoch: {epoch}")
    obs = env.reset()
    data, labels = [], []
    expert = Expert(agent, env, _c.plt, conf.draw_method)
    for _ in range(conf.T):
        action = agent.select_action(obs, epsilon=conf.epsilon)
        assert isinstance(action, int)
        obs_next, reward, done, _ = env.step(action)
        if conf.T == 0xFFFFFFFF:
            label = expert.label(obs)
            if label == -2:
                expert_halt_game = True
                break
            elif not label == -1:
                data.append(obs)
                labels.append(label)
        obs = obs_next
        # if the episode has terminated, we need to reset the environment.
        obs = env.reset() if done else obs
    labels = label_data(expert, data, conf) if conf.T != 0xFFFFFFFF else labels
    if conf.save_img:
        save_result(data, labels, conf)
    logging.info("start to update the agent")
    logging.debug(data, labels)
    if len(labels) != 0:
        agent.update(data, labels)
    logging.debug(f"epoch: {epoch} finished")
    return data, labels, len(labels)


def val(epoch, query_cnt, env, agent, conf):
    logging.info(f"start to val the agent on epoch {epoch}")
    start = _c.start
    total_steps, rewards, total_reward = (epoch + 1) * conf.T, [], 0
    obs, displayer = env.reset(), Expert(agent, env, _c.plt, conf.draw_method)
    draw = displayer.draw
    if conf.val_draw:
        if conf.draw_method == "kitty":
            draw(obs), time.sleep(0.0025)
        else:
            draw(obs)
    for _ in range(conf.val_T):
        action = agent.select_action(obs)
        # you can render to get visual results
        # env.render() # broken
        obs_next, reward, done, _ = env.step(action)
        if conf.val_draw:
            if conf.draw_method == "kitty":
                draw(obs_next), time.sleep(0.0025)
            else:
                draw(obs_next)
        total_reward += reward
        obs = obs_next
        rewards.append(total_reward)
        if done:
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


def main(conf):
    _c.start, _c.plt = time.time(), plt
    # agent, environment initial
    env = Env(conf.env_name, conf.num_fpstep)

    # You can play this game yourself for fun
    if conf.play_game:
        play_game(env, conf)
        exit(0)

    # agent = DummyAgent(env=env)
    agent = conf.agent(env=env, init_dir=conf.agent_init_dir)

    # start train your agent
    # epochs = int(conf.num_steps // conf.T)
    data_set, label_set, records = train(conf.epochs, env, agent, conf)
    plot(records)
