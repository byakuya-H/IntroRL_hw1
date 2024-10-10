#!/bin/env python
import time
import logging

import matplotlib.pyplot as plt

from arguments import get_args
from utils import plot, table, prompt_getkey, save_result
from Dagger import Env, Agent, DummyAgent, SvmAgent, SgdAgent, RfAgent, DtAgent
from train import _c, play_game, train_once, train, val

args = get_args()
conf = table(
    env_name="MontezumaRevengeNoFrameskip-v0",
    num_fpstep=8,
    # T=400,
    # T=100,
    T=0xFFFFFFFF,
    num_steps=100000,
    # epochs=250,
    epochs=40,
    val_T=2000,
    val_draw=True,
    epsilon=0.05,
    log_interval=5,
    save_img=True,
    save_dir="imgs",
    save_interval=10,
    play_game=False,
    play_save_interval=100,
    player_dir="played_data",
    draw_method="kitty",
    # draw_method="plt",
    log_file="log",
    # agent=SvmAgent,
    # agent=SgdAgent,
    # agent=RfAgent,
    agent=DtAgent,
    agent_init_dir="",
    # agent_init_dir="played_data",
)

conf = table({**conf, **args})

logging.basicConfig(filename=conf.log_file, filemode="a", level=logging.INFO)
logging.info(
    f"time: {time.strftime('%D[%H:%M:%S]', time.gmtime(time.time()))}, config: {dict(conf)}"
)


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


if __name__ == "__main__":
    main(conf)
