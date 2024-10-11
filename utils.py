from typing import Any, List, Tuple
from collections import OrderedDict
import sys, os, io, termios, tty, time, logging
from base64 import standard_b64encode
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image


def plot(record: dict):
    """
    plot the performance of the agent accroding to `record`
    |args|:
        record: dict[str, ], records of the performance data, with key:
            steps
            query
            min
            max
            mean
    """
    plt.figure()
    fig, ax = plt.subplots()
    steps, mean, _max, _min, query = (
        record["steps"],
        record["mean"],
        record["max"],
        record["min"],
        record["query"],
    )
    ax.plot(steps, mean, color="blue", label="reward")
    ax.fill_between(steps, _min, _max, color="blue", alpha=0.2)
    ax.set_xlabel("number of steps")
    ax.set_ylabel("Average score per episode")
    ax1 = ax.twinx()
    ax1.plot(steps, query, color="red", label="query")
    ax1.set_ylabel("queries")
    reward_patch = mpatches.Patch(lw=1, linestyle="-", color="blue", label="score")
    query_patch = mpatches.Patch(lw=1, linestyle="-", color="red", label="query")
    patch_set = [reward_patch, query_patch]
    ax.legend(handles=patch_set)
    fig.savefig("performance.png")
    fig.show()


class table(OrderedDict):
    def __getattr__(self, attr) -> Any:
        return self.get(attr, None)

    def __setattr__(self, attr: str, value) -> None:
        self[attr] = value


class _Drawer_kitty:
    clear_screen = b"\x1b_Ga=d\x1b\\"

    def __init__(self, plt):
        self.plt = plt

    def draw(self, obs: np.ndarray):
        self._w(self.clear_screen)
        buf = io.BytesIO()
        self.plt.imsave(buf, obs, format="png")
        data = standard_b64encode(buf.getvalue())
        im = b""
        while data:
            chunk, data = data[:4096], data[4096:]
            m = 1 if data else 0
            im = b"\x1b_G"
            im += (f"m={m}" + ",a=T,f=100,c=50,C=1,X=50,Y=50").encode("ascii")
            im += b";" + chunk if chunk else b""
            im += b"\x1b\\"
        self._w(im)

    def _w(self, im):
        sys.stdout.buffer.write(im)
        sys.stdout.flush()

    def __del__(self):
        self._w(self.clear_screen)


class _Drawer_plt:
    def __init__(self, plt):
        self.plt = plt
        self.plt.ion()
        self.plt.show(block=False)

    def draw(self, obs: np.ndarray):
        with self.plt.ion():
            self.plt.imshow(obs)
            self.plt.draw()
            self.plt.pause(0.3)

    def __del__(self):
        self.plt.clf()
        self.plt.cla()
        self.plt.ioff()


def _getkey():
    """
    ref: https://stackoverflow.com/posts/72825322/revisions
    """
    fd = sys.stdin.fileno()
    orig = termios.tcgetattr(fd)
    C = ""
    try:
        tty.setcbreak(fd)  # or tty.setraw(fd) if you prefer raw mode's behavior.
        c = "*"
        while (
            c
            not in "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ,./ "
        ):
            c = sys.stdin.read(1)
            C += c
    finally:
        termios.tcsetattr(fd, termios.TCSAFLUSH, orig)
    return C


def prompt_getkey(prompt: str = ""):
    p = partial(print, end="", flush=True)
    p(prompt)
    c = _getkey()
    p("\r\x1b[K")
    return c


def _save(data: np.ndarray, file_name: str):
    im = Image.fromarray(data.astype(np.uint8))
    im.save(file_name)


def save_result(data_set: List[np.ndarray], label_set: List[int], conf):
    os.makedirs(conf.save_dir, exist_ok=True)
    dirs = os.listdir(conf.save_dir)
    start = len(dirs) - 1 if len(dirs) != 0 else len(dirs)
    with open(os.path.join(conf.save_dir, "label.csv"), "+a") as f:
        for i in range(len(data_set)):
            name = str(start + i)
            _save(data_set[i], os.path.join(conf.save_dir, name + ".png"))
            f.write(f"{name},{label_set[i]}\n")


def load_data(dir_path: str) -> Tuple[List[np.ndarray], List[int]]:
    _f = lambda f: os.path.join(dir_path, f)
    if not os.path.isdir(dir_path):
        logging.error(f"dir path: {dir_path} not exitst")
        return [], []
    elif not os.path.exists(_f("label.csv")):
        logging.error(f"file {_f('label.csv')} not exitst")
        return [], []
    data, labels = [], []
    with open(_f("label.csv"), "r") as f:
        for line in f.readlines():
            im_name, label = line.strip().split(",")[:2]
            im_name = _f(im_name) + ".png"
            if not os.path.exists(im_name):
                continue
            im = np.array(Image.open(im_name)).astype(np.uint8)
            data.append(im), labels.append(int(label))
    return data, labels
