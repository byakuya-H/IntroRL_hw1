#!/bin/env python
import sys, os, io, termios, fcntl, tty
from base64 import standard_b64encode

import numpy as np
import matplotlib.pyplot as plt


def draw_img():
    data = np.full((10, 10, 3), 255, dtype=np.uint8)
    buf = io.BytesIO()
    plt.imsave(buf, data, format="png")
    data = standard_b64encode(buf.getvalue())
    im = b""
    while data:
        chunk, data = data[:4096], data[4096:]
        m = 1 if data else 0
        im = b"\033_G"
        im += ",".join(f"{k}={v}" for k, v in dict(m=m, a="T", f=100).items()).encode(
            "ascii"
        )
        im += b";" + chunk if chunk else b""
        im += b"\033\\"
    sys.stdout.buffer.write(im)
    sys.stdout.flush()


def _getch():
    """
    ref: https://stackoverflow.com/posts/72825322/revisions
    """
    fd = sys.stdin.fileno()
    orig = termios.tcgetattr(fd)

    try:
        tty.setcbreak(fd)  # or tty.setraw(fd) if you prefer raw mode's behavior.
        C, c = "", "*"
        while (
            c
            not in "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ,./ "
        ):
            c = sys.stdin.read(1)
            C += c
        return C
    finally:
        termios.tcsetattr(fd, termios.TCSAFLUSH, orig)


# print("aaa", end="")
# print("\r", end="")
# print("bbb")
# sys.stdin.read(1)
a = _getch()
print("a", repr(a), type(a), a == "\x1b", "b")
