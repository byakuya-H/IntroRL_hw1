import argparse
from utils import table


def get_args():
    parser = argparse.ArgumentParser(description="RL")
    _args = [
        ["--env-name", dict(type=str, default="MontezumaRevengeNoFrameskip-v0")],
        ["--num-fpstep", dict(type=int, default=8)],
        ["--T", dict(type=int, default=400)],
        ["--num-steps", dict(type=int, default=100000)],
        ["--epochs", dict(type=int, default=250)],
        ["--test-T", dict(type=int, default=2000)],
        ["--epsilon", dict(type=float, default=0.05)],
        ## other parameter
        ["--log-interval", dict(type=int, default=10, help="log interval, one log per n updates (default: 10)")],
        ["--save-img", dict(type=bool, default=True)],
        ["--save-dir", dict(type=str, default="./img")],
        ["--save-interval", dict(type=int, default=10, help="save interval, one eval per n updates (default: None)")],
        ["--play-game", dict(type=bool, default=False)],
    ]
    for arg in _args:
        parser.add_argument(arg[0], **arg[1])
    args = parser.parse_args()
    a = table()
    for arg in _args:
        attr = arg[0][2:].replace('-', '_')
        ar = getattr(args, attr)
        if ar != arg[1]["default"]:
            setattr(a, attr, ar)
    return a
