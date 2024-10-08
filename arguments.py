import argparse


def get_args():
    parser = argparse.ArgumentParser(description="RL")
    a = parser.add_argument
    a("--env-name", type=str, default="MontezumaRevengeNoFrameskip-v0")
    a("--num-stacks", type=int, default=8)
    a("--num-steps", type=int, default=400)
    a("--test-steps", type=int, default=2000)
    a("--num-frames", type=int, default=100000)
    ## other parameter
    a("--log-interval", type=int, default=10, help="log interval, one log per n updates (default: 10)")
    a("--save-img", type=bool, default=True)
    a("--save-interval", type=int, default=10, help="save interval, one eval per n updates (default: None)")
    a("--play-game", type=bool, default=False)
    args = parser.parse_args()
    return args
