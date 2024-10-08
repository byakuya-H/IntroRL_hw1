import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot(record):
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
    ax.plot(record["steps"], record["mean"], color="blue", label="reward")
    ax.fill_between(
        record["steps"], record["min"], record["max"], color="blue", alpha=0.2
    )
    ax.set_xlabel("number of steps")
    ax.set_ylabel("Average score per episode")
    ax1 = ax.twinx()
    ax1.plot(record["steps"], record["query"], color="red", label="query")
    ax1.set_ylabel("queries")
    reward_patch = mpatches.Patch(lw=1, linestyle="-", color="blue", label="score")
    query_patch = mpatches.Patch(lw=1, linestyle="-", color="red", label="query")
    patch_set = [reward_patch, query_patch]
    ax.legend(handles=patch_set)
    fig.savefig("performance.png")
    fig.show()
