from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import ListedColormap
from tensordict.tensordict import TensorDict

from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def render(td: TensorDict, idx: int):
    inst = td[idx]

    n_jobs = inst["job_ops_adj"].size(0)
    n_machines = inst["ma_assignment"].size(0)
    n_trucks = inst["truck_in_process"].size(0)

    colors = plt.cm.Pastel1(np.linspace(0, 1, n_jobs))
    cmap = ListedColormap(colors)

    ma_assign = inst["ma_assignment"].nonzero()
    machine_schedule = defaultdict(list)
    for val in ma_assign:
        machine = val[0].item()
        op = val[1].item()
        machine_start = inst["machine_start_times"][val[1]].item()
        machine_end = inst["machine_finish_times"][val[1]].item()
        machine_schedule[machine].append((op, machine_start, machine_end))

    # n_ops_max = числу производственных операций
    # n_ops_max = числу транспортировочных операций
    truck_start_times = inst["truck_start_times"]  # (n_ops_max,)
    truck_finish_times = inst["truck_finish_times"]  # (n_ops_max,)
    truck_tr_ops = inst["truck_tr_ops"]  # (n_ops_max,) номер truck
    trucks_schedule = defaultdict(list)
    for op in range(inst["truck_operation"]):
        truck = truck_tr_ops[op].item()
        truck_start = truck_start_times[op].item()
        truck_end = truck_finish_times[op].item()
        trucks_schedule[truck].append((op, truck_start, truck_end))

    fig, ax = plt.subplots(figsize=(15, 10))

    # Plot horizontal bars for each processing task
    for ma, ops in machine_schedule.items():
        for op, start, end in ops:
            job = inst["job_ops_adj"][:, op].nonzero().item()
            ax.barh(
                ma,
                end - start,
                left=start,
                height=0.6,
                facecolor=cmap(job),
                edgecolor="gray",
                linewidth=1,
            )
            ax.text(start + (end - start) / 2, ma, op, ha="center", va="center", color="black", fontsize=13)

    # Plot horizontal bars for each transportation task
    for tr, ops in trucks_schedule.items():
        for tr_op, start, end in ops:
            job = inst["job_tr_ops"][tr_op].item()
            op = inst["tr_op_to_op"][tr_op].item()
            ax.barh(
                n_machines + tr,
                end - start,
                left=start,
                height=0.6,
                facecolor=cmap(job),
                edgecolor="gray",
                hatch="//",
                linewidth=1,
            )
            ax.text(start + (end - start) / 2, n_machines + tr, op, ha="center", va="center", color="black", fontsize=13)

    # Set labels and title
    ax.set_yticks(range(n_machines + n_trucks))
    ax.set_yticklabels([f"Machine {i}" for i in range(n_machines)] + [f"Truck {i}" for i in range(n_trucks)], fontsize=13)
    ax.set_xlabel("Time", fontsize=13)
    ax.set_title("Gantt Chart", fontsize=13)

    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(i)) for i in range(n_jobs)]
    fig.legend(
        handles,
        [f"Job {label}" for label in range(n_jobs)],
        loc="center left",
        bbox_to_anchor=(0.9, 0.5),
        fontsize=13
    )

    return ax, machine_schedule, trucks_schedule
