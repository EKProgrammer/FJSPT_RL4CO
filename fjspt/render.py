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
    n_trucks = inst["truck_start_times"].size(0)

    machine_colors = plt.cm.Set2(np.linspace(0, 1, max(n_machines, 1)))
    machine_cmap = ListedColormap(machine_colors)

    truck_colors = plt.cm.Pastel1(np.linspace(0, 1, max(n_trucks, 1)))
    truck_cmap = ListedColormap(truck_colors)

    ma_assign = inst["ma_assignment"].nonzero()
    machine_schedule = defaultdict(list)
    for val in ma_assign:
        machine = val[0].item()
        op = val[1].item()
        machine_start = inst["machine_start_times"][val[1]]
        machine_end = inst["machine_finish_times"][val[1]]
        machine_schedule[machine].append((op, machine_start, machine_end))

    # n_ops_max = числу производственных операций
    # n_ops_max = числу транспортировочных операций
    truck_start_times = td["truck_start_times"]  # (bs, num_trucks, n_ops_max)
    truck_finish_times = td["truck_finish_times"]  # (bs, num_trucks, n_ops_max)
    job_tr_ops = td["job_tr_ops"]  # (bs, n_ops_max) номер job
    truck_tr_ops = td["truck_tr_ops"]  # (bs, n_ops_max) номер truck
    trucks_schedule = defaultdict(list)
    for op in range(job_tr_ops.size(1)):
        job = job_tr_ops[op]
        truck = truck_tr_ops[op]
        truck_start = truck_start_times[op]
        truck_end = truck_finish_times[op]
        trucks_schedule[truck].append((job, truck_start, truck_end))

    _, ax = plt.subplots()

    # Plot horizontal bars for each processing task
    for ma, ops in machine_schedule.items():
        for op, start, end in ops:
            job = inst["job_ops_adj"][:, op].nonzero().item()
            ax.barh(
                ma,
                end - start,
                left=start,
                height=0.6,
                color=machine_cmap(ma % machine_cmap.N),
                edgecolor="black",
                linewidth=1,
            )
            ax.text(start + (end - start) / 2, job, op, ha="center", va="center", color="white")

    # Plot horizontal bars for each transportation task
    for tr, ops in trucks_schedule.items():
        for job, start, end in ops:
            ax.barh(
                tr,
                end - start,
                left=start,
                height=0.6,
                color=truck_cmap(tr % truck_cmap.N),
                edgecolor="black",
                hatch="//",
                linewidth=1,
            )
            ax.text(start + (end - start) / 2, job, tr, ha="center", va="center", color="white")

    # Set labels and title
    ax.set_yticks(range(n_jobs))
    ax.set_yticklabels([f"Job {i}" for i in range(n_jobs)])
    ax.set_xlabel("Time")
    ax.set_title("Gantt Chart")

    # 1. Machine legend
    if n_machines > 0:
        machine_handles = [
            plt.Rectangle((0, 0), 1, 1, color=machine_cmap(i % machine_cmap.N), edgecolor="black")
            for i in range(n_machines)
        ]
        ax.legend(
            machine_handles,
            [f"Machine {i}" for i in range(n_machines)],
            loc="center left",
            bbox_to_anchor=(1, 0.7),
            title="Machines",
            fontsize=8,
            title_fontsize=9
        )

    # 2. Truck legend
    if n_trucks > 0:
        truck_handles = [
            plt.Rectangle((0, 0), 1, 1, color=truck_cmap(i % truck_cmap.N),
                          edgecolor="black", alpha=0.7, hatch="//")
            for i in range(n_trucks)
        ]
        ax.legend(
            truck_handles,
            [f"Truck {i}" for i in range(n_trucks)],
            loc="center left",
            bbox_to_anchor=(1, 0.3),
            title="Trucks",
            fontsize=8,
            title_fontsize=9
        )

    # 3. Operation type legend
    machine_patch = plt.Rectangle((0, 0), 1, 1, color='gray', alpha=0.9, edgecolor="black")
    transport_patch = plt.Rectangle((0, 0), 1, 1, color='gray', alpha=0.7, edgecolor="black", hatch="//")
    ax.legend(
        [machine_patch, transport_patch],
        ["Processing", "Transport"],
        loc="upper right",
        fontsize=9
    )

    plt.tight_layout()
    return ax
