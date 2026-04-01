from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import ListedColormap
from tensordict.tensordict import TensorDict

from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def check_overlaps(ops):
    overlaps = []

    for i in range(len(ops)):
        for j in range(i + 1, len(ops)):
            a = ops[i]
            b = ops[j]

            if a["end"] > b["start"] and b["end"] > a["start"]:
                overlaps.append((a, b))

    return overlaps


def render(td: TensorDict, idx: int):
    inst = td[idx]

    n_jobs = inst["job_ops_adj"].size(0)
    n_machines = inst["ma_assignment"].size(0)
    n_trucks = inst["truck_in_process"].size(0)

    machine_colors = plt.cm.Set2(np.linspace(0, 1, max(n_machines, 1)))
    machine_cmap = ListedColormap(machine_colors)

    truck_colors = plt.cm.Pastel1(np.linspace(0, 1, max(n_trucks, 1)))
    truck_cmap = ListedColormap(truck_colors)

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
    job_tr_ops = inst["job_tr_ops"]  # (n_ops_max,) номер job
    truck_tr_ops = inst["truck_tr_ops"]  # (n_ops_max,) номер truck
    trucks_schedule = defaultdict(list)
    for op in range(inst["truck_operation"]):
        job = job_tr_ops[op].item()
        truck = truck_tr_ops[op].item()
        truck_start = truck_start_times[op].item()
        truck_end = truck_finish_times[op].item()
        trucks_schedule[truck].append((op, job, truck_start, truck_end))

    fig, ax = plt.subplots(figsize=(15, 10))

    # Plot horizontal bars for each processing task
    for ma, ops in machine_schedule.items():
        for op, start, end in ops:
            job = inst["job_ops_adj"][:, op].nonzero().item()
            ax.barh(
                job,
                end - start,
                left=start,
                height=0.6,
                facecolor=machine_cmap(ma),
                edgecolor="gray",
                linewidth=1,
            )
            ax.text(start + (end - start) / 2, job, op, ha="center", va="center", color="black")

    # Plot horizontal bars for each transportation task
    for tr, ops in trucks_schedule.items():
        for op, job, start, end in ops:
            ax.barh(
                job,
                end - start,
                left=start,
                height=0.6,
                facecolor=truck_cmap(tr),
                edgecolor="gray",
                hatch="//",
                linewidth=1,
            )
            ax.text(start + (end - start) / 2, job, op, ha="center", va="center", color="black")

    # Set labels and title
    ax.set_yticks(range(n_jobs))
    ax.set_yticklabels([f"Job {i}" for i in range(n_jobs)])
    ax.set_xlabel("Time")
    ax.set_title("Gantt Chart")

    # 1. Machine legend
    if n_machines > 0:
        machine_handles = [
            plt.Rectangle((0, 0), 1, 1, facecolor=machine_cmap(i), edgecolor="gray")
            for i in range(n_machines)
        ]
        fig.legend(
            machine_handles,
            [f"Machine {i}" for i in range(n_machines)],
            loc="center left",
            bbox_to_anchor=(0.9, 0.8),
            title="Machines",
            fontsize=9,
        )

    # 2. Truck legend
    if n_trucks > 0:
        truck_handles = [
            plt.Rectangle((0, 0), 1, 1, facecolor=truck_cmap(i),
                          edgecolor="gray", alpha=1, hatch="//")
            for i in range(n_trucks)
        ]
        fig.legend(
            truck_handles,
            [f"Truck {i}" for i in range(n_trucks)],
            loc="center left",
            bbox_to_anchor=(0.9, 0.5),
            title="Trucks",
            fontsize=9,
        )

    # 3. Operation type legend
    machine_patch = plt.Rectangle((0, 0), 1, 1, edgecolor="gray", facecolor="white")
    transport_patch = plt.Rectangle((0, 0), 1, 1, edgecolor="gray", facecolor="white", hatch="//")
    fig.legend(
        [machine_patch, transport_patch],
        ["Processing", "Transport"],
        loc="center left",
        bbox_to_anchor=(0.9, 0.2),
        title="Rectangle type",
        fontsize=9
    )

    # -----------------------------
    ops_debug = []
    for ma, ops in machine_schedule.items():
        for op, start, end in ops:
            job = inst["job_ops_adj"][:, op].nonzero().item()
            ops_debug.append({
                "type": "machine",
                "machine": ma,
                "job": job,
                "op": op,
                "start": start,
                "end": end,
            })
    for tr, ops in trucks_schedule.items():
        for op, job, start, end in ops:
            ops_debug.append({
                "type": "transport",
                "truck": tr,
                "op": op,
                "job": job,
                "start": start,
                "end": end,
            })

    machine_conflicts = {}
    for ma, ops in machine_schedule.items():
        sorted_ops = sorted(ops, key=lambda x: x[1])
        machine_conflicts[ma] = check_overlaps([
            {"start": s, "end": e, "op": op} for op, s, e in sorted_ops
        ])

    truck_conflicts = {}
    for tr, ops in trucks_schedule.items():
        sorted_ops = sorted(ops, key=lambda x: x[1])
        truck_conflicts[tr] = check_overlaps([
            {"start": s, "end": e} for _, _, s, e in sorted_ops
        ])

    job_ops = defaultdict(list)
    for op in ops_debug:
        job_ops[op["job"]].append(op)
    job_conflicts = {}
    for job, ops in job_ops.items():
        job_conflicts[job] = check_overlaps(ops)

    return ax, machine_schedule, trucks_schedule, ops_debug, machine_conflicts, truck_conflicts, job_conflicts
