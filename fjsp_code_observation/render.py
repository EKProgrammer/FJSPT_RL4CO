from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import ListedColormap
from tensordict.tensordict import TensorDict

from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def render(td: TensorDict, idx: int):
    inst = td[idx]
    num_jobs = inst["job_ops_adj"].size(0)

    # Define a colormap with a color for each job
    colors = plt.cm.tab10(np.linspace(0, 1, num_jobs))
    cmap = ListedColormap(colors)

    assign = inst["ma_assignment"].nonzero()

    schedule = defaultdict(list)

    # Для каждой машины machine хранится список операций с их start и finish временем.
    for val in assign:
        machine = val[0].item()
        op = val[1].item()
        # get start and end times of operation
        start = inst["start_times"][val[1]]
        end = inst["finish_times"][val[1]]
        # write information to schedule dictionary
        schedule[machine].append((op, start, end))

    _, ax = plt.subplots()

    # Plot horizontal bars for each task
    for ma, ops in schedule.items():
        for op, start, end in ops:
            job = inst["job_ops_adj"][:, op].nonzero().item()
            ax.barh(
                ma,
                end - start,
                left=start,
                height=0.6,
                color=cmap(job),
                edgecolor="black",
                linewidth=1,
            )
            # y=ma
            # вертикальная позиция
            # номер машины
            #
            # width=end - start
            # длина прямоугольника по оси X
            # время выполнения операции
            #
            # left=start
            # откуда начинается прямоугольник по X
            # время начала операции
            #
            # height=0.6
            # толщина полосы
            #
            # color=cmap(job)
            # цвет операции, одинаков для всех операций одной job
            #
            # edgecolor="black"
            # чёрная рамка
            #
            # linewidth=1
            # толщина рамки

            ax.text(start + (end - start) / 2, ma, op, ha="center", va="center", color="white")
            # x = start + (end - start) / 2
            # середина операции по времени; текст центрируется по горизонтали
            # y = ma
            # на той же строке, что и barh; совпадает с машиной
            # op
            # текст, который рисуем; обычно номер операции или job-op
            # ha="center"
            #     точка (x, y) становится горизонтальным центром текста
            # va="center"
            #     точка (x, y) становится вертикальным центром текста

    # Set labels and title
    ax.set_yticks(range(len(schedule)))
    ax.set_yticklabels([f"Machine {i}" for i in range(len(schedule))])
    ax.set_xlabel("Time")
    ax.set_title("Gantt Chart")

    # Add a legend for class labels
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(i)) for i in range(num_jobs)]
    # handle — это графический объект, который legend умеет показывать
    # в данном случае — прямоугольник, имитирующий bar
    # plt.Rectangle((0, 0), 1, 1, ...)
    # (0, 0) — фиктивная позиция (неважно, где он реально будет)
    # 1, 1 — фиктивные размеры
    # color=cmap(i) — цвет job i
    # Эти прямоугольники не рисуются на графике,
    # они существуют только для легенды.

    ax.legend(
        handles,
        [f"Job {label}" for label in range(num_jobs)],
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )

    plt.tight_layout()
    return ax
