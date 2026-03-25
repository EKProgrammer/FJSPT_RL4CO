from constants import INIT_FINISH
from typing import Literal

import torch
from tensordict import TensorDict

from torch import nn

from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class GreedyPolicy(nn.Module):
    def __init__(self, policy_type: Literal["FIFO", "MOPNR", "SPT", "MWKR"] = "SPT"):
        super().__init__()
        policy_inst = {"FIFO": FIFO, "MOPNR": MOPNR, "SPT": SPT, "MWKR": MWKR}
        self.policy = policy_inst[policy_type]()

    def forward(self, td: TensorDict):
        return self.policy(td)


class FIFO(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, td: TensorDict):
        batch_size = td.size(0)
        n_jobs = td["job_done"].size(1)
        n_mas = td["machine_busy_until"].size(1)
        n_trs = td["truck_location"].size(1)

        # td["ops_ma_adj"] - (bs, num_mas, num_ops)
        # permuted_ops_ma_adj.shape = (bs, ops, num_mas)
        permuted_ops_ma_adj = td["ops_ma_adj"].permute(0, 2, 1)
        # для каждой работы - маска валидных машин для следующей производственной операции
        # valid_machines_mask.shape = (bs, num_jobs, num_mas)
        valid_machines_mask = permuted_ops_ma_adj[
            torch.arange(batch_size).unsqueeze(1),
            td["next_op"]  # (bs, num_jobs) - номер следующей операции для каждой job
        ]

        # заменяем маску машин-операций на времена освобождения машин
        # td["machine_busy_until"].shape = (bs, num_mas)
        # machine_times_expanded.shape = (bs, num_jobs, num_mas)
        machine_times_expanded = td["machine_busy_until"].unsqueeze(1).expand(-1, n_jobs, -1)
        valid_machine_times = torch.where(
            valid_machines_mask == 1,
            machine_times_expanded,
            torch.tensor(INIT_FINISH, device=td.device)
        )
        flat = valid_machine_times.view(batch_size, -1)  # (bs, num_jobs * num_mas)
        flat_indices = torch.argmin(flat, dim=-1)  # (bs,)
        selected_job = flat_indices // n_mas
        selected_machine = flat_indices % n_mas

        # td["truck_busy_until"].shape = (bs, num_trucks)
        selected_truck = torch.argmin(td["truck_busy_until"], dim=-1)

        mask = td["action_mask"]
        logits = torch.zeros((batch_size, 1 + n_jobs * n_mas * n_trs))
        action_idx = 1 + selected_job * n_mas * n_trs + selected_machine * n_trs + selected_truck
        logits[torch.arange(batch_size), action_idx] = 1.0
        return logits, mask


class MOPNR(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, td: TensorDict):
        batch_size = td.size(0)
        n_jobs = td["job_done"].size(1)
        n_mas = td["machine_busy_until"].size(1)
        n_trs = td["truck_location"].size(1)

        job_ops_adj = td["job_ops_adj"]  # (bs, n_jobs, n_ops)
        op_scheduled = td["op_scheduled"]  # (bs, n_ops)
        op_scheduled_exp = op_scheduled.unsqueeze(1)  # (bs, 1, n_ops)
        remaining_mask = job_ops_adj * (~op_scheduled_exp)  # (bs, n_jobs, n_ops)
        remaining_ops = remaining_mask.sum(dim=-1)  # (bs, n_jobs)
        selected_job = remaining_ops.argmax(dim=-1)  # (bs,)

        batch_idx = torch.arange(batch_size, device=td.device)
        op = td["next_op"][batch_idx, selected_job]  # (bs,)
        proc_time = td["proc_times"][batch_idx, :, op]  # (bs, n_mas)
        valid_machines = td["ops_ma_adj"][batch_idx, :, op]  # (bs, n_mas)
        proc_time[valid_machines == 0] = float("inf")
        selected_machine = torch.argmin(proc_time, dim=-1)

        # td["truck_busy_until"].shape = (bs, num_trucks)
        selected_truck = torch.argmin(td["truck_busy_until"], dim=-1)

        mask = td["action_mask"]
        logits = torch.zeros((batch_size, 1 + n_jobs * n_mas * n_trs))
        action_idx = 1 + selected_job * n_mas * n_trs + selected_machine * n_trs + selected_truck
        logits[torch.arange(batch_size), action_idx] = 1.0
        return logits, mask


class SPT(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, td: TensorDict):
        batch_size = td.size(0)
        n_jobs = td["job_done"].size(1)
        n_mas = td["machine_busy_until"].size(1)
        n_trs = td["truck_location"].size(1)

        # td["next_op"].shape = (bs, num_jobs) - номер следующей операции для каждой job
        next_op_expanded = td["next_op"].unsqueeze(1).expand(-1, n_mas, -1)
        proc_times = td["proc_times"]  # (bs, n_mas, n_ops)
        proc_times[proc_times == 0] = float("inf")

        # Получаем времена обработки следующей операции
        next_op_times = torch.gather(
            proc_times,
            dim=2,  # собираем по измерению операций (n_ops)
            index=next_op_expanded  # (bs, n_mas, n_jobs) - индексы нужных операций
        )  # результат: (bs, n_mas, n_jobs)
        next_op_times = next_op_times.permute(0, 2, 1)  # (bs, n_jobs, n_mas)

        flat = next_op_times.reshape(batch_size, -1)  # (bs, num_jobs * num_mas)
        flat_indices = torch.argmin(flat, dim=-1)  # (bs,)
        selected_job = flat_indices // n_mas
        selected_machine = flat_indices % n_mas

        selected_truck = torch.argmin(td["truck_busy_until"], dim=-1)  # (bs,)

        mask = td["action_mask"]
        action_idx = 1 + selected_job * n_mas * n_trs + selected_machine * n_trs + selected_truck
        logits = torch.zeros((batch_size, 1 + n_jobs * n_mas * n_trs))
        logits[torch.arange(batch_size), action_idx] = 1.0
        return logits, mask


class MWKR(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, td: TensorDict):
        batch_size = td.size(0)
        n_jobs = td["job_done"].size(1)
        n_mas = td["machine_busy_until"].size(1)
        n_trs = td["truck_location"].size(1)

        job_ops_adj = td["job_ops_adj"]
        op_scheduled = td["op_scheduled"]
        proc_times = td["proc_times"]
        min_proc_time, _ = proc_times.min(dim=1)  # (bs, n_ops)
        remaining_mask = job_ops_adj * (~op_scheduled.unsqueeze(1))
        remaining_work = (remaining_mask * min_proc_time.unsqueeze(1)).sum(dim=-1)
        remaining_work[td["job_done"]] = -float("inf")
        selected_job = remaining_work.argmax(dim=-1)

        batch_idx = torch.arange(batch_size, device=td.device)
        op = td["next_op"][batch_idx, selected_job]
        proc_time = td["proc_times"][batch_idx, :, op]
        valid_machines = td["ops_ma_adj"][batch_idx, :, op]
        proc_time[valid_machines == 0] = float("inf")
        selected_machine = torch.argmin(proc_time, dim=-1)

        selected_truck = torch.argmin(td["truck_busy_until"], dim=-1)  # (bs,)

        mask = td["action_mask"]
        action_idx = 1 + selected_job * n_mas * n_trs + selected_machine * n_trs + selected_truck
        logits = torch.zeros((batch_size, 1 + n_jobs * n_mas * n_trs))
        logits[torch.arange(batch_size), action_idx] = 1.0
        return logits, mask
