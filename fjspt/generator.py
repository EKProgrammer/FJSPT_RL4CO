from functools import partial

import numpy as np
import torch

from tensordict.tensordict import TensorDict

from rl4co.envs.common.utils import Generator
from rl4co.utils.pylogger import get_pylogger

from .parser import get_max_ops_from_files, read, file2lines

log = get_pylogger(__name__)


class FJSPTGenerator(Generator):
    """Data generator for the Flexible Job-Shop Scheduling Problem with Transportation resources (FJSPT)."""

    def __init__(
        self,
        num_jobs: int = 10,
        num_machines: int = 5,
        num_trucks: int = 2,
        min_ops_per_job: int = 4,
        max_ops_per_job: int = 6,
        min_processing_time: int = 1,
        max_processing_time: int = 20,
        min_transportation_time: int = 0,
        max_transportation_time: int = 15,
        min_eligible_ma_per_op: int = 1,
        max_eligible_ma_per_op: int = None,
        same_mean_per_op: bool = True,
        **unused_kwargs,
    ):
        self.num_jobs = num_jobs
        self.num_mas = num_machines
        self.num_trucks = num_trucks
        self.min_ops_per_job = min_ops_per_job
        self.max_ops_per_job = max_ops_per_job
        self.min_processing_time = min_processing_time
        self.max_processing_time = max_processing_time
        self.min_transportation_time = min_transportation_time
        self.max_transportation_time = max_transportation_time
        self.min_eligible_ma_per_op = min_eligible_ma_per_op
        self.max_eligible_ma_per_op = max_eligible_ma_per_op or num_machines
        # determines whether to use a fixed number of total operations or let it vary between instances
        # NOTE: due to the way rl4co builds datasets, we need a fixed size here
        self.n_ops_max = max_ops_per_job * num_jobs
        self.same_mean_per_op = same_mean_per_op
        # FFSP environment doen't have any other kwargs
        if len(unused_kwargs) > 0:
            log.error(f"Found {len(unused_kwargs)} unused kwargs: {unused_kwargs}")

    def _simulate_processing_times(self, n_eligible_per_ops: torch.Tensor) -> torch.Tensor:
        bs, n_ops_max = n_eligible_per_ops.shape

        # (bs, max_ops, machines)
        ma_seq_per_ops = torch.arange(1, self.num_mas + 1)[None, None].expand(
            bs, n_ops_max, self.num_mas
        )

        ma_ops_edges_unshuffled = torch.Tensor.float(
            ma_seq_per_ops <= n_eligible_per_ops[..., None]
        )
        # random shuffling
        idx = torch.rand_like(ma_ops_edges_unshuffled).argsort()
        ma_ops_edges = ma_ops_edges_unshuffled.gather(2, idx).transpose(1, 2)

        # (bs, max_ops, machines)
        if self.same_mean_per_op:
            proc_times = torch.ones((bs, self.num_mas, n_ops_max))
            proc_time_means = torch.randint(
                self.min_processing_time, self.max_processing_time, (bs, n_ops_max)
            )
            low_bounds = torch.maximum(
                torch.full_like(proc_times, self.min_processing_time),
                (proc_time_means * (1 - 0.2)).round().unsqueeze(1),
            )
            high_bounds = (
                torch.minimum(
                    torch.full_like(proc_times, self.max_processing_time),
                    (proc_time_means * (1 + 0.2)).round().unsqueeze(1),
                )
                + 1
            )
            proc_times = (
                torch.randint(2**63 - 1, size=proc_times.shape) % (high_bounds - low_bounds)
                + low_bounds
            )
        else:
            proc_times = torch.randint(
                self.min_processing_time,
                self.max_processing_time + 1,
                size=(bs, self.num_mas, n_ops_max),
            )

        # remove proc_times for which there is no corresponding ma-ops connection
        proc_times = proc_times * ma_ops_edges
        return proc_times

    def _simulate_trucks_times(self, batch_size) -> torch.Tensor:
        trucks_times = torch.randint(
            self.min_transportation_time,
            # соблюдаем неравенство треугольника !!!
            min(2 * self.min_transportation_time - 1, self.max_transportation_time),
            size=(batch_size, self.num_mas + 1, self.num_mas + 1),
            # Всего: LU + self.num_mas
        )
        # обнуление диагонали для батча матриц
        mask = ~torch.eye(self.num_mas + 1, dtype=torch.bool, device=trucks_times.device)
        trucks_times = trucks_times * mask
        return trucks_times

    def _generate(self, batch_size) -> TensorDict:
        # simulate how many operations each job has
        n_ope_per_job = torch.randint(
            self.min_ops_per_job,
            self.max_ops_per_job + 1,
            size=(*batch_size, self.num_jobs),
        )

        # determine the total number of operations per batch instance (which may differ)
        n_ops_batch = n_ope_per_job.sum(1)  # (bs)
        # determine the maximum total number of operations over all batch instances
        n_ops_max = self.n_ops_max or n_ops_batch.max()

        # generate a mask, specifying which operations are padded
        pad_mask = torch.arange(n_ops_max).unsqueeze(0).expand(*batch_size, -1)
        pad_mask = pad_mask.ge(n_ops_batch[:, None].expand_as(pad_mask))

        # determine the id of the end operation for each job
        end_op_per_job = n_ope_per_job.cumsum(1) - 1

        # determine the id of the starting operation for each job
        # (bs, num_jobs)
        start_op_per_job = torch.cat(
            (
                torch.zeros((*batch_size, 1)).to(end_op_per_job),
                end_op_per_job[:, :-1] + 1,
            ),
            dim=1,
        )

        # here we simulate the eligible machines per operation and the processing times
        n_eligible_per_ops = torch.randint(
            self.min_eligible_ma_per_op,
            self.max_eligible_ma_per_op + 1,
            (*batch_size, n_ops_max),
        )
        n_eligible_per_ops[pad_mask] = 0

        # simulate processing times for machine-operation pairs
        # (bs, num_mas, n_ops_max)
        proc_times = self._simulate_processing_times(n_eligible_per_ops)
        trucks_times = self._simulate_trucks_times(batch_size)

        td = TensorDict(
            {
                "start_op_per_job": start_op_per_job,
                "end_op_per_job": end_op_per_job,
                "proc_times": proc_times,
                "trucks_times": trucks_times,
                "pad_mask": pad_mask,
            },
            batch_size=batch_size,
        )

        return td


class FJSPTFileGenerator(Generator):
    """Data generator for the Flexible Job-Shop Scheduling Problem with Transportation resources (FJSPT)
    using instance files"""

    def __init__(self, proc_file_path: str, trucks_file_path: str, **unused_kwargs):
        self.proc_files = self.list_files(proc_file_path)
        self.num_samples = len(self.proc_files)

        if len(unused_kwargs) > 0:
            log.error(f"Found {len(unused_kwargs)} unused kwargs: {unused_kwargs}")

        if len(self.proc_files) > 1:
            n_ops_max = get_max_ops_from_files(self.proc_files)

        ret = map(partial(read, max_ops=n_ops_max), self.proc_files)

        td_list, num_jobs, num_machines, num_trucks, max_ops_per_job = list(zip(*list(ret)))
        num_jobs, num_machines, num_trucks = map(lambda x: x[0], (num_jobs, num_machines, num_trucks))
        max_ops_per_job = max(max_ops_per_job)

        trucks_times = torch.tensor(file2lines(trucks_file_path))
        assert trucks_times.ndim == 2 and trucks_times.size(0) == trucks_times.size(1)
        # !!! есть датасеты с матрицей для большего числа станков - обрезаем матрицу
        trucks_times = trucks_times[:num_machines + 1, :num_machines + 1]
        td_list["trucks_times"] = [trucks_times for _ in range(self.num_samples)]

        self.td = torch.cat(td_list, dim=0)
        self.num_mas = num_machines
        self.num_jobs = num_jobs
        self.num_trucks = num_trucks
        self.max_ops_per_job = max_ops_per_job
        self.n_ops_max = max_ops_per_job * num_jobs

        self.start_idx = 0

    def _generate(self, batch_size: list[int]) -> TensorDict:
        batch_size = np.prod(batch_size)
        if batch_size > self.num_samples:
            log.warning(
                f"Only found {self.num_samples} instance files, but specified dataset size is {batch_size}"
            )
        end_idx = self.start_idx + batch_size
        td = self.td[self.start_idx : end_idx]
        self.start_idx += batch_size
        if self.start_idx >= self.num_samples:
            self.start_idx = 0
        return td

    @staticmethod
    def list_files(path):
        """
        Функция находит в папку, игнорирует другие папки внутри неё, собирает пути
        ко всем «настоящим» файлам и проверяет, чтобы список не оказался пустым.
        """
        import os

        files = [
            os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))
        ]
        assert len(files) > 0
        return files
