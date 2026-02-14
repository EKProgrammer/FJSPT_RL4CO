import torch

from tensordict.tensordict import TensorDict

from rl4co.envs.common.utils import Generator
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class FJSPGenerator(Generator):
    """Data generator for the Flexible Job-Shop Scheduling Problem (FJSP).

    Args:
        num_stage: количество стадий
        num_machine: количество машин
        num_job: количество задач
        min_time: минимальное время обработки операции на машине
        max_time: максимальное время обработки операции на машине
        flatten_stages: флаг, объединять ли стадии в один список операций

    Returns:
        A TensorDict with the following key:
            start_op_per_job [batch_size, num_jobs]: first operation of each job
            Для каждого батч-элемента и каждой job хранит индекс (!!!) первой операции этой job.
            Пример:
            batch_size = 2, num_jobs = 3
            start_op_per_job = [[0, 3, 6],
                                [0, 2, 5]]

            end_op_per_job [batch_size, num_jobs]: last operation of each job
            Для каждой job хранит индекс последней операции.
            Пример:
            end_op_per_job = [[2, 5, 8],
                              [1, 4, 7]]

            proc_times [batch_size, num_machines, total_n_ops]: processing time of ops on machines
            Основной тензор, кодирующий время обработки каждой операции на каждой машине.
            proc_times[b, m, o] = время, которое займёт операция o на машине m в батче b.
            Если операция не может выполняться на машине → proc_times[b, m, o] = 0.
            Пример:
            proc_times[0, :, :] =
            [[5, 0, 3],  # машина 0: op0=5, op1 нельзя, op2=3
             [0, 2, 4],  # машина 1: ...
             [1, 1, 0]]  # машина 2: ...
            Этот тензор задаёт возможные дизъюнктивные рёбра по машинам.

            pad_mask [batch_size, total_n_ops]: not all instances have the same number of ops, so padding is used
            Маска для фиктивных операций, которые появились из-за выравнивания до n_ops_max (максимальное число операций).
            True → операция padding (не настоящая)
            False → операция реально существует
            Пример:
            pad_mask[0] = [False, False, False, False, True, True]
            Первые 4 операции реально существуют, последние 2 — фиктивные.
    """

    def __init__(
        self,
        num_jobs: int = 10,
        num_machines: int = 5,
        min_ops_per_job: int = 4,
        max_ops_per_job: int = 6,
        min_processing_time: int = 1,
        max_processing_time: int = 20,
        min_eligible_ma_per_op: int = 1,
        max_eligible_ma_per_op: int = None,
        same_mean_per_op: bool = True,
        **unused_kwargs,
    ):
        self.num_jobs = num_jobs
        self.num_mas = num_machines
        self.min_ops_per_job = min_ops_per_job
        self.max_ops_per_job = max_ops_per_job
        self.min_processing_time = min_processing_time
        self.max_processing_time = max_processing_time
        self.min_eligible_ma_per_op = min_eligible_ma_per_op
        self.max_eligible_ma_per_op = max_eligible_ma_per_op or num_machines
        # determines whether to use a fixed number of total operations or let it vary between instances
        # NOTE: due to the way rl4co builds datasets, we need a fixed size here
        self.n_ops_max = max_ops_per_job * num_jobs
        self.same_mean_per_op = same_mean_per_op
        # same_mean_per_op:
        # если True → времена обработки одной операции на разных машинах похожи,
        # если False → полностью случайные.

        # FFSP environment doen't have any other kwargs
        if len(unused_kwargs) > 0:
            log.error(f"Found {len(unused_kwargs)} unused kwargs: {unused_kwargs}")

    def _simulate_processing_times(self, n_eligible_per_ops: torch.Tensor) -> torch.Tensor:
        """
        Для каждой операции o и каждой машины m функция:
        1. Случайно выбирает подмножество допустимых машин
        2. Генерирует время обработки p_{m,o}
        3. Обнуляет p_{m,o}, если операция не может выполняться на машине
        """

        # n_eligible_per_ops - сколько машин разрешено для данной операции
        bs, n_ops_max = n_eligible_per_ops.shape

        # Список машин для каждой операции
        # (bs, max_ops, machines)
        ma_seq_per_ops = torch.arange(1, self.num_mas + 1)[None, None].expand(
            bs, n_ops_max, self.num_mas
        )
        # Создаёт тензор:
        # [1, 2, 3, ..., num_mas] преобразуется в...
        # Пример (bs=1, n_ops_mask=3, num_mas=4):
        # [
        #   [1,2,3,4],   # op0
        #   [1,2,3,4],   # op1
        #   [1,2,3,4],   # op2
        # ]

        # Шаблон допустимых машин (до перемешивания)
        # Для каждой операции создаётся строка: первые k = n_eligible машин → 1, остальные → 0
        # generate a matrix of size (ops, mas) per batch, each row having as many ones as the operation eligible machines
        # E.g. n_eligible_per_ops=[1,3,2]; num_mas=4
        # [[1,0,0,0],
        #   1,1,1,0],
        #   1,1,0,0]]
        # This will be shuffled randomly to generate a machine-operation mapping
        ma_ops_edges_unshuffled = torch.Tensor.float(
            ma_seq_per_ops <= n_eligible_per_ops[..., None]
        )

        # random shuffling
        # torch.rand_like - создаёт тензор того же размера, заполненный случайными числами:
        # rand = [[[0.91, 0.44, 0.05, 0.66], [...], [...], [...]], ...]
        # argsort(dim=-1) возвращает индексы, которые сортируют тензор.
        # argsort(rand) = [[[2, 1, 3, 0], [...], [...], [...]], ...]
        # Итог: idx[b, o, :] = случайная перестановка [0, 1, ..., num_mas-1]
        idx = torch.rand_like(ma_ops_edges_unshuffled).argsort()
        # gather(dim=2, idx): берёт элементы по размерности dim=2 (машины)
        ma_ops_edges = ma_ops_edges_unshuffled.gather(2, idx).transpose(1, 2)
        # gather: до - [1, 1, 0, 0], после - [0, 1, 0, 1]
        # transpose: до - (bs, n_ops, num_mas), после - (bs, num_mas, n_ops)

        # (bs, max_ops, machines)
        if self.same_mean_per_op:
            # Если одна операция имеет примерно одинаковую сложность на разных машинах.
            proc_times = torch.ones((bs, self.num_mas, n_ops_max))
            # Выбираем среднее время на операцию
            proc_time_means = torch.randint(
                self.min_processing_time, self.max_processing_time, (bs, n_ops_max)
            )
            # Нижняя граница
            # low_bounds = max(min_time, 0.8 * mean)
            low_bounds = torch.maximum(
                torch.full_like(proc_times, self.min_processing_time),
                (proc_time_means * (1 - 0.2)).round().unsqueeze(1),
            )
            # Верхняя граница
            # high_bounds = min(max_time, 1.2 * mean) + 1
            high_bounds = (
                torch.minimum(
                    torch.full_like(proc_times, self.max_processing_time),
                    (proc_time_means * (1 + 0.2)).round().unsqueeze(1),
                )
                + 1
            )
            # p_{m,o} ~ U([0.8 μ_o, 1.2 μ_o])
            # proc_times = randint(low_bounds, high_bounds)
            proc_times = (
                torch.randint(2**63 - 1, size=proc_times.shape) % (high_bounds - low_bounds)
                + low_bounds
            )
        else:
            # p_{m,o} ~ U(min, max)
            proc_times = torch.randint(
                self.min_processing_time,
                self.max_processing_time + 1,
                size=(bs, self.num_mas, n_ops_max),
            )

        # remove proc_times for which there is no corresponding ma-ops connection
        # Обнуление недопустимых машин
        proc_times = proc_times * ma_ops_edges
        return proc_times

    def _generate(self, batch_size) -> TensorDict:
        """
        Генерирует batch независимых экземпляров FJSP (описания задачи), каждый из которых задаётся:
            цепочками операций внутри job,
            допустимыми машинами,
            processing times,
            padding-маской.
        """
        # simulate how many operations each job has
        # Сколько операций у каждого job
        n_ope_per_job = torch.randint(
            self.min_ops_per_job,
            self.max_ops_per_job + 1,
            size=(*batch_size, self.num_jobs),
        )

        # determine the total number of operations per batch instance (which may differ)
        # Общее число операций в батче
        n_ops_batch = n_ope_per_job.sum(1)  # (bs)
        # determine the maximum total number of operations over all batch instances
        # Фиксированный максимум
        n_ops_max = self.n_ops_max or n_ops_batch.max()

        # generate a mask, specifying which operations are padded
        # pad_mask = torch.arange(n_ops_max)   shape = (n_ops_max,)
        # pad_mask.unsqueeze(0)                shape = (1, n_ops_max)
        # pad_mask.expand(*batch_size, -1)     shape = (*batch_size, n_ops_max)
        pad_mask = torch.arange(n_ops_max).unsqueeze(0).expand(*batch_size, -1)
        # Сравниваем: op_id >= реальное_число_операций
        # Если True → padding
        #
        # Пример:
        # pad_mask (до) = [0,1,2,3,4,5,6,7,8,9]
        # n_ops_batch   = [7]
        #
        # Сравнение:
        # 0 >= 7 → False
        # 1 >= 7 → False
        # ...
        # 6 >= 7 → False
        # 7 >= 7 → True
        # 8 >= 7 → True
        # 9 >= 7 → True
        #
        # Результат:
        # pad_mask = [F, F, F, F, F, F, F, T, T, T]
        pad_mask = pad_mask.ge(n_ops_batch[:, None].expand_as(pad_mask))

        # determine the id of the end operation for each job
        # Индексы последних операций job
        # Пример:
        # n_ope_per_job = [3, 5, 4]
        #
        # Но Env и Policy работают с глобальной нумерацией операций:
        # op_id = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
        #
        # Нужно построить соответствие:
        # Job 0 → ops 0..2
        # Job 1 → ops 3..7
        # Job 2 → ops 8..11
        #
        # cumsum = [3, 8, 12] - общее количество операций до конца каждого job
        # end = [2, 7, 11] - индексация с нуля
        end_op_per_job = n_ope_per_job.cumsum(1) - 1

        # determine the id of the starting operation for each job
        # (bs, num_jobs)
        # Индексы первых операций job
        # Для job:
        # Job 0 всегда начинается с операции 0
        # Каждый следующий job начинается после последнего предыдущего
        start_op_per_job = torch.cat(
            (
                torch.zeros((*batch_size, 1)).to(end_op_per_job),
                end_op_per_job[:, :-1] + 1,
            ),
            dim=1,
        )

        # here we simulate the eligible machines per operation and the processing times
        # Сколько машин доступно каждой операции
        n_eligible_per_ops = torch.randint(
            self.min_eligible_ma_per_op,
            self.max_eligible_ma_per_op + 1,
            (*batch_size, n_ops_max),
        )
        n_eligible_per_ops[pad_mask] = 0

        # simulate processing times for machine-operation pairs
        # (bs, num_mas, n_ops_max)
        proc_times = self._simulate_processing_times(n_eligible_per_ops)

        td = TensorDict(
            {
                "start_op_per_job": start_op_per_job,
                "end_op_per_job": end_op_per_job,
                "proc_times": proc_times,
                "pad_mask": pad_mask,
            },
            batch_size=batch_size,
        )

        return td
