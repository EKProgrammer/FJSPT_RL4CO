import torch

from einops import rearrange, reduce
from tensordict.tensordict import TensorDict
from torchrl.data import Bounded, Composite, Unbounded

from rl4co.envs.common.base import RL4COEnvBase as EnvBase
from rl4co.utils.ops import gather_by_index, sample_n_random_actions

from . import INIT_FINISH, NO_OP_ID
from .generator import FJSPGenerator
from .render import render
from .utils import calc_lower_bound, get_job_ops_mapping, op_is_ready


class FJSPEnv(EnvBase):
    """Flexible Job-Shop Scheduling Problem (FJSP) environment
    На каждом шаге agent выбирает комбинацию (job, machine). Следующая операция выбранной job выполняется на выбранной машине.
    Reward равна 0, пока agent не запланирует все operations всех jobs.
    После этого reward равна (-)makespan (длине расписания): Максимизация reward эквивалентна минимизации makespan.

    Наблюдения (Observations):
        - time: текущее время
        - next_op: следующая операция для каждой работы
        - proc_times: времена обработки для пар (операция, машина)
        - pad_mask: маска для дополненных (padding) операций
        - start_op_per_job: id первой операции для каждой работы
        - end_op_per_job: id последней операции для каждой работы
        - start_times: время начала операции (0, если не запланирована)
        - finish_times: время окончания операции (INIT_FINISH, если не запланирована)
        - job_ops_adj: матрица смежности работа–операция (adjacency matrix specifying job-operation affiliation)
        - ops_job_map: то же самое, но в виде id работ (same as above but using ids of jobs to indicate affiliation)
        - ops_sequence_order: порядок выполнения операций внутри работы
        - ma_assignment: какая операция назначена на какую машину
        - busy_until: до какого времени машина занята
        - num_eligible: число машин, которые могут выполнить операцию
        - job_in_process: находится ли работа в обработке
        - job_done:  завершена ли работа

    Ограничения (Constraints):
        Agent не может выбирать:
         - машины, которые сейчас заняты
         - работы, которые уже завершены
         - работы, которые сейчас обрабатываются
         - пары (работа, машина), где машина не может выполнить следующую операцию

    Условие завершения (Finish condition):
        - агент запланировал все операции всех работ

    Награда (Reward):
        - отрицательный makespan итогового расписания

    Аргументы:
        generator: объект FJSPGenerator для генерации данных
        generator_params: параметры генератора
        mask_no_ops: если True, агент не может выбирать "ожидание" (если только задача не завершена)
    """

    name = "fjsp"

    def __init__(
        self,
        generator: FJSPGenerator = None,
        generator_params: dict = {},
        mask_no_ops: bool = True,
        check_mask: bool = False,
        stepwise_reward: bool = False,
        **kwargs,
    ):
        # Вызов конструктора базовой среды RL4CO.
        # check_solution=False - не проверять решение на корректность (для ускорения).
        super().__init__(check_solution=False, **kwargs)
        if generator is None:
            # генерируем случайные инстансы
            generator = FJSPGenerator(**generator_params)
        self.generator = generator
        self._num_mas = generator.num_mas  # число машин
        self._num_jobs = generator.num_jobs  # число работ
        self._n_ops_max = generator.max_ops_per_job * self.num_jobs  # максимальное число операций = max_ops_per_job * num_jobs

        self.mask_no_ops = mask_no_ops  # можно ли ждать
        self.check_mask = check_mask  # проверять ли корректность маски
        self.stepwise_reward = stepwise_reward  # использовать ли промежуточную награду
        self._make_spec(self.generator)
        # Создаёт:
        # - observation_spec
        # - action_spec
        # - reward_spec
        # - done_spec
        # То есть формально описывает:
        # - какие наблюдения
        # - какие действия
        # - какие типы и размеры тензоров
        #
        # Это нужно TorchRL / RL4CO.

    @property
    def num_mas(self):
        # Геттер для количества машин
        return self._num_mas

    @property
    def num_jobs(self):
        # геттер для количества работ
        return self._num_jobs

    @property
    def n_ops_max(self):
        # Геттер для максимального числа операций
        return self._n_ops_max

    def set_instance_params(self, td):
        # start_op_per_job имеет форму (bs, num_jobs)
        # size(1) = число jobs в этом конкретном инстансе
        self._num_jobs = td["start_op_per_job"].size(1)
        # proc_times имеет форму (bs, num_mas, n_ops_max)
        # size(1) = число машин
        self._num_mas = td["proc_times"].size(1)
        # size(2) = число операций (с padding)
        self._n_ops_max = td["proc_times"].size(2)

    def _decode_graph_structure(self, td: TensorDict):
        """
        Эта функция строит граф операций для FJSP:
            - какие операции принадлежат каким jobs
            - в каком порядке операции внутри job
            - какие операции являются предшественниками / последователями
            - adjacency-матрицы для графовых нейросетей
        То есть она переводит: табличное описание FJSP в графовую структуру задач
        """

        batch_size = td.batch_size
        # индекс первой операции каждого job
        start_op_per_job = td["start_op_per_job"]
        # индекс последней операции каждого job
        end_op_per_job = td["end_op_per_job"]
        # какие операции являются padding
        pad_mask = td["pad_mask"]
        n_ops_max = td["pad_mask"].size(-1)

        # here we will generate the operations-job mapping:
        ops_job_map, ops_job_bin_map = get_job_ops_mapping(
            start_op_per_job, end_op_per_job, n_ops_max
        )
        # Эта функция строит:
        #     - ops_job_map с формой (bs, n_ops_max),
        # который содержит для каждой операции - id job, к которому она принадлежит
        # Пример: ops_job_map = [0,0,1,1,1,2,2,2,2,...]
        #     - ops_job_bin_map с формой (bs, num_jobs, n_ops_max) - это
        # бинарная матрица: ops_job_bin_map[j, o] = 1, если операция o принадлежит job j.
        # Это матрица инцидентности job-operation.

        # mask invalid edges (caused by padding)
        ops_job_bin_map[pad_mask.unsqueeze(1).expand_as(ops_job_bin_map)] = 0
        # Пример:
        # 3 работы (jobs)
        # максимальное число операций n_ops_max = 6
        #
        # реальные операции:
        # Job 0: ops [0, 1]
        # Job 1: ops [2, 3]
        # Job 2: ops [4]        # только одна операция
        #
        # Padding:
        # op 5 = padding
        #
        # Тогда:
        # start_op_per_job = [0, 2, 4]
        # end_op_per_job   = [1, 3, 4]
        # pad_mask         = [F, F, F, F, F, T]
        #
        # ops_job_bin_map (до mask)
        # Форма: (num_jobs, n_ops_max)
        #           op0 op1 op2 op3 op4 op5
        # Job 0:     1   1   0   0   0   0
        # Job 1:     0   0   1   1   0   0
        # Job 2:     0   0   0   0   1   1   <-- padding op5
        #
        # ops_job_bin_map (после mask)
        #           op0 op1 op2 op3 op4 op5
        # Job 0:     1   1   0   0   0   0
        # Job 1:     0   0   1   1   0   0
        # Job 2:     0   0   0   0   1   0   <-- padding УДАЛЁН

        # generate for each batch a sequence specifying the position of all operations in their respective jobs,
        # e.g. [0,1,0,0,1,2,0,1,2,3,0,0] for jops with n_ops=[2,1,3,4,1,1]
        # (bs, max_ops)
        ops_seq_order = torch.sum(ops_job_bin_map * (ops_job_bin_map.cumsum(2) - 1), dim=1)
        # Для каждой операции вычисляется её номер внутри job: 0, 1, 2, ...
        # * cumsum(2) считает порядковый номер операции в job
        # * -1 → делаем 0-based
        # * умножаем на бинарную маску
        # * суммируем по job'ам

        # predecessor and successor adjacency matrices
        pred = torch.diag_embed(torch.ones(n_ops_max - 1), offset=-1)[None].expand(*batch_size, -1, -1)
        # diag_embed размещает входной вектор на указанной диагонали квадратной матрицы
        # offset=-1 означает первая поддиагональ (ниже главной диагонали) (pred[i, i-1] = 1)
        # [None] Добавляет новую размерность в начало
        # .expand(*batch_size, -1, -1) Расширяет тензор до нужного размера батча
        #
        # Пример:
        # Job 0: ops [0, 1]
        # Job 1: ops [2, 3, 4]
        # Job 2: ops [5]
        # ops_seq_order = [0, 1, 0, 1, 2, 0]
        #
        # Матрица predecessor:
        #       0 1 2 3 4 5   (откуда)
        # 0:    0 0 0 0 0 0
        # 1:    1 0 0 0 0 0
        # 2:    0 1 0 0 0 0
        # 3:    0 0 1 0 0 0
        # 4:    0 0 0 1 0 0
        # 5:    0 0 0 0 1 0
        # (куда)
        #
        # Это говорит:
        # op1 <- op0
        # op2 <- op1
        # op3 <- op2
        # ...
        # ❌ Но это НЕ учитывает job boundaries!

        # the start of the sequence (of each job) does not have a predecessor, therefore we can
        # mask all first ops of a job in the predecessor matrix
        pred = pred * ops_seq_order.gt(0).unsqueeze(-1).expand_as(pred).to(pred)
        # .gt(0) - поэлементное сравнение "больше 0" (greater than)
        # Результат: булевский тензор той же формы, где True для элементов > 0, False для элементов ≤ 0
        # .unsqueeze(-1) Добавляет размерность в конец тензора
        # .to(pred) Приводит тип тензора к типу pred
        # pred * ... Поэлементное умножение
        #
        # Продолжение примера выше:
        # ops_seq_order = [0,1,0,1,2,0]
        # ops_seq_order.gt(0) = [F, T, F, T, T, F]
        # Смысл:
        # True = НЕ первая операция job
        # False = первая операция job
        #
        # unsqueeze(-1):
        # [[F],
        #  [T],
        #  [F],
        #  [T],
        #  [T],
        #  [F]]
        #
        # expand_as(pred)
        # строка 0: все F
        # строка 1: все T
        # строка 2: все F
        # строка 3: все T
        # строка 4: все T
        # строка 5: все F
        #
        # pred * mask - обнуляем predecessor у первых ops job:
        # Было:
        # 1 <- 0   (OK)
        # 2 <- 1   (НО op2 — первая в job1 → НЕ должно быть!)
        #
        # После mask:
        # строка 2 = 0  → predecessor убран
        # строка 5 = 0  → predecessor убран
        #
        # Итог pred:
        # op1 <- op0   (осталось)
        # op3 <- op2   (осталось)
        # op4 <- op3   (осталось)
        #
        # А:
        # op2 (первая job1) - нет predecessor
        # op5 (первая job2) - нет predecessor

        # аналогично
        succ = torch.diag_embed(torch.ones(n_ops_max - 1), offset=1)[None].expand(*batch_size, -1, -1)
        # torch.diag_embed(torch.ones(n_ops_max - 1), offset=1)  Создаёт: succ[i, i+1] = 1
        # succ =
        # 0 1 0 0 0 0   # 0 -> 1
        # 0 0 1 0 0 0   # 1 -> 2
        # 0 0 0 1 0 0   # 2 -> 3
        # 0 0 0 0 1 0   # 3 -> 4
        # 0 0 0 0 0 1   # 4 -> 5
        # 0 0 0 0 0 0

        # apply the same logic as above to mask the last op of a job, which does not have a successor. The last job of a job
        # always comes before the 1st op of the next job, therefore performing a left shift of the ops seq tensor here
        succ = succ * torch.cat(
            (ops_seq_order[:, 1:], ops_seq_order.new_full((*batch_size, 1), 0)), dim=1
        ).gt(0).to(succ).unsqueeze(-1).expand_as(succ)
        # ops_seq_order.new_full((*batch_size, 1), 0)
        # Это значит:
        #     Создай тензор формы (*batch_size, 1)
        #     на том же device и с тем же dtype, что ops_seq_order и заполни его нулями.
        # ops_seq_order[:, 1:] = [1, 0, 1, 2, 0]
        # Добавляем 0 в конец (cat): [1, 0, 1, 2, 0, 0]
        # .gt(0): mask_vec = [T, F, T, T, F, F]
        #
        # 0 -> 1    ✅ (job0)
        # 1 -> X    ❌ (job0 last)  УБРАНО
        # 2 -> 3    ✅ (job1)
        # 3 -> 4    ✅ (job1)
        # 4 -> X    ❌ (job1 last)  УБРАНО
        # 5 -> X    ❌ (job2 single)

        # pred =
        #       0 1 2 3 4 5   (j = predecessor)
        # i=0:  0 0 0 0 0 0   # op0 first in job0
        # i=1:  1 0 0 0 0 0   # op1 <- op0
        # i=2:  0 0 0 0 0 0   # op2 first in job1
        # i=3:  0 0 1 0 0 0   # op3 <- op2
        # i=4:  0 0 0 1 0 0   # op4 <- op3
        # i=5:  0 0 0 0 0 0   # op5 first (and last) in job2

        # succ =
        #       0 1 2 3 4 5   (j = successor)
        # i=0:  0 1 0 0 0 0   # op0 -> op1
        # i=1:  0 0 0 0 0 0   # op1 last in job0
        # i=2:  0 0 0 1 0 0   # op2 -> op3
        # i=3:  0 0 0 0 1 0   # op3 -> op4
        # i=4:  0 0 0 0 0 0   # op4 last in job1
        # i=5:  0 0 0 0 0 0   # op5 single op job2

        # adjacency matrix = predecessors, successors and self loops
        # форма: (bs, max_ops, max_ops, 2)
        # Это ориентированный граф операций: [..., 0] = predecessor edges, [..., 1] = successor edges
        ops_adj = torch.stack((pred, succ), dim=3)

        td = td.update(
            {
                "ops_adj": ops_adj,
                "job_ops_adj": ops_job_bin_map,
                "ops_job_map": ops_job_map,
                # "op_spatial_enc": ops_spatial_enc,
                "ops_sequence_order": ops_seq_order,
            }
        )

        return td, n_ops_max

    def _reset(self, td: TensorDict = None, batch_size=None) -> TensorDict:
        self.set_instance_params(td)

        td_reset = td.clone()

        td_reset, n_ops_max = self._decode_graph_structure(td_reset)

        # schedule
        start_op_per_job = td_reset["start_op_per_job"]
        start_times = torch.zeros((*batch_size, n_ops_max))
        finish_times = torch.full((*batch_size, n_ops_max), INIT_FINISH)
        ma_assignment = torch.zeros((*batch_size, self.num_mas, n_ops_max))

        # состояние машин
        # reset feature space
        busy_until = torch.zeros((*batch_size, self.num_mas))

        # матрица допустимости ops-machine
        # (bs, ma, ops)
        ops_ma_adj = (td_reset["proc_times"] > 0).to(torch.float32)
        # 1 → машина может выполнять операцию
        # 0 → не может

        # сколько машин может выполнять op
        # (bs, ops)
        num_eligible = torch.sum(ops_ma_adj, dim=1)

        td_reset = td_reset.update(
            {
                "start_times": start_times,
                "finish_times": finish_times,
                "ma_assignment": ma_assignment,
                "busy_until": busy_until,
                "num_eligible": num_eligible,
                "next_op": start_op_per_job.clone().to(torch.int64),
                "ops_ma_adj": ops_ma_adj,
                "op_scheduled": torch.full((*batch_size, n_ops_max), False),  # Какие операции уже запланированы
                "job_in_process": torch.full((*batch_size, self.num_jobs), False),  # Для каждой job: сейчас выполняется или нет
                "reward": torch.zeros((*batch_size,), dtype=torch.float32),  # Награда на шаг.
                "time": torch.zeros((*batch_size,)),  # Текущее время среды.
                "job_done": torch.full((*batch_size, self.num_jobs), False),  # Job полностью завершена?
                "done": torch.full((*batch_size, 1), False),
            },
        )

        td_reset.set("action_mask", self.get_action_mask(td_reset))
        # add additional features to tensordict
        td_reset["lbs"] = calc_lower_bound(td_reset)
        td_reset = self._get_features(td_reset)

        return td_reset

    def _get_job_machine_availability(self, td: TensorDict):
        # Эта функция определяет, какие пары (job, machine) сейчас НЕЛЬЗЯ выбирать.
        # Важно: здесь True = запрещено, False = разрешено

        batch_size = td.size(0)

        # (bs, jobs, machines)
        action_mask = torch.full((batch_size, self.num_jobs, self.num_mas), False).to(td.device)
        # ВСЁ разрешено

        # mask jobs that are done already
        action_mask.add_(td["job_done"].unsqueeze(2))
        # td["job_done"]  shape = (bs, num_jobs)
        # job_done = [F, T, F]
        # .unsqueeze(2): (bs, num_jobs, 1)
        # Broadcast по machines: job 1 запрещён на ВСЕХ машинах
        # add_ == "+="

        # as well as jobs that are currently processed
        action_mask.add_(td["job_in_process"].unsqueeze(2))
        # Если job сейчас выполняется: нельзя запускать следующую операцию этого job

        # mask machines that are currently busy
        action_mask.add_(td["busy_until"].gt(td["time"].unsqueeze(1)).unsqueeze(1))
        # td["busy_until"].gt(td["time"].unsqueeze(1))
        # busy_until  (bs, num_mas)
        # time  (bs,) → unsqueeze(1) → (bs, 1)
        # .unsqueeze(1): (bs, 1, num_mas)
        # Broadcast по jobs:
        # gt (greater than): True, если машина занята

        # exclude job-machine combinations, where the machine cannot process the next op of the job
        # td["proc_times"]  shape = (bs, num_mas, n_ops)
        # td["next_op"]     shape = (bs, num_jobs)
        # → unsqueeze(1):   shape = (bs, 1, num_jobs)
        #
        # gather_by_index - возьми из src элементы по индексам idx вдоль измерения dim, сохрани все остальные измерения
        # Берём processing time:
        # для каждой пары (job, machine)
        # ТОЛЬКО для next_op(job)
        #
        # Результат до transpose: shape = (bs, num_mas, num_jobs)
        # После transpose: shape = (bs, num_jobs, num_mas)
        #
        # next_ops_proc_times == 0
        # Если 0 → машина НЕ может выполнять эту операцию.
        next_ops_proc_times = gather_by_index(
            td["proc_times"], td["next_op"].unsqueeze(1), dim=2, squeeze=False
        ).transpose(1, 2)
        action_mask.add_(next_ops_proc_times == 0)
        return action_mask

    def get_action_mask(self, td: TensorDict) -> torch.Tensor:
        """
        Строит допустимые действия:
        какие (job, machine) можно
        можно ли NO-OP
        """
        # 1 indicates machine or job is unavailable at current time step
        action_mask = self._get_job_machine_availability(td)
        # action_mask[b, j, m] = True  → НЕЛЬЗЯ выбрать (j,m)
        # action_mask[b, j, m] = False → МОЖНО выбрать (j,m)

        # no_op_mask = маска допустимости действия NO-OP (no operation)
        if self.mask_no_ops:   # можно ли ждать
            # Если mask_no_ops=True, то ждать можно ТОЛЬКО когда инстанс уже закончен.
            # masking is only allowed if instance is finished
            no_op_mask = td["done"]  # (bs, 1)
        else:
            # if no job is currently processed and instance is not finished yet, waiting is not allowed
            no_op_mask = (td["job_in_process"].any(1, keepdims=True) & (~td["done"])) | td["done"]
            # Случай A: есть job в процессе, не done
            # job_in_process = True
            # done = False
            # → no_op_mask = True
            # → ждать РАЗРЕШЕНО
            # Логично: если идёт обработка — можно подождать.
            #
            # Случай B: нет job в процессе, не done
            # job_in_process = False
            # done = False
            # → no_op_mask = False
            # → ждать ЗАПРЕЩЕНО
            # Логично: если всё свободно, агент ОБЯЗАН назначить работу, а не ждать.
            #
            # Случай C: done = True
            # done = True
            # → no_op_mask = True
            # Технически ждать разрешено, но это уже конец эпизода.

        # flatten action mask to correspond with logit shape
        action_mask = rearrange(action_mask, "bs j m -> bs (j m)")
        # NOTE: 1 means feasible action, 0 means infeasible action
        mask = torch.cat((no_op_mask, ~action_mask), dim=1)
        # ~action_mask: True = МОЖНО выбрать (j,m)

        return mask

    def _translate_action(self, td):
        """This function translates an action into a machine, job tuple."""
        # Policy выбирает одно целое число:
        # action ∈ {0, 1, 2, ..., num_jobs * num_mas - 1} откуда эта информация?
        selected_job = td["action"] // self.num_mas  # td["action"].shape = (bs,)
        # gather берёт для каждого batch — next_op именно выбранной job
        selected_op = td["next_op"].gather(1, selected_job[:, None]).squeeze(1)
        selected_machine = td["action"] % self.num_mas
        return selected_job, selected_op, selected_machine

    def _step(self, td: TensorDict):
        """
        _step делает один шаг среды для батча инстансов FJSP.
        За один вызов он может:
        назначить операцию (job–machine),
        или выполнить NO-OP (подождать),
        или автоматически продвинуть время вперёд,
        пересчитать маски, признаки и награды.
        Важно:
        один шаг RL != один шаг времени.
        Один шаг RL = “пока на текущем времени есть что делать”.
        """
        # cloning required to avoid inplace operation which avoids gradient backtracking
        td = td.clone()
        # Зачем:
        # Внутри много inplace-операций (add_, scatter, subtract_)
        # TorchRL требует сохранять корректный autograd graph
        # Поэтому работаем с копией

        td["action"].subtract_(1)
        # Что было ДО
        # Policy выбирает действие из:
        # 0               → NO-OP
        # 1 .. J*M        → (job, machine)
        # Что стало ПОСЛЕ
        # -1              → NO-OP
        # 0 .. J*M-1      → (job, machine)
        # Это делается один раз, чтобы дальше код работал с:
        # NO_OP_ID = -1
        # обычными индексами для (job, machine)

        dones = td["done"].squeeze(1)  # dones.shape = (bs,)
        # specify which batch instances require which operation
        no_op = td["action"].eq(NO_OP_ID)  # no_op.shape = (bs,)
        # Теперь no_op[b] = True -> агент выбрал "ждать"
        no_op = no_op & ~dones
        # Убираем done из NO-OP
        req_op = ~no_op & ~dones  # req_op.shape == (bs,)
        # req_op - это оставшиеся элементы батча, где не NO-OP и не done
        # req_op[b] = True - в batch b: эпизод НЕ done, агент выбрал реальное действие (job, machine)

        # transition to next time for no op instances
        # Если есть NO-OP -> двигаем время
        if no_op.any():
            td, dones = self._transit_to_next_time(no_op, td)

        # select only instances that perform a scheduling action
        td_op = td.masked_select(req_op)
        # В td_op лежат ТОЛЬКО инстансы, где:
        # эпизод не завершён
        # агент выбрал (job, machine)

        td_op = self._make_step(td_op)
        # update the tensordict
        # Обновляем исходный td результатами реальных действий
        td[req_op] = td_op

        # action mask
        td.set("action_mask", self.get_action_mask(td))
        # Пересчёт допустимых действий
        # Потому что:
        # машины могли освободиться
        # новые операции стали доступными

        step_complete = self._check_step_complete(td, dones)
        # Если всё ещё нечего выбирать -> двигаем время дальше.
        while step_complete.any():
            # _transit_to_next_time перематывает время до минимального busy until, то есть до момента,
            # когда хотя бы одна машина освободится. Она не знает, какая операция станет готовой
            # на следующем шаге, а просто гарантирует, что время продвинулось до ближайшего события,
            # после чего агент может снова действовать.
            td, dones = self._transit_to_next_time(step_complete, td)
            td.set("action_mask", self.get_action_mask(td))
            step_complete = self._check_step_complete(td, dones)
        if self.check_mask:  # проверять ли корректность маски
            assert reduce(td["action_mask"], "bs ... -> bs", "any").all()

        # Проверяем, нужно ли выдавать шаговую награду (reward на каждом действии), а не только в конце эпизода.
        if self.stepwise_reward:
            # if we require a stepwise reward, the change in the calculated lower bounds could serve as such
            lbs = calc_lower_bound(td)
            # calc_lower_bound вычисляет нижнюю границу времени завершения операций для текущего состояния среды
            # Возвращает LBs - тензор, который содержит нижнюю границу finish time для каждой операции (bs, n_ops_max)
            td["reward"] = -(lbs.max(1).values - td["lbs"].max(1).values)
            # lbs.max(1).values — максимальная lower bound по операциям для каждого элемента батча.
            # td["lbs"].max(1).values — максимальная lower bound, сохранённая на предыдущем шаге.
            # Разница (lbs.max - td["lbs"].max) показывает изменение минимально возможного finish time.
            # Минус - означает: уменьшение lower bound (лучше расписание) даёт положительную награду,
            # а увеличение — отрицательную.
            td["lbs"] = lbs
        else:
            td["lbs"] = calc_lower_bound(td)

        # add additional features to tensordict
        td = self._get_features(td)

        return td

    def _get_features(self, td):
        """
        После каждого:
        - назначения операции или
        - перехода времени
        среда пересчитывает какие операции готовы к запуску.
        """
        # after we have transitioned to a next time step, we determine which operations are ready
        td["is_ready"] = op_is_ready(td)
        # td["lbs"] = calc_lower_bound(td)

        return td

    @staticmethod
    def _check_step_complete(td, dones):
        """check whether there a feasible actions left to be taken during the current
        time step. If this is not the case (and the instance is not done),
        we need to adance the timer of the repsective instance
        """
        """
        Step complete означает:
        «На текущем времени t больше нет ни одного допустимого действия,
        пора двигать время вперёд»
        
        td["action_mask"] - напоминание о формате
        action_mask.shape = (bs, num_jobs * num_mas)
        
        reduce(td["action_mask"], "bs ... -> bs", "any")
        Для каждого элемента батча:
        any_action_feasible[b] = ANY(action_mask[b, :])
        Результат:
        shape = (bs,)
        True -> есть хотя бы одно допустимое действие
        False -> нет ни одного допустимого действия
        
        & ~dones
        Потому что:
        если done=True, ничего двигать уже не надо
        эпизод закончен
        """
        return ~reduce(td["action_mask"], "bs ... -> bs", "any") & ~dones

    def _make_step(self, td: TensorDict) -> TensorDict:
        """
        Environment transition function
        """

        # td.size(0) = batch size
        # batch_idx = [0, 1, 2, ...]
        batch_idx = torch.arange(td.size(0))

        # 3*(#req_op)
        selected_job, selected_op, selected_machine = self._translate_action(td)

        # mark job as being processed
        # Job переходит в состояние "в процессе"
        td["job_in_process"][batch_idx, selected_job] = 1

        # mark op as schedules
        # Отмечаем операцию как запланированную
        td["op_scheduled"][batch_idx, selected_op] = True

        # update machine state
        # Время обработки выбранной операции
        proc_time_of_action = td["proc_times"][batch_idx, selected_machine, selected_op]
        # we may not select a machine that is busy
        # Проверка: машина должна быть свободна
        # Если машина занята — это баг в action_mask или policy.
        assert torch.all(td["busy_until"][batch_idx, selected_machine] <= td["time"])

        # update schedule
        # Обновление расписания
        td["start_times"][batch_idx, selected_op] = td["time"]
        td["finish_times"][batch_idx, selected_op] = td["time"] + proc_time_of_action
        td["ma_assignment"][batch_idx, selected_machine, selected_op] = 1
        # update the state of the selected machine
        # Машина становится занятой
        td["busy_until"][batch_idx, selected_machine] = td["time"] + proc_time_of_action
        # update adjacency matrices (remove edges)
        # Обновление графа допустимости: убираем операцию из всех машин
        # Что это делает: для всех машин proc_times[:, :, selected_op] = 0
        # Это означает, что операция больше не может быть назначена снова
        td["proc_times"] = td["proc_times"].scatter(
            2,
            selected_op[:, None, None].expand(-1, self.num_mas, 1),
            torch.zeros_like(td["proc_times"]),
        )
        # Общий смысл scatter
        # out = input.scatter(dim, index, src)
        # Означает:
        # в input по измерению dim
        # в позиции index
        # записать значения из src
        # !!! Главное правило: index.shape == src.shape
        # input может быть больше index/src
        # но по всем измерениям, кроме dim, размеры должны совпадать или быть broadcastable
        #
        # Простейший пример:
        # x = torch.tensor([[10, 20, 30]])
        # idx = torch.tensor([[1]])
        # src = torch.tensor([[0]])
        # x.scatter(1, idx, src) -> [[10, 0, 30]]
        # ------------------------------------
        # мы записываем значения по измерению операций:
        # dim=0 → batch
        # dim=1 → machine
        # dim=2 → operation  ← здесь
        # ------------------------------------
        # selected_op[:, None, None]
        # selected_op.shape == (bs,) -> (bs, 1, 1)
        # ------------------------------------
        # .expand(-1, self.num_mas, 1) - это метод тензора, который позволяет изменить его shape,
        # размножая данные вдоль измерений размера 1, не копируя данные в памяти. Это создает новый
        # view тензора, что делает операцию мгновенной и крайне эффективной по памяти.
        # -1 -> не меняем размерность batch
        # num_mas -> копируем для каждой машины вторую размерность
        # 1 → один индекс операции (не меняем третью размерность)
        # Итоговая форма: (bs, num_mas, 1)
        # ------------------------------------
        # zeros_like - создает тензор с такой же формой, как у входного.
        # td["proc_times"].shape = (bs, num_mas, n_ops_max)

        # Обновляем матрицу допустимости (bs, ma, job):
        # 1 если операция может выполняться на машине, 0 иначе
        td["ops_ma_adj"] = td["proc_times"].contiguous().gt(0).to(torch.float32)
        # Пересчитываем число допустимых машин
        td["num_eligible"] = torch.sum(td["ops_ma_adj"], dim=1)

        # update the positions of an operation in the job (subtract 1 from each operation of the selected job)
        # Обновление порядка операций в job
        td["ops_sequence_order"] = (
            td["ops_sequence_order"] - gather_by_index(td["job_ops_adj"], selected_job, 1)
        ).clip(0)
        # Что делает gather_by_index?
        # gather_by_index(td["job_ops_adj"], selected_job, dim=1)
        # Вход
        # src -> (bs, num_jobs, num_ops)
        # idx -> (bs,)
        # dim=1 -> выбираем job
        # После gather результат: (bs, num_ops)
        #
        # Это маска операций выбранной job:
        # [0 0 1 1 1 0 ... ]
        # (1 - операции выбранной job)
        #
        # До вычитания:
        # ops_sequence_order =     [0 1 0 1 2 0 ... ]
        # Выбрана job с job_mask = [0 0 1 1 1 0 ... ]
        # После вычитания:         [0 1 -1 0 1 0 ... ]
        # .clip(0)
        # обнуляет отрицательные значения: [0 1 0 0 1 0 ... ]
        #
        # Почему “бывшая -1” операция не будет выбрана снова?
        # Эта операция:
        # - уже помечена как выполненная
        # - уже удалена из proc_times
        # - уже стоит в op_scheduled = True
        # То есть она НЕ становится снова доступной

        # some checks
        # assert torch.allclose(
        #     td["proc_times"].sum(1).gt(0).sum(1),  # num ops with eligible machine
        #     (~(td["op_scheduled"] + td["pad_mask"])).sum(1),  # num unscheduled ops
        # )

        return td

    def _transit_to_next_time(self, step_complete, td: TensorDict) -> TensorDict:
        """
        Transit to the next time
        """
        """
        Эта функция перематывает время вперёд, когда в текущий момент нельзя сделать ни одного валидного действия
        То есть:
        - агент больше не может ничего назначить
        - среда должна сама продвинуться до следующего события
        
        Функция вызывается, если step_complete == True
        Это означает:
        - action_mask пуст
        - или выбран NO_OP
        - или все операции в процессе
        - или все машины заняты
        """

        # we need a transition to a next time step if either
        # 1.) all machines are busy
        # 2.) all operations are already currently in process (can only happen if num_jobs < num_machines)
        # 3.) idle machines can not process any of the not yet scheduled operations
        # 4.) no_op is choosen
        available_time_ma = td["busy_until"]
        end_op_per_job = td["end_op_per_job"]
        # we want to transition to the next time step where a machine becomes idle again. This time step must be
        # in the future, therefore we mask all machine idle times lying in the past / present
        # Вычисление следующего времени
        # available_time_ma > td["time"][:, None]
        #     берём только те машины,
        #     которые освободятся строго в будущем
        # иначе -> ∞, чтобы .min() их игнорировал
        # .min(1).values - ближайшее событие
        # Итог: available_time.shape == (bs,)
        available_time = (
            torch.where(available_time_ma > td["time"][:, None], available_time_ma, torch.inf)
            .min(1).values
        )

        # Проверка корректности:
        # Если ∞, значит нет будущих событий
        # но step_complete=True -> логическая ошибка
        assert not torch.any(available_time[step_complete].isinf())
        # step_complete = False -> в текущий момент времени есть валидные действия
        # То есть:
        #     есть свободная машина
        #     есть готовая операция
        #     агент может назначить (job, machine)
        # Время НЕ должно меняться
        #
        # step_complete = True -> среда обязана сама продвинуть время
        td["time"] = torch.where(step_complete, available_time, td["time"])

        # this may only be set when the operation is finished, not when it is scheduled
        # operation of job is finished, set next operation and flag job as being idle
        # Проверка завершения операций
        # td["next_op"] — индекс текущей операции job
        # gather вытаскивает finish_time именно этой операции
        curr_ops_end = td["finish_times"].gather(1, td["next_op"])
        # Истина если job выполнялась и её операция завершилась к текущему времени
        op_finished = td["job_in_process"] & (curr_ops_end <= td["time"][:, None])
        # check whether a job is finished, which is the case when the last operation of the job is finished
        # Определяем, завершилась ли job целиком
        # Job завершена если операция завершилась и это была последняя операция
        job_finished = op_finished & (td["next_op"] == end_op_per_job)
        # determine the next operation for a job that is not done, but whose latest operation is finished
        # если операция закончилась и job не последняя -> переходим к следующей операции
        td["next_op"] = torch.where(
            op_finished & ~job_finished, td["next_op"] + 1, td["next_op"],
        )
        # Job становится idle: может быть снова выбрана для планирования
        td["job_in_process"][op_finished] = False

        # Обновление статусов завершения
        # job_done — bool, но сложение работает как OR (False + True = True)
        td["job_done"] = td["job_done"] + job_finished
        # эпизод завершён, когда все jobs завершены
        td["done"] = td["job_done"].all(1, keepdim=True)

        # Интуитивная схема
        # [time t]
        #    |
        #    |  агент не может действовать
        #    v
        # [min busy_until > t]
        #    |
        #    v
        # [time t']
        #    |
        #    | операции закончились?
        #    v
        # [обновление jobs]
        return td, td["done"].squeeze(1)

    def _get_reward(self, td, actions=None) -> TensorDict:
        # stepwise reward — награда на каждом шаге
        # terminal reward — награда только в конце эпизода
        # actions is None -> значит _get_reward вызван после шага среды
        # td["reward"] уже был посчитан внутри step()
        if self.stepwise_reward and actions is None:
            return td["reward"]
        else:
            # td["done"].all() -> все инстансы завершены
            assert td["done"].all(), (
                "Set stepwise_reward to True if you want reward prior to completion"
            )
            # td["finish_times"].shape == (bs, n_ops_max) - время завершения каждой операции
            # .masked_fill(...)
            # там, где pad_mask == True -> значение -∞
            # Зачем: padding операции не должны влиять на makespan
            # max() их полностью игнорирует
            # Фактически код реализует вычисления makespan: reward = -makespan
            return -td["finish_times"].masked_fill(td["pad_mask"], -torch.inf).max(1).values

    def _make_spec(self, generator: FJSPGenerator):
        """
        Создаёт:
        observation_spec
        action_spec
        reward_spec
        done_spec
        То есть полное описание MDP.

        Unbounded - тензор без числовых ограничений.
        Формально значения могут быть любыми но форма (shape) и тип (dtype) фиксированы
        Bounded - тензор со строгими границами: low <= x <= high

        Composite = словарь specs
        То есть:
        {
          key1: Spec,
          key2: Spec,
          ...
        }
        Он описывает структуру TensorDict.
        """
        self.observation_spec = Composite(
            time=Unbounded(  # текущее время среды
                shape=(1,),
                dtype=torch.int64,
            ),
            next_op=Unbounded(  # индекс следующей операции
                shape=(self.num_jobs,),
                dtype=torch.int64,
            ),
            proc_times=Unbounded(  # матрица времени обработки
                shape=(self.num_mas, self.n_ops_max),
                dtype=torch.float32,
            ),
            pad_mask=Unbounded(  # True -> padding операция, False -> реальная операция
                shape=(self.num_mas, self.n_ops_max),
                dtype=torch.bool,
            ),
            start_op_per_job=Unbounded(
                shape=(self.num_jobs,),
                # !!!!!!!!! Это странное место: по смыслу это индексы операций - должны быть int64, но указано bool.
                dtype=torch.bool,
            ),
            end_op_per_job=Unbounded(
                shape=(self.num_jobs,),
                # !!!!!!!!! Это странное место: по смыслу это индексы операций - должны быть int64, но указано bool.
                dtype=torch.bool,
            ),
            start_times=Unbounded(  # расписание старта операций, INIT_FINISH для незапланированных
                shape=(self.n_ops_max,),
                dtype=torch.int64,
            ),
            finish_times=Unbounded(  # расписание конца операций, INIT_FINISH для незапланированных
                shape=(self.n_ops_max,),
                dtype=torch.int64,
            ),
            job_ops_adj=Unbounded(  # матрица инцидентности job–operation: 1, если операция принадлежит job
                shape=(self.num_jobs, self.n_ops_max),
                dtype=torch.int64,
            ),
            ops_job_map=Unbounded(  # Для каждой операции: id job
                shape=(self.n_ops_max),
                dtype=torch.int64,
            ),
            ops_sequence_order=Unbounded(  # номер операции внутри job
                shape=(self.n_ops_max),
                dtype=torch.int64,
            ),
            ma_assignment=Unbounded(  # бинарная матрица, какая операция назначена на какую машину
                shape=(self.num_mas, self.n_ops_max),
                dtype=torch.int64,
            ),
            busy_until=Unbounded(  # состояние машин, когда каждая освободится
                shape=(self.num_mas,),
                dtype=torch.int64,
            ),
            num_eligible=Unbounded(  # сколько машин может выполнить операцию
                shape=(self.n_ops_max,),
                dtype=torch.int64,
            ),
            job_in_process=Unbounded(  # сейчас выполняется операция этой job?
                shape=(self.num_jobs,),
                dtype=torch.bool,
            ),
            job_done=Unbounded(  # job полностью завершена?
                shape=(self.num_jobs,),
                dtype=torch.bool,
            ),
            shape=(),
        )
        self.action_spec = Bounded(
            # action ∈ {-1, 0, 1, 2, ..., num_jobs * num_mas - 1}
            # -1 -> NO_OP
            shape=(1,),
            dtype=torch.int64,
            low=-1,
            high=self.n_ops_max,
        )
        self.reward_spec = Unbounded(shape=(1,))  # reward скалярный
        self.done_spec = Unbounded(shape=(1,), dtype=torch.bool)  # стандартный флаг окончания эпизода

    @staticmethod
    def render(td, idx):
        return render(td, idx)

    def select_start_nodes(self, td: TensorDict, num_starts: int):
        return sample_n_random_actions(td, num_starts)

    def get_num_starts(self, td):
        # NOTE in the paper they use N_s = 100
        return 100
