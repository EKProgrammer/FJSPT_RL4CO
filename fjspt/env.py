import torch

from einops import rearrange, reduce
from tensordict.tensordict import TensorDict
from torchrl.data import Bounded, Composite, Unbounded

from rl4co.envs.common.base import RL4COEnvBase as EnvBase
from rl4co.utils.ops import gather_by_index, sample_n_random_actions

from . import INIT_FINISH, NO_OP_ID
from .generator import FJSPTGenerator
from .render import render
from .utils import calc_lower_bound, get_job_ops_mapping, op_is_ready


class FJSPTEnv(EnvBase):
    name = "fjspt"

    def __init__(
        self,
        generator: FJSPTGenerator = None,
        generator_params: dict = {},
        mask_no_ops: bool = True,
        check_mask: bool = False,
        stepwise_reward: bool = False,
        **kwargs,
    ):
        super().__init__(check_solution=False, **kwargs)
        if generator is None:
            # генерируем случайные инстансы
            generator = FJSPTGenerator(**generator_params)
        self.generator = generator
        self._num_mas = generator.num_mas
        self._num_jobs = generator.num_jobs
        self._num_tracks = generator.num_tracks
        self._n_ops_max = generator.max_ops_per_job * self.num_jobs

        self.mask_no_ops = mask_no_ops
        self.check_mask = check_mask
        self.stepwise_reward = stepwise_reward
        self._make_spec(self.generator)

    @property
    def num_mas(self):
        return self._num_mas

    @property
    def num_jobs(self):
        return self._num_jobs

    def num_tracks(self):
        return self._num_tracks

    @property
    def n_ops_max(self):
        return self._n_ops_max

    def set_instance_params(self, td):
        self._num_jobs = td["start_op_per_job"].size(1)
        self._num_mas = td["proc_times"].size(1)
        self._n_ops_max = td["proc_times"].size(2)

    def _decode_graph_structure(self, td: TensorDict):
        batch_size = td.batch_size
        start_op_per_job = td["start_op_per_job"]
        end_op_per_job = td["end_op_per_job"]
        pad_mask = td["pad_mask"]
        n_ops_max = td["pad_mask"].size(-1)

        # here we will generate the operations-job mapping:
        ops_job_map, ops_job_bin_map = get_job_ops_mapping(
            start_op_per_job, end_op_per_job, n_ops_max
        )

        # mask invalid edges (caused by padding)
        ops_job_bin_map[pad_mask.unsqueeze(1).expand_as(ops_job_bin_map)] = 0

        # generate for each batch a sequence specifying the position of all operations in their respective jobs,
        # e.g. [0,1,0,0,1,2,0,1,2,3,0,0] for jops with n_ops=[2,1,3,4,1,1]
        # (bs, max_ops)
        ops_seq_order = torch.sum(ops_job_bin_map * (ops_job_bin_map.cumsum(2) - 1), dim=1)

        # predecessor and successor adjacency matrices
        pred = torch.diag_embed(torch.ones(n_ops_max - 1), offset=-1)[None].expand(
            *batch_size, -1, -1
        )
        # the start of the sequence (of each job) does not have a predecessor, therefore we can
        # mask all first ops of a job in the predecessor matrix
        pred = pred * ops_seq_order.gt(0).unsqueeze(-1).expand_as(pred).to(pred)
        succ = torch.diag_embed(torch.ones(n_ops_max - 1), offset=1)[None].expand(
            *batch_size, -1, -1
        )
        # apply the same logic as above to mask the last op of a job, which does not have a successor. The last job of a job
        # always comes before the 1st op of the next job, therefore performing a left shift of the ops seq tensor here
        succ = succ * torch.cat(
            (ops_seq_order[:, 1:], ops_seq_order.new_full((*batch_size, 1), 0)), dim=1
        ).gt(0).to(succ).unsqueeze(-1).expand_as(succ)

        # adjacency matrix = predecessors, successors and self loops
        # (bs, max_ops, max_ops, 2)
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
        machine_start_times = torch.zeros((*batch_size, n_ops_max))
        machine_finish_times = torch.full((*batch_size, n_ops_max), INIT_FINISH)
        track_start_times = torch.zeros((*batch_size, self.num_tracks))
        track_finish_times = torch.full((*batch_size, self.num_tracks), INIT_FINISH)
        ma_assignment = torch.zeros((*batch_size, self.num_mas, n_ops_max))

        # reset feature space
        machine_busy_until = torch.zeros((*batch_size, self.num_mas))
        truck_busy_until = torch.zeros((*batch_size, self.num_tracks))
        # (bs, ma, ops)
        ops_ma_adj = (td_reset["proc_times"] > 0).to(torch.float32)
        # (bs, ops)
        num_eligible = torch.sum(ops_ma_adj, dim=1)

        td["job_location"][batch_idx, selected_job] = selected_machine
        td["track_location"][batch_idx, selected_track] = selected_machine

        td_reset = td_reset.update(
            {
                "machine_start_times": machine_start_times,
                "machine_finish_times": machine_finish_times,
                "track_start_times": track_start_times,
                "track_finish_times": track_finish_times,
                "ma_assignment": ma_assignment,
                "machine_busy_until": machine_busy_until,
                "truck_busy_until": truck_busy_until,
                "num_eligible": num_eligible,
                "next_op": start_op_per_job.clone().to(torch.int64),
                "ops_ma_adj": ops_ma_adj,
                "op_scheduled": torch.full((*batch_size, n_ops_max), False),
                "job_in_process": torch.full((*batch_size, self.num_jobs), False),
                "reward": torch.zeros((*batch_size,), dtype=torch.float32),
                "time": torch.zeros((*batch_size,)),
                "job_done": torch.full((*batch_size, self.num_jobs), False),
                "done": torch.full((*batch_size, 1), False),
                # перевозит ли данный грузовик какую-то деталь
                "track_in_process": torch.full((*batch_size, self.num_tracks), False),
            },
        )

        td_reset.set("machine_action_mask", self.get_machine_action_mask(td_reset))
        td_reset.set("transportation_action_mask", self.get_transportation_action_mask(td_reset))
        # add additional features to tensordict
        td_reset["lbs"] = calc_lower_bound(td_reset)
        td_reset = self._get_features(td_reset)

        return td_reset

    def _get_job_machine_availability(self, td: TensorDict):
        batch_size = td.size(0)

        # (bs, jobs, machines)
        machine_action_mask = torch.full((batch_size, self.num_jobs, self.num_mas), False).to(td.device)

        # mask jobs that are done already
        machine_action_mask.add_(td["job_done"].unsqueeze(2))
        # as well as jobs that are currently processed
        machine_action_mask.add_(td["job_in_process"].unsqueeze(2))

        # mask machines that are currently busy
        machine_action_mask.add_(td["machine_busy_until"].gt(td["time"].unsqueeze(1)).unsqueeze(1))

        # exclude job-machine combinations, where the machine cannot process the next machine op of the job
        next_ops_proc_times = gather_by_index(
            td["proc_times"], td["next_op"].unsqueeze(1), dim=2, squeeze=False
        ).transpose(1, 2)
        machine_action_mask.add_(next_ops_proc_times == 0)
        return machine_action_mask

    def get_machine_action_mask(self, td: TensorDict) -> torch.Tensor:
        # 1 indicates machine or job is unavailable at current time step
        machine_action_mask = self._get_job_machine_availability(td)
        if self.mask_no_ops:
            # masking is only allowed if instance is finished
            no_op_mask = td["done"]
        else:
            # if no job is currently processed and instance is not finished yet, waiting is not allowed
            no_op_mask = (td["job_in_process"].any(1, keepdims=True) & (~td["done"])) | td["done"]
        # flatten action mask to correspond with logit shape
        machine_action_mask = rearrange(machine_action_mask, "bs j m -> bs (j m)")
        # NOTE: 1 means feasible action, 0 means infeasible action
        mask = torch.cat((no_op_mask, ~machine_action_mask), dim=1)

        return mask

    def _get_job_trucks_availability(self, td: TensorDict):
        pass

    def get_transportation_action_mask(self, td: TensorDict) -> torch.Tensor:
        transportation_action_mask = self._get_job_trucks_availability(td)
        pass

    def _translate_action(self, td):
        """Разбивает одно число action на три составляющих."""
        action = td["action"]

        # Размеры для деления
        n_m = self.num_mas
        n_t = self.num_tracks

        # Извлекаем индексы (обратный порядок умножения)
        selected_machine = action % n_m
        remaining = action // n_m
        selected_track = remaining % n_t
        selected_job = remaining // n_t

        selected_op = td["next_op"].gather(1, selected_job[:, None]).squeeze(1)
        return selected_job, selected_op, selected_track, selected_machine

    def _step(self, td: TensorDict):
        # cloning required to avoid inplace operation which avoids gradient backtracking
        td = td.clone()
        td["action"].subtract_(1)
        # (bs)
        dones = td["done"].squeeze(1)
        # specify which batch instances require which operation
        no_op = td["action"].eq(NO_OP_ID)
        no_op = no_op & ~dones
        req_op = ~no_op & ~dones

        # transition to next time for no op instances
        if no_op.any():
            td, dones = self._transit_to_next_time(no_op, td)

        # select only instances that perform a scheduling action
        td_op = td.masked_select(req_op)

        td_op = self._make_step(td_op)
        # update the tensordict
        td[req_op] = td_op

        # machine action mask
        td.set("machine_action_mask", self.get_machine_action_mask(td))

        step_complete = self._check_step_complete(td, dones)
        while step_complete.any():
            td, dones = self._transit_to_next_time(step_complete, td)
            td.set("machine_action_mask", self.get_machine_action_mask(td))
            step_complete = self._check_step_complete(td, dones)
        if self.check_mask:
            assert reduce(td["machine_action_mask"], "bs ... -> bs", "any").all()

        if self.stepwise_reward:
            # if we require a stepwise reward, the change in the calculated lower bounds could serve as such
            lbs = calc_lower_bound(td)
            td["reward"] = -(lbs.max(1).values - td["lbs"].max(1).values)
            td["lbs"] = lbs
        else:
            td["lbs"] = calc_lower_bound(td)

        # add additional features to tensordict
        td = self._get_features(td)

        return td

    def _get_features(self, td):
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
        return ~reduce(td["machine_action_mask"], "bs ... -> bs", "any") & ~dones

    def _make_step(self, td: TensorDict) -> TensorDict:
        """
        Environment transition function
        """

        batch_idx = torch.arange(td.size(0))

        # 3*(#req_op)
        selected_job, selected_op, selected_track, selected_machine = self._translate_action(td)

        # 1. Считаем время транспортировки
        # Откуда забираем деталь
        job_loc = td["job_location"][batch_idx, selected_job]
        # Где сейчас грузовик
        track_loc = td["track_location"][batch_idx, selected_track]

        # Время доехать пустым до детали + довезти деталь до станка
        dist_empty = td["tracks_times"][batch_idx, track_loc, job_loc]
        dist_loaded = td["tracks_times"][batch_idx, job_loc, selected_machine]

        # Грузовик может начать движение только когда он свободен
        # Деталь можно забрать, когда она готова (закончена прошлая операция)
        track_ready_to_pick = td["track_busy_until"][batch_idx, selected_track],

        arrival_at_job = track_ready_to_pick + dist_empty
        arrival_at_machine = arrival_at_job + dist_loaded

        # 2. Считаем время обработки
        # Станок начинает работу, когда приехала деталь И станок освободился
        proc_start = torch.max(arrival_at_machine, td["machine_busy_until"][batch_idx, selected_machine])
        proc_time = td["proc_times"][batch_idx, selected_machine, selected_op]
        proc_end = proc_start + proc_time

        # mark job as being processed
        td["job_in_process"][batch_idx, selected_job] = 1

        # mark op as schedules
        td["op_scheduled"][batch_idx, selected_op] = True

        # update machine state
        proc_time_of_action = td["proc_times"][batch_idx, selected_machine, selected_op]
        # we may not select a machine that is busy
        assert torch.all(td["machine_busy_until"][batch_idx, selected_machine] <= td["time"])

        # update schedule
        td["machine_start_times"][batch_idx, selected_op] = proc_start
        td["machine_finish_times"][batch_idx, selected_op] = proc_end  # !!! selected_op
        td["ma_assignment"][batch_idx, selected_machine, selected_op] = 1

        # td["track_start_times"][batch_idx, selected_track] =
        # td["track_finish_times"][batch_idx, selected_track] =
        # Нужны ли вообще machine_start_times, machine_finish_times, track_start_times, track_finish_times?

        # update the state of the selected machine
        td["machine_busy_until"][batch_idx, selected_machine] = arrival_at_machine  # !!! selected_machine
        td["machine_busy_until"][batch_idx, selected_machine] = proc_end

        # Обновляем локации для следующих шагов
        td["job_location"][batch_idx, selected_job] = selected_machine
        td["track_location"][batch_idx, selected_track] = selected_machine

        # update adjacency matrices (remove edges)
        td["proc_times"] = td["proc_times"].scatter(
            2,
            selected_op[:, None, None].expand(-1, self.num_mas, 1),
            torch.zeros_like(td["proc_times"]),
        )
        td["ops_ma_adj"] = td["proc_times"].contiguous().gt(0).to(torch.float32)
        td["num_eligible"] = torch.sum(td["ops_ma_adj"], dim=1)
        # update the positions of an operation in the job (subtract 1 from each operation of the selected job)
        td["ops_sequence_order"] = (
            td["ops_sequence_order"] - gather_by_index(td["job_ops_adj"], selected_job, 1)
        ).clip(0)
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

        # we need a transition to a next time step if either
        # 1) all machines are busy
        # 2) all operations are already currently in process (can only happen if num_jobs < num_machines)
        # 3) idle machines can not process any of the not yet scheduled operations
        # 4) no_op is choosen

        # Время освобождения станков
        ma_free_time = torch.where(td["machine_busy_until"] > td["time"][:, None],
                                   td["machine_busy_until"], torch.inf)
        # Время освобождения грузовиков
        track_free_time = torch.where(td["truck_busy_until"] > td["time"][:, None],
                                      td["truck_busy_until"], torch.inf)
        # Находим глобальный минимум
        min_ma = ma_free_time.min(1).values
        min_track = track_free_time.min(1).values
        available_time = torch.minimum(min_ma, min_track)

        assert not torch.any(available_time[step_complete].isinf())
        td["time"] = torch.where(step_complete, available_time, td["time"])

        curr_ops_end = td["machine_finish_times"].gather(1, td["next_op"])
        op_finished = td["job_in_process"] & (curr_ops_end <= td["time"][:, None])

        job_finished = op_finished & (td["next_op"] == td["end_op_per_job"])

        td["next_op"] = torch.where(op_finished & ~job_finished, td["next_op"] + 1, td["next_op"])
        td["job_in_process"][op_finished] = False

        td["job_done"] = td["job_done"] + job_finished
        td["done"] = td["job_done"].all(1, keepdim=True)

        return td, td["done"].squeeze(1)

    def _get_reward(self, td, actions=None) -> TensorDict:
        if self.stepwise_reward and actions is None:
            return td["reward"]
        else:
            assert td["done"].all(), (
                "Set stepwise_reward to True if you want reward prior to completion"
            )
            # нет смысла здесь учитывать транспортировку, т.к. последниее действие всегда производственное
            return -td["machine_finish_times"].masked_fill(td["pad_mask"], -torch.inf).max(1).values

    def _make_spec(self, generator: FJSPTGenerator):
        self.observation_spec = Composite(
            time=Unbounded(
                shape=(1,),
                dtype=torch.int64,
            ),
            next_op=Unbounded(
                shape=(self.num_jobs,),
                dtype=torch.int64,
            ),
            proc_times=Unbounded(
                shape=(self.num_mas, self.n_ops_max),
                dtype=torch.float32,
            ),
            pad_mask=Unbounded(
                shape=(self.num_mas, self.n_ops_max),
                dtype=torch.bool,
            ),
            start_op_per_job=Unbounded(
                shape=(self.num_jobs,),
                dtype=torch.bool,
            ),
            end_op_per_job=Unbounded(
                shape=(self.num_jobs,),
                dtype=torch.bool,
            ),
            machine_start_times=Unbounded(
                shape=(self.n_ops_max,),
                dtype=torch.int64,
            ),
            machine_finish_times=Unbounded(
                shape=(self.n_ops_max,),
                dtype=torch.int64,
            ),
            job_ops_adj=Unbounded(
                shape=(self.num_jobs, self.n_ops_max),
                dtype=torch.int64,
            ),
            ops_job_map=Unbounded(
                shape=(self.n_ops_max),
                dtype=torch.int64,
            ),
            ops_sequence_order=Unbounded(
                shape=(self.n_ops_max),
                dtype=torch.int64,
            ),
            ma_assignment=Unbounded(
                shape=(self.num_mas, self.n_ops_max),
                dtype=torch.int64,
            ),
            machine_busy_until=Unbounded(
                shape=(self.num_mas,),
                dtype=torch.int64,
            ),
            truck_busy_until=Unbounded(
                shape=(self.num_tracks,),
                dtype=torch.int64,
            ),
            num_eligible=Unbounded(
                shape=(self.n_ops_max,),
                dtype=torch.int64,
            ),
            job_in_process=Unbounded(
                shape=(self.num_jobs,),
                dtype=torch.bool,
            ),
            job_done=Unbounded(
                shape=(self.num_jobs,),
                dtype=torch.bool,
            ),
            action_type=Unbounded(
                shape=(self.num_jobs,),
                dtype=torch.bool,
            ),
            track_in_process=Unbounded(
                shape=(self.num_tracks,),
                dtype=torch.bool,
            ),
            track_start_times=Unbounded(
                shape=(self.num_tracks,),
                dtype=torch.bool,
            ),
            track_finish_times=Unbounded(
                shape=(self.num_tracks,),
                dtype=torch.bool,
            ),
            shape=(),
        )
        self.action_spec = Bounded(
            shape=(1,),
            dtype=torch.int64,
            low=-1,
            high=self.n_ops_max,
        )
        self.reward_spec = Unbounded(shape=(1,))
        self.done_spec = Unbounded(shape=(1,), dtype=torch.bool)

    @staticmethod
    def render(td, idx):
        return render(td, idx)

    def select_start_nodes(self, td: TensorDict, num_starts: int):
        return sample_n_random_actions(td, num_starts)

    def get_num_starts(self, td):
        # NOTE in the paper they use N_s = 100
        return 100
