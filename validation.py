from constants import INIT_FINISH, NO_OP_ID
from tensordict import TensorDict
import torch


def validate_solution(td: TensorDict, idx: int = 0, verbose: bool = True):
    """
    Полная проверка корректности расписания.
    Бросает AssertionError при любой ошибке.
    """

    inst = td[idx]

    errors = []

    def err(msg):
        errors.append(msg)
        if verbose:
            print(msg)

    # Базовые данные
    n_jobs = inst["job_ops_adj"].size(0)
    n_machines = inst["ma_assignment"].size(0)

    machine_start = inst["machine_start_times"]
    machine_end = inst["machine_finish_times"]
    truck_start = inst["truck_start_times"]
    truck_end = inst["truck_finish_times"]
    tr_op_to_op = inst["tr_op_to_op"]
    op_to_tr_op = inst["op_to_tr_op"]

    ma_assignment = inst["ma_assignment"]
    job_ops_adj = inst["job_ops_adj"]

    # 1. Проверка assignment
    assigned_per_op = ma_assignment.sum(dim=0)
    real_ops = job_ops_adj.sum(dim=0)
    invalid = assigned_per_op != real_ops
    if invalid.any():
        bad_ops = invalid.nonzero(as_tuple=True)[0]
        for op in bad_ops.tolist():
            err(
                f"Op {op}: assigned {assigned_per_op[op].item()} "
                f"(expected {real_ops[op].item()})"
            )

    # 2. Проверка времени операций
    real_ops_mask = real_ops > 0
    machine_ops = real_ops_mask.nonzero(as_tuple=True)[0]
    for op in machine_ops:
        if machine_end[op] < machine_start[op]:
            err(f"Machine op {op}: end < start")

    n_tr_ops = inst["truck_operation"].item()
    for op in range(n_tr_ops):
        if truck_end[op] < truck_start[op]:
            err(f"Truck op {op}: end < start")

    # 3. Машины: no overlap
    for ma in range(n_machines):
        real_machine_ops = (ma_assignment[ma] == 1) & real_ops_mask
        ops = real_machine_ops.nonzero(as_tuple=True)[0]

        intervals = []
        for op in ops:
            intervals.append((op.item(),
                              machine_start[op].item(),
                              machine_end[op].item()))

        intervals.sort(key=lambda x: x[1])

        for i in range(len(intervals) - 1):
            a = intervals[i]
            b = intervals[i + 1]
            if a[2] > b[1]:
                err(f"Machine {ma} overlap: op {a[0]} and {b[0]}")

    # 4. Грузовики: no overlap
    truck_tr_ops = inst["truck_tr_ops"]

    truck_schedules = {}
    for i in range(n_tr_ops):
        tr = truck_tr_ops[i].item()
        truck_schedules.setdefault(tr, [])
        truck_schedules[tr].append((i,
                                    truck_start[i].item(),
                                    truck_end[i].item()))

    for tr, ops in truck_schedules.items():
        ops.sort(key=lambda x: x[1])
        for i in range(len(ops) - 1):
            a = ops[i]
            b = ops[i + 1]
            if a[2] > b[1]:
                err(f"Truck {tr} overlap between ops {a[0]} and {b[0]}")

    # 5. precedence (job order)
    for job in range(n_jobs):
        ops = (job_ops_adj[job] == 1).nonzero(as_tuple=True)[0]

        ops = sorted(ops.tolist(), key=lambda op: inst["ops_sequence_order"][op].item())

        for i in range(len(ops) - 1):
            op1 = ops[i]
            op2 = ops[i + 1]

            if machine_end[op1] > machine_start[op2]:
                err(f"Job {job} precedence violated: op {op1} -> op {op2}")

    # 6. job не делает 2 вещи сразу
    job_intervals = {j: [] for j in range(n_jobs)}

    # machine ops
    for op in machine_ops:
        job = job_ops_adj[:, op].nonzero().item()
        job_intervals[job].append((machine_start[op].item(),
                                   machine_end[op].item(),
                                   f"machine_op_{op}"))

    # truck ops
    job_tr_ops = inst["job_tr_ops"]
    for i in range(n_tr_ops):
        job = job_tr_ops[i].item()
        job_intervals[job].append((truck_start[i].item(),
                                   truck_end[i].item(),
                                   f"truck_op_{i}"))

    for job, intervals in job_intervals.items():
        intervals.sort(key=lambda x: (x[0], x[1]))
        for i in range(len(intervals) - 1):
            a = intervals[i]
            b = intervals[i + 1]
            if a[1] > b[0]:
                err(f"Job {job} overlap: {a[2]} and {b[2]}")

    # 7. transport -> processing
    # предполагаем 1 transport на op (по порядку)
    tr_ops = torch.arange(n_tr_ops)
    ops = tr_op_to_op[:n_tr_ops]
    tr_end = truck_end[:n_tr_ops]
    proc_start = machine_start[ops]
    viol = tr_end > proc_start
    if viol.any():
        bad = tr_ops[viol]
        err(f"Transport->processing violation for transport ops: {bad.tolist()}")

    # 8. Каждая операция имеет transport
    if (op_to_tr_op[real_ops_mask] == NO_OP_ID).any():
        err("Some operations have no transport")

    # 8. все операции завершены
    for op in machine_ops:
        if machine_end[op] >= INIT_FINISH:
            err(f"Op {op} not completed")

    # 9. количество операций совпадает
    total_ops = (job_ops_adj.sum()).item()
    if n_tr_ops != total_ops:
        err(f"Mismatch: {n_tr_ops} transport ops vs {total_ops} real ops")

    # 10. done consistency
    if inst["done"].item() != inst["job_done"].all().item():
        err("done != all(job_done)")

    # 11. finished jobs действительно завершены
    for job in range(n_jobs):
        if inst["job_done"][job]:
            ops = (job_ops_adj[job] == 1).nonzero(as_tuple=True)[0]
            if (machine_end[ops] >= INIT_FINISH).any():
                err(f"Job {job} marked done but has unfinished ops")

    # 12. done=True -> всё завершено
    if inst["done"]:
        if ((machine_end >= INIT_FINISH) & real_ops_mask).any():
            err("done=True but unfinished ops exist")

    # ФИНАЛ
    if len(errors) == 0:
        if verbose:
            print("VALID SOLUTION")
        return True
    else:
        raise AssertionError(f"Validation failed with {len(errors)} errors")
