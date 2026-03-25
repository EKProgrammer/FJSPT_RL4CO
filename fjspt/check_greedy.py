import torch

from env import FJSPTEnv

from greedy_policies import GreedyPolicy

from rl4co.envs import ENV_REGISTRY

ENV_REGISTRY["fjspt"] = FJSPTEnv

env = FJSPTEnv(
    generator_params={
      "num_jobs": 7,  # the total number of jobs
      "num_machines": 8,  # the total number of machines that can process operations
      "num_trucks": 2,  # the total number of trucks
      "min_ops_per_job": 2,  # minimum number of operatios per job
      "max_ops_per_job": 5,  # maximum number of operations per job
      "min_processing_time": 15,  # the minimum time required for a machine to process an operation
      "max_processing_time": 40,  # the maximum time required for a machine to process an operation
      "min_transportation_time": 0,  # the minimum time required for a truck to transport
      "max_transportation_time": 12,  # the maximum time required for a truck to transport
      "min_eligible_ma_per_op": 2,  # the minimum number of machines capable to process an operation
      "max_eligible_ma_per_op": 2,  # the maximum number of machines capable to process an operation
    },
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy = GreedyPolicy(policy_type="MWKR")
policy = policy.to(device)

policy.eval()
td = env.reset(batch_size=[1]).to(device)

with torch.no_grad():
    out = policy(td)
print(out)
