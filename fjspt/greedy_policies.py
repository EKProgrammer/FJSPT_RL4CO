from typing import Literal
from tensordict import TensorDict

from torch import nn

from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class GreedyPolicy(nn.Module):
    def __init__(self, policy_type: Literal["FIFO", "MOPNR", "SPT", "MWKR"] = "SPT"):
        policy_inst = {"FIFO": FIFO, "MOPNR": MOPNR, "SPT": SPT, "MWKR": MWKR}
        self.policy = policy_inst[policy_type]()


class FIFO:
    def forward(self, td: TensorDict):
        mask = td["action_mask"]
        return logits, mask


class MOPNR:
    def forward(self, td: TensorDict):
        pass


class SPT:
    def forward(self, td: TensorDict):
        pass


class MWKR:
    def forward(self, td: TensorDict):
        pass
