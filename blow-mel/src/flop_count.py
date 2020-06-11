from flopth import flopth
import torch.nn as nn
import torch

from utils import utils


class TwoLinear(nn.Module):
    def __init__(self):
        super(TwoLinear, self).__init__()

        self.l1 = nn.Linear(10, 1994)
        self.l2 = nn.Linear(1994, 10)

    def forward(self, x, y):
        x = self.l1(x) * y
        x = self.l2(x) * y

        return x


_, _, blow_model, _ = utils.load_stuff('/rscratch/xuezhenruo/blow_vctk/blow_200331_test/ckpt_60000', 'cuda')
blow_model.use_coeff = False
print(blow_model)

sum_flops = flopth(blow_model, in_size=[80, 64], extra_params=torch.LongTensor([1]).cuda())
print(sum_flops)
