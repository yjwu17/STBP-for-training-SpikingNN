import torch, time, os
import torch.nn as nn
import numpy as np
import random
thresh = 0.3
lens = 0.5
decay = 0.5
num_classes = 10
batch_size = 50
num_epochs = 101
learning_rate = 5e-4
time_window = 8

device = torch.device("cpu")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class ActFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(0.).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input) < lens
        return grad_input * temp.float()

cfg_fc = [512, 10]
probs = 0.5
act_fun = ActFun.apply

def lr_scheduler(optimizer, epoch, init_lr=0.1, lr_decay_epoch=100):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    if epoch % lr_decay_epoch == 0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.5
    return optimizer

class SNN_Model(nn.Module):
    def __init__(self, p=0.5):
        super(SNN_Model, self).__init__()
        self.fc1 = nn.Linear(784, cfg_fc[0], )
        self.fc2 = nn.Linear(cfg_fc[0], cfg_fc[1], )

    def forward(self, input,  wins=time_window):
        batch_size = input.size(0)
        h1_mem = h1_spike = h1_sumspike = torch.zeros(batch_size, cfg_fc[0], device=device)
        h2_mem = h2_spike = h2_sumspike = torch.zeros(batch_size, cfg_fc[1], device=device)
        for step in range(wins):
            x = input.view(batch_size, -1).float()
            h1_spike, h1_mem  = mem_update(self.fc1, x, h1_spike, h1_mem)
            h2_spike, h2_mem  = mem_update(self.fc2, h1_spike, h2_spike, h2_mem)
            h2_sumspike = h2_sumspike + h2_spike
        outs = h2_sumspike / wins
        return outs, h2_sumspike

def mem_update(fc, inputs, spike, mem):
    state = fc(inputs)
    mem = mem * (1 - spike) * decay + state
    now_spike = act_fun(mem - thresh)
    return now_spike.float(), mem


