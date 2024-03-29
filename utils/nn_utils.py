import torch
import torch.nn as nn
torch.set_default_dtype(torch.float64)



class Clip(nn.Module):
    def __init__(self, min=0, max=1):
        super(Clip, self).__init__()
        self.min = min
        self.max = max

    def forward(self, x):
        return torch.clamp(x, self.min, self.max)

act_list = {'relu': nn.ReLU(), 'silu': nn.SiLU(), 'softplus': nn.Softplus(),
            'sigmoid': nn.Sigmoid(), 'softmax': nn.Softmax(dim=-1), 'clip': Clip()}


###################################################################
# NN: FC-MLP, ResNet
###################################################################
class FCNet(nn.Module):
    def __init__(self, nin, nout, nh, nl, act='sigmoid'):
        super().__init__()
        net = [nn.Linear(nin, nh), nn.ReLU()]
        for _ in range(nl):
            net += [nn.Linear(nh, nh), nn.ReLU()]
        net.append(nn.Linear(nh, nout))
        if act is not None:
            net.append(act_list[act])
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class ResNet(nn.Module):
    def __init__(self, nin, nout, nh, nl, act='sigmoid'):
        super().__init__()
        net = [nn.Linear(nin, nh)]
        for _ in range(nl):
            # net.append(ResBlock(nh, nh//2))
            net += [ResBlock(nh, min(nh//2, 1024))]
        net.append(nn.Linear(nh, nout))
        if act is not None:
            net.append(act_list[act])
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class ResBlock(nn.Module):
    def __init__(self, n_in, n_hid):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, n_hid),
                                 nn.ReLU(),
                                 nn.Linear(n_hid, n_in))
    def forward(self, x):
        return x + self.net(x)


   
