import torch
import torch.nn as nn
from .flows.iresblock import iResidualLayer, iGraphResidualLayer
from .flows.coupling import LUInvertibleMM, ActNorm, CouplingLayer, MADE, Sigmoid
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
# GNN: Graph based NN, ResNet
###################################################################
class GraphLinear(nn.Module):
    def __init__(self, num_node, n_in, n_hid):
        super().__init__()
        self.num_node = num_node
        self.linear = nn.Linear(n_in, n_hid)

    def forward(self, x, adj=None):
        """
        x: batch * node * feature
        adj: num_node * num_node
        """
        adj = adj + torch.eye(self.num_node)
        adj = adj.view(1, self.num_node, self.num_node, 1)
        deg = self.adj.sum(2)

        emb = self.linear(x)
        emb = x.view(-1, 1, self.num_node, emb.shape[-1])
        emb = (adj * emb).sum(2) / deg
        return emb


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


###################################################################
# INN
###################################################################
class INN(nn.Module):
    """ A sequence of invertible layers """
    def __init__(self, nin, nhid, cin, nl, inv='made', outact='sigmoid', bilip=False, lip=2):
        super().__init__()
        mask = torch.zeros(size=[1, nin])
        mask[:, :nin // 2] = 1
        flows = []
        for _ in range(nl):
            flows += [ActNorm(nin), LUInvertibleMM(nin)]
            if inv == 'made':
                flows.append( MADE(nin, nhid, cin, bilip=bilip, lip=lip) )
            elif inv == 'coupling':
                flows.append( CouplingLayer(nin, nhid, mask, cin, bilip=bilip, lip=lip) )
            elif inv == 'residual':
                flows.append( iResidualLayer(nin, nhid, cin) )
        if outact == 'sigmoid':
            flows += [ActNorm(nin), Sigmoid()]
        self.flows = nn.ModuleList(flows)
        self.con_emb = None


    def forward(self, x, c):
        b = x.shape[0]
        if self.training:
            log_det = torch.zeros(b).to(x.device)
            log_dis = torch.zeros(b).to(x.device)
            if self.con_emb is not None:
                c = self.con_emb(c)
            for flow in self.flows:
                x, ls = flow.forward(x, c)
                ld = ls.sum(-1)
                dis = torch.max(ls,-1)[0] - torch.min(ls,-1)[0]
                log_det += ld.view(-1)
                log_dis += dis.view(-1)
                # print(flow.__class__.__name__, ld.mean())
        else:
            log_det = log_dis = None
            for flow in self.flows:
                x, _ = flow.forward(x, c)
        return x, log_det, log_dis

    def forward_traj(self, x, c):
        b = x.shape[0]
        x_list = []
        if self.con_emb is not None:
            c = self.con_emb(c)
        for flow in self.flows:
            x, ls = flow.forward(x, c)
            x_list.append(x)
        return x_list

    def inverse(self, z, c):
        if self.con_emb is not None:
            c = self.con_emb(c)
        for flow in self.flows[::-1]:
            z, _ = flow.forward(z, c, mode='inverse')
        return z, None, None



###################################################################
# G-INN
###################################################################
class GINN(nn.Module):
    def __init__(self, adj, nin, nhid, cin, nl, outact=None):
        super().__init__()
        self.num_node = adj.shape[1]
        flows = []
        for _ in range(nl):
            flows += [ActNorm(nin), LUInvertibleMM(nin)]
            flows += [iGraphResidualLayer(adj, nin, nhid, cin)]
        if outact == 'sigmoid':
            flows += [ActNorm(nin), Sigmoid()]
        self.flows = nn.ModuleList(flows)

    def forward(self, x, c):
        x = x.view(-1, x.shape[-1])
        c = c.view(-1, c.shape[-1])
        be = x.shape[0]
        if self.training:
            log_det = torch.zeros(be).to(x.device)
            log_dis = torch.zeros(be).to(x.device)
            for flow in self.flows:
                x, ls = flow.forward(x, c)
                ld = ls.sum(-1)
                dis = torch.max(ls, -1)[0] - torch.min(ls, -1)[0]
                log_det += ld.view(-1)
                log_dis += dis.view(-1)
        else:
            log_det = log_dis = None
            for flow in self.flows:
                x, _ = flow.forward(x, c)
        x = x.view(-1, self.num_node, x.shape[-1])
        return x, log_det, log_dis

    def inverse(self, z, c):
        z = z.view(-1, z.shape[-1])
        c = c.view(-1, c.shape[-1])
        for flow in self.flows[::-1]:
            z, _ = flow.forward(z, c, mode='inverse')
        z = z.view(-1, self.num_node, z.shape[-1])
        return z, None, None





###################################################################
# Gauge-NN
###################################################################
class GaugeNN(nn.Module):
    """
    gauge NN: y = f(c) + g(x,c) * x
        f(c): interior point finder
        g(x,c): scaling function, monotonic on ||x||, conditioned on x/||x|| and c
    """
    def __init__(self, nx, nc, nh):
        super().__init__()
        self.interior_nn = nn.Sequential(nn.Linear(nc, nh), nn.ReLU(),
                                         nn.Linear(nh,nh), nn.ReLU(),
                                         nn.Linear(nh,nh), nn.ReLU(),
                                         nn.Linear(nh, nx))
        self.scaling_nn = nn.Sequential(nn.Linear(nx+nc, nh), nn.ReLU(),
                                         nn.Linear(nh,nh),  nn.ReLU(),
                                         nn.Linear(nh,nh), nn.ReLU(),
                                         nn.Linear(nh, nx), nn.Softplus())
    def forward(self, x, c):
        x0 = self.interior_nn(c)
        xn = torch.norm(x, dim=-1, p=2, keepdim=True)
        vec = x / (xn+1e-8)
        wx = self.scaling_nn(torch.cat([vec, c], dim=-1))
        return wx * x + x0

    def interior_forward(self, c):
        return self.interior_nn(c)
    
    def scaling_forward(self, x, c):
        xn = torch.norm(x, dim=-1, p=2, keepdim=True)
        vec = x / (xn+1e-8)
        wx = self.scaling_nn(torch.cat([vec, c], dim=-1))
        return torch.log(wx)
    
    def distortion_forward(self, x, c, sample_num=10, step_size=0.1):
        uni_vec = torch.randn(size=[1, sample_num, x.shape[1]], device = x.device)
        uni_vec = uni_vec / (torch.norm(uni_vec, dim=-1, p=2, keepdim=True) + 1e-8) * step_size
        x = x.unsqueeze(1)
        c = c.unsqueeze(1)
        c = torch.repeat_interleave(c, sample_num, 1)
        x = torch.repeat_interleave(x, sample_num, 1)
        dx = torch.log(torch.norm(self.forward(x + uni_vec,c) - self.forward(x,c), dim=-1, p=2))
        log_dis = torch.max(dx, dim=1)[0] - torch.min(dx, dim=1)[0]
        return log_dis

        
