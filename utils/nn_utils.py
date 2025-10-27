import torch
import torch.nn as nn
from .flows.iresblock import iResidualLayer, iGraphResidualLayer
from .flows.coupling import *
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
# FNN: FC-MLP, ResNet
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

class ResBlock(nn.Module):
    def __init__(self, n_in, n_hid):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, n_hid),
                                 nn.ReLU(),
                                 nn.Linear(n_hid, n_in),)
    def forward(self, x):
        return x + self.net(x)

class ResNet(nn.Module):
    def __init__(self, nin, nout, nh, nl, act='sigmoid'):
        super().__init__()
        net = [nn.Linear(nin, nh)]
        for _ in range(nl):
            net += [ResBlock(nh, nh), nn.Dropout(0.1)]
        net.append(nn.Linear(nh, nout))
        if act is not None:
            net.append(act_list[act])
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)

class NoiseResNet(nn.Module):
    def __init__(self, nin, nout, nh, nl, act='sigmoid', fixed_margin=False, gamma=0, noise_type='add'):
        super().__init__()
        net = [nn.Linear(nin, nh)]
        for _ in range(nl):
            net += [ResBlock(nh, nh), nn.Dropout(0.1)]
        net.append(nn.Linear(nh, nout))
        if gamma>0:
            net.append(NoiseModule(fixed_margin, gamma, noise_type))
        if act is not None:
            net.append(act_list[act])
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class NoiseModule(nn.Module):
    def __init__(self, fixed_margin, gamma, noise_type='add'):
        super().__init__()
        # Parameterize gamma in log-space to ensure positivity
        log_gamma = torch.log(torch.tensor(gamma))
        self.log_gamma = log_gamma if fixed_margin else nn.Parameter(log_gamma)
        self.noise_type = noise_type

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x)
            gamma = torch.exp(self.log_gamma)

            if self.noise_type == 'add':
                x = x + noise.detach() * gamma
            else:  # 'mul'
                x = x * (1 + noise.detach() * gamma)
        return x



###################################################################
# INN
###################################################################
class INN(nn.Module):
    """ A sequence of invertible layers """
    def __init__(self, nin, nhid, cin, nl, inv='made', outact='sigmoid', bilip=False, lip=2):
        super().__init__()
        flows = []
        for _ in range(nl):
            flows += [LUInvertibleMM(nin), Con_ActNorm(nin, cin)]
            if inv == 'made':
                flows.append( MADE(nin, nhid, cin, bilip=bilip, lip=lip) )
            elif inv == 'coupling':
                flows.append( CouplingLayer(nin, nhid, cin, bilip=bilip, lip=lip) )
            elif inv == 'residual':
                flows.append( iResidualLayer(nin, nhid, cin) )
        flows += [LUInvertibleMM(nin), Con_ActNorm(nin, cin)]
        if outact == 'sigmoid':
            flows += [ModifiedSigmoid()]
        if outact == 'tanh':
            flows += [Tanh()]
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
                # if flow.__class__.__name__ == 'iResidualLayer':
                #     x, ld = flow.forward(x, c)
                # else:
                x, ls = flow.forward(x, c)
                ld = ls.sum(-1)
                dis = torch.max(ls,-1)[0] - torch.min(ls,-1)[0]
                log_det += ld.view(-1)
                log_dis += dis.view(-1)
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
# GNN: Graph based NN, ResNet
###################################################################
class GraphResBlock(nn.Module):
    def __init__(self, n_in, n_hid, ein=None):
        super().__init__()
        self.node_emb_f = nn.Sequential(nn.Linear(n_in, n_hid))
        if ein is not None:
            self.node_emb_t = nn.Sequential(nn.Linear(n_in, n_hid))
            self.edge_emb = nn.Sequential(nn.Linear(ein, n_hid))
        self.net = nn.Sequential(nn.Linear(n_hid, n_hid), nn.ReLU(), nn.Linear(n_hid, n_in))

    def forward(self, x, e=None, adj=None):
        """
        x: batch * node * feature
        e : batch * node * node * feature
        adj: batch * num_node * num_node
        """
        num_node = x.shape[1]
        diag_indices = torch.arange(num_node)
        if e is not None:
            node_emb_1 = self.node_emb_f(x)
            node_emb_2 = self.node_emb_t(x)
            node_emb = node_emb_1.unsqueeze(1) + node_emb_2.unsqueeze(2)
            edge_emb = self.edge_emb(e)
            edge_emb[:, diag_indices, diag_indices, :] = 0
            node_emb[:, diag_indices, diag_indices, :] = 0
            node_emb = torch.relu(edge_emb + node_emb).mean(2)
        elif adj is not None:
            node_emb = self.node_emb_f(x)
            adj = adj.view(-1, num_node, num_node) + \
                  torch.eye(num_node).view(1, num_node, num_node).to(x.device)
            adj =  adj / adj.sum(2).view(-1, num_node, 1)
            node_emb = torch.matmul(adj, node_emb)
        else:
            raise NotImplementedError
        emb = self.net(node_emb)
        return emb + x

class GraphNet(nn.Module):
    def __init__(self, nin, nout, nh, ein=None, nl=3, act='sigmoid'):
        net = [nn.Linear(nin, nh)]
        for _ in range(nl):
            net += [GraphResBlock(nh, nh//2, ein)]
        net.append(nn.Linear(nh, nout))
        if act is not None:
            net.append(act_list[act])
        self.net = nn.ModuleList(net)

    def forward(self, x, e=None, adj=None):
        for layer in self.net:
            if isinstance(layer, GraphResBlock):
                x = layer(x, e, adj)
            else:
                x = layer(x)
        return x



###################################################################
# G-INN
###################################################################
class GINN(nn.Module):
    def __init__(self, num_node, nin, nhid, cin, ein,  nl, outact=None):
        super().__init__()
        flows = []
        for _ in range(nl):
            flows += [LUInvertibleMM(nin), ActNorm(nin)]
            flows += [iGraphResidualLayer(num_node, nin, nhid, cin, ein)]
        flows += [LUInvertibleMM(nin), ActNorm(nin)]
        if outact == 'sigmoid':
            flows += [Sigmoid()]
        self.flows = nn.ModuleList(flows)

    def forward(self, x, c, e, adj):
        num_node = x.shape[1]
        x = x.view(-1, x.shape[-1])
        if c is not None:
            c = c.view(-1, c.shape[-1])
        log_det = 0
        log_dis = 0
        for flow in self.flows:
            if isinstance(flow, iGraphResidualLayer):
                x, ls = flow.forward(x, c, e, adj)
                dis = 0
            else:
                x, ls = flow.forward(x)
                dis = torch.max(ls, -1)[0] - torch.min(ls, -1)[0]
            if self.training:
                ld = ls.sum(-1)
                log_det += ld
                log_dis += dis
            else:
                log_det = log_dis = None
        x = x.view(-1, num_node, x.shape[-1])
        return x, log_det, log_dis

    def inverse(self, z, c, adj):
        z = z.view(-1, z.shape[-1])
        if c is not None:
            c = c.view(-1, c.shape[-1])
        for flow in self.flows[::-1]:
            if isinstance(flow, iGraphResidualLayer):
                z, _ = flow.forward(z, c, adj, mode='inverse')
            else:
                z, _ = flow.forward(z, mode='inverse')
        z = z.view(-1, self.num_node, z.shape[-1])
        return z, None, None





###################################################################
# Gauge-NN
###################################################################
class GaugeNN(nn.Module):
    """
    gauge NN: x = f(c) + g(z,c) * z
        f(c): interior point predictor
        z: unit vectors in sphere
        g(z,c): scaling function (>0), such that x is boundary point
    objective function: min_{f,g}  V(x) - [max_z g(z,c) - min_z g(x,z)]
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

        
