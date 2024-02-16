import numpy as np
import scipy as sp
import scipy.linalg
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)

###################################################################
# Mask auto-regressive layers
###################################################################
def get_mask(in_features, out_features, in_flow_features, mask_type=None):
    """
    mask_type: input | None | output

    See Figure 1 for a better illustration:
    https://arxiv.org/pdf/1502.03509.pdf
    """
    if mask_type == 'input':
        in_degrees = torch.arange(in_features) % in_flow_features
    else:
        in_degrees = torch.arange(in_features) % (in_flow_features - 1)

    if mask_type == 'output':
        out_degrees = torch.arange(out_features) % in_flow_features - 1
    else:
        out_degrees = torch.arange(out_features) % (in_flow_features - 1)

    return (out_degrees.unsqueeze(-1) >= in_degrees.unsqueeze(0))

class MaskedLinear(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 mask,
                 cond_in_features=None,
                 bias=True):
        super(MaskedLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        if cond_in_features is not None:
            self.cond_linear = nn.Sequential(nn.Linear(cond_in_features, 2*out_features))
        self.register_buffer('mask', mask)
    def forward(self, inputs, cond_inputs=None):
        output = F.linear(inputs, self.linear.weight * self.mask, self.linear.bias)
        if cond_inputs is not None:
            w, b = self.cond_linear(cond_inputs).chunk(2, 1)
            output = output * w + b
        return output

nn.MaskedLinear = MaskedLinear

class MADE(nn.Module):
    """ An implementation of MADE
    (https://arxiv.org/abs/1502.03509).
    """
    def __init__(self,
                 num_inputs,
                 num_hidden,
                 num_cond_inputs=None,
                 act='relu',
                 bilip=False,
                 lip = 1.5):
        super(MADE, self).__init__()
        activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh}
        act_func = activations[act]
        input_mask = get_mask(num_inputs, num_hidden, num_inputs, mask_type='input')
        hidden_mask = get_mask(num_hidden, num_hidden, num_inputs)
        output_mask = get_mask(num_hidden, num_inputs * 2, num_inputs, mask_type='output')
        self.joiner = nn.MaskedLinear(num_inputs, num_hidden, input_mask, num_cond_inputs)
        self.trunk = nn.Sequential(act_func(), nn.MaskedLinear(num_hidden, num_hidden, hidden_mask),
                                   act_func(), nn.MaskedLinear(num_hidden, num_inputs * 2, output_mask))
        self.LogL = np.log(lip)
        self.bilip = bilip

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            h = self.joiner(inputs, cond_inputs)
            m, a = self.trunk(h).chunk(2, 1)
            if self.bilip:
                a = torch.tanh(a) * self.LogL
            u = (inputs - m) * torch.exp(-a)
            return u, -a
        else:
            x = torch.zeros_like(inputs)
            for i_col in range(inputs.shape[1]):
                h = self.joiner(x, cond_inputs)
                m, a = self.trunk(h).chunk(2, 1)
                if self.bilip:
                    a = torch.tanh(a) * self.LogL
                x[:, i_col] = inputs[:, i_col] * torch.exp(a[:, i_col]) + m[:, i_col]
            return x, -a


###################################################################
# Coupling layers
###################################################################
class CouplingLayer(nn.Module):
    """ 
    An implementation of a coupling layer
    from RealNVP (https://arxiv.org/abs/1605.08803).
    """
    def __init__(self,
                 num_inputs,
                 num_hidden,
                 mask,
                 num_cond_inputs=None,
                 act='relu',
                 bilip=False,
                 lip = 1.5):
        super(CouplingLayer, self).__init__()

        activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh}
        act_func = activations[act]

        self.num_inputs = num_inputs
        self.mask = mask

        if num_cond_inputs is not None:
            total_inputs = num_inputs + num_cond_inputs
        else:
            total_inputs = num_inputs
  
        self.net = nn.Sequential(nn.Linear(total_inputs, num_hidden), act_func(),
                                        nn.Linear(num_hidden, num_hidden), act_func(),
                                        nn.Linear(num_hidden, num_inputs*2))
        # def init(m):
        #     if isinstance(m, nn.Linear):
        #         m.bias.data.fill_(0)
        #         nn.init.orthogonal_(m.weight.data)
        self.LogL = np.log(lip)
        self.bilip = bilip


    def forward(self, inputs, cond_inputs=None, mode='direct'):
        mask = self.mask
        masked_inputs = inputs * mask
        if cond_inputs is not None:
            masked_inputs = torch.cat([masked_inputs, cond_inputs], -1)

        h = self.net(masked_inputs)
        log_s, t = h.chunk(2, 1)
        log_s = log_s * (1 - mask)
        t = t * (1 - mask)

        if self.bilip:
            log_s = torch.tanh(log_s) * self.LogL

        if mode == 'direct':
            s = torch.exp(log_s)
            return inputs * s + t, log_s
        else:
            s = torch.exp(-log_s)
            return (inputs - t) * s, -log_s


###################################################################
# Residual layers
###################################################################
class iResidualLayer(nn.Module):
    def __init__(self,
                 num_inputs,
                 num_hidden,
                 num_cond_inputs=None):
        super(iResidualLayer, self).__init__() 
        if num_cond_inputs is not None:
            self.w = nn.Sequential(nn.Linear(num_cond_inputs, num_hidden), nn.Tanh())
            self.b = nn.Sequential(nn.Linear(num_cond_inputs, num_hidden))
        self.emb = nn.Sequential(LinearNormalized(num_inputs, num_hidden), nn.ReLU())
        self.cat = nn.Sequential(LinearNormalized(num_hidden, num_hidden), nn.ReLU(),
                                 LinearNormalized(num_hidden, num_inputs))
    
    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            emb = self.emb(inputs)
            if cond_inputs is not None:
                w = self.w(cond_inputs)
                b = self.b(cond_inputs)
                emb = emb * w  + b
            gx = self.cat(emb)
            return inputs + gx, self.logdetestimator(inputs, gx)
        else:
            x = inputs
            if cond_inputs is not None:
                w = self.w(cond_inputs)
                b = self.b(cond_inputs)
            for _ in range(100):
                emb = self.emb(x)
                if cond_inputs is not None:
                    emb = emb * w + b
                gx = self.cat(emb)
                x = inputs - gx
            return x, None
    
    def logdetestimator(self, x, gx):
        jacobian = []
        for i  in range(gx.shape[1]):
            gt = gx[:,i:i+1]
            jt = torch.autograd.grad(
                outputs=gt,
                inputs=x,
                grad_outputs=torch.ones_like(gt),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,)[0]
            jacobian.append(jt)
        jacobian = torch.stack(jacobian, dim=-1)

        num_power_series = 10
        logdet = 0
        coefficient = [(-1)**(n-1)/n for n in range(1, num_power_series+1)]
        u = torch.randn(size=[jacobian.shape[0], jacobian.shape[1]])
        ut = u
        for n in range(num_power_series):
            ut = (jacobian * ut.view(jacobian.shape[0], 1, jacobian.shape[1])).sum(-1)
            logdet += coefficient[n] * (ut * u).sum(-1, keepdim=True)
        return logdet.view(-1,1,1)


from torch.nn.utils import spectral_norm
class LinearNormalized(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearNormalized, self).__init__(in_features, out_features, bias)
        self.linear = spectral_norm(nn.Linear(in_features, out_features))

    def forward(self, x):
        return self.linear(x)



###################################################################
# 1x1 Conv layers
###################################################################
class LUInvertibleMM(nn.Module):
    """ An implementation of a invertible matrix multiplication
    layer from Glow: Generative Flow with Invertible 1x1 Convolutions
    (https://arxiv.org/abs/1807.03039).
    """

    def __init__(self, num_inputs,
                       bilip=False,
                       lip = 1.5):
        super(LUInvertibleMM, self).__init__()
        self.W = torch.Tensor(num_inputs, num_inputs)
        nn.init.orthogonal_(self.W)
        self.L_mask = torch.tril(torch.ones(self.W.size()), -1)
        self.U_mask = self.L_mask.t().clone()
        self.I = torch.eye(self.W.size(0))

        # P, L, U = sp.linalg.lu(self.W.numpy())
        P, L, U = torch.linalg.lu(self.W)
        LU = L * self.L_mask + U * self.U_mask
        self.P = P
        self.LU = nn.Parameter(LU)
        # self.L = nn.Parameter(torch.tensor(L))
        # self.U = nn.Parameter(torch.tensor(U))
        S = torch.diag(U)
        sign_S = torch.sign(S)
        log_S = torch.log(torch.abs(S))
        self.sign_S = sign_S
        self.log_S = nn.Parameter(log_S)

        self.bilip = bilip
        self.LogL = np.log(lip)

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if str(self.L_mask.device) != str(self.log_S.device):
            self.L_mask = self.L_mask.to(self.log_S.device)
            self.U_mask = self.U_mask.to(self.log_S.device)
            self.I = self.I.to(self.log_S.device)
            self.P = self.P.to(self.log_S.device)
            self.sign_S = self.sign_S.to(self.log_S.device)


        if self.bilip:
            log_s = torch.tanh(self.log_S) * self.LogL
        else:
            log_s = self.log_S
        L = self.LU * self.L_mask + self.I
        U = self.LU * self.U_mask + torch.diag(
            self.sign_S * torch.exp(log_s))


        if mode == 'direct':
            W = self.P @ L @ U
            return inputs @ W, log_s.unsqueeze(
                0).repeat(inputs.size(0), 1)
        else:
            W_inv = torch.inverse(U) @ torch.inverse(L) @ self.P.t()
            return inputs @ W_inv, -log_s.unsqueeze(0).repeat(
                    inputs.size(0), 1)


###################################################################
# Act-Norm
###################################################################
class ActNorm(nn.Module):
    """ An implementation of a activation normalization layer
    from Glow: Generative Flow with Invertible 1x1 Convolutions
    (https://arxiv.org/abs/1807.03039).
    """
    def __init__(self, num_inputs):
        super(ActNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_inputs))
        self.bias = nn.Parameter(torch.zeros(num_inputs))
        self.initialized = False

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if self.initialized == False:
            self.weight.data.copy_(torch.log(1.0 / (inputs.std(0) + 1e-12)))
            self.bias.data.copy_(inputs.mean(0))
            self.initialized = True

        if mode == 'direct':
            return (inputs - self.bias) * torch.exp(self.weight), self.weight.unsqueeze(0).repeat(inputs.size(0), 1)
        else:
            return inputs * torch.exp( -self.weight) + self.bias, -self.weight.unsqueeze(0).repeat(inputs.size(0), 1)

class Con_ActNorm(nn.Module):
    """ An implementation of a activation normalization layer
    from Glow: Generative Flow with Invertible 1x1 Convolutions
    (https://arxiv.org/abs/1807.03039).
    """
    def __init__(self, num_inputs, num_cond_inputs):
        super(Con_ActNorm, self).__init__()
        n_hid = (num_inputs+num_cond_inputs)//2
        self.weight = nn.Sequential(nn.Linear(num_cond_inputs, n_hid),nn.ReLU(),
                                    nn.Linear(n_hid, num_inputs))
        self.bias = nn.Sequential(nn.Linear(num_cond_inputs, n_hid),nn.ReLU(),
                                    nn.Linear(n_hid, num_inputs))
        self.initialized = False

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if self.initialized == False:
            weight = torch.log(1.0 / (inputs.std(0) + 1e-12)).view(1,-1)
            bias = inputs.mean(0).view(1,-1)
            self.initialized = True
        else:
            weight = self.weight(cond_inputs)
            bias = self.bias(cond_inputs)

        if mode == 'direct':
            return (inputs - bias) * torch.exp(weight), weight
        else:
            return inputs * torch.exp( - weight) + bias, - weight


###################################################################
# Non-linear activation
###################################################################
class Tanh(nn.Module):
    def __init__(self):
        super(Tanh, self).__init__()
    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            y = torch.tanh(inputs)
            return y, torch.log((1 - y**2))
        else:
            return torch.atanh(inputs), -torch.log(1 - inputs**2)

class Tanh_inverse(nn.Module):
    def __init__(self):
        super(Tanh_inverse, self).__init__()
    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            return torch.atanh(inputs), -torch.log(1 - inputs**2)
        else:
            y = torch.tanh(inputs)
            return y, torch.log((1 - y ** 2))

class Sigmoid_inverse(nn.Module):
    def __init__(self):
        super(Sigmoid_inverse, self).__init__()

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            return torch.log(inputs / (1 - inputs)), -torch.log(inputs - inputs**2)
        else:
            y = torch.sigmoid(inputs)
            return y, torch.log(y * (1 - y))

class Sigmoid(nn.Module):
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.lower = -0.1
        self.upper = 1.1

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            y = torch.sigmoid(inputs)
            scale_y =  y * (self.upper-self.lower) + self.lower
            return scale_y, torch.log(y * (1 - y) * (self.upper-self.lower))
        else:
            x = (inputs - self.lower)/(self.upper-self.lower)
            return torch.log(x / (1 - x)), -torch.log((inputs - inputs**2)/(self.upper-self.lower))


# import os
# import sys
# url = os.path.join(os.getcwd(),'nflib')
# sys.path.append(url)


###################################################################
# NN 
###################################################################
class ResBlock(nn.Module):
    def __init__(self, n_in, n_hid):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, n_hid),
                                 nn.ReLU(),
                                 nn.Linear(n_hid, n_in))
    def forward(self, x):
        return x + self.net(x)

class ResNet(nn.Module):
    def __init__(self, nin, nout, nh, nl, act='sigmoid'):
        super().__init__()
        act_list = {'relu': nn.ReLU(), 'silu': nn.SiLU(), 'softplus': nn.Softplus(), 
                    'sigmoid':  nn.Sigmoid(), 'softmax': nn.Softmax(dim=-1)}
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

class FCNet(nn.Module):
    def __init__(self, nin, nout, nh, nl, act='sigmoid'):
        super().__init__()
        act_list = {'relu': nn.ReLU(), 'silu': nn.SiLU(), 'softplus': nn.Softplus(), 
                    'sigmoid':  nn.Sigmoid(), 'softmax': nn.Softmax(dim=-1)}
        net = [nn.Linear(nin, nh), nn.BatchNorm1d(nh), nn.ReLU()]
        for _ in range(nl):
            net += [nn.Linear(nh, nh), nn.BatchNorm1d(nh), nn.ReLU()]
        net.append(nn.Linear(nh, nout))
        if act is not None:
            net.append(act_list[act])
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)
    

###################################################################
# INN
###################################################################
class INN(nn.Module):
    """ A sequence of invertible layers """
    def __init__(self, flows, con_emb=None):
        super().__init__()
        self.flows = nn.ModuleList(flows)
        self.con_emb = con_emb

    def forward(self, x, t):
        m, _ = x.shape
        log_det = torch.zeros(m).to(x.device)
        log_dis = torch.zeros(m).to(x.device)
        if self.con_emb is not None:
            t = self.con_emb(t)
        for flow in self.flows:
            x, ls = flow.forward(x, t)
            # print(flow.__class__.__name__, ls.exp().max().detach(),ls.exp().min().detach())
            ld = ls.sum(-1)
            dis = torch.max(ls,-1)[0] - torch.min(ls,-1)[0]
            log_det += ld.view(-1)
            # if flow.__class__.__name__ != 'ActNorm':
            log_dis += dis.view(-1)
        return x, log_det, log_dis

    def forward_traj(self, x, t):
        m, _ = x.shape
        x_list = []
        if self.con_emb is not None:
            t = self.con_emb(t)
        for flow in self.flows:
            x, ls = flow.forward(x, t)
            x_list.append(x)
        return x_list

    def backward(self, z, t):
        if self.con_emb is not None:
            t = self.con_emb(t)
        for flow in self.flows[::-1]:
            z, _ = flow.forward(z, t, mode='inverse')
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

        
