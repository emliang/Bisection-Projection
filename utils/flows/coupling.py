import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_default_dtype(torch.float64)

###################################################################
# Mask auto-regressive layers
# https://arxiv.org/pdf/1502.03509.pdf
# mask_type: input | None | output
###################################################################
def get_mask(in_features, out_features, in_flow_features, mask_type=None):
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
            self.cond_linear = nn.Sequential(nn.Linear(cond_in_features, 2 * out_features))
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
                 lip=1.5):
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
                 num_cond_inputs=None,
                 act='relu',
                 bilip=False,
                 lip=1.5):
        super(CouplingLayer, self).__init__()

        activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh}
        act_func = activations[act]
        self.num_inputs = num_inputs
        mask = torch.zeros(size=[1, num_inputs])
        mask[:, :num_inputs // 2] = 1
        self.mask = mask

        if num_cond_inputs is not None:
            total_inputs = num_inputs + num_cond_inputs
        else:
            total_inputs = num_inputs

        self.net = nn.Sequential(nn.Linear(total_inputs, num_hidden), act_func(),
                                 nn.Linear(num_hidden, num_hidden), act_func(),
                                 nn.Linear(num_hidden, num_inputs * 2))
        # def init(m):
        #     if isinstance(m, nn.Linear):
        #         m.bias.data.fill_(0)
        #         nn.init.orthogonal_(m.weight.data)
        self.LogL = np.log(lip)
        self.bilip = bilip

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        mask = self.mask.to(inputs.device)
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
# Invertible linear layer
###################################################################
class LUInvertibleMM(nn.Module):
    """ An implementation of a invertible matrix multiplication
    layer from Glow: Generative Flow with Invertible 1x1 Convolutions
    (https://arxiv.org/abs/1807.03039).
    """

    def __init__(self, num_inputs,
                 bilip=False,
                 lip=1.5):
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
        U = self.LU * self.U_mask + torch.diag(self.sign_S * torch.exp(log_s))

        if mode == 'direct':
            W = self.P @ L @ U
            return inputs @ W, log_s.unsqueeze(0).repeat(inputs.size(0), 1)
        else:
            W_inv = torch.inverse(U) @ torch.inverse(L) @ self.P.t()
            return inputs @ W_inv, -log_s.unsqueeze(0).repeat(inputs.size(0), 1)


###################################################################
# Normalization layer
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
            return inputs * torch.exp(-self.weight) + self.bias, -self.weight.unsqueeze(0).repeat(inputs.size(0), 1)


class Con_ActNorm(nn.Module):
    """ An implementation of a activation normalization layer
    from Glow: Generative Flow with Invertible 1x1 Convolutions
    (https://arxiv.org/abs/1807.03039).
    """
    def __init__(self, num_inputs, num_cond_inputs):
        super(Con_ActNorm, self).__init__()
        n_hid = (num_inputs + num_cond_inputs) // 2
        self.weight = nn.Sequential(nn.Linear(num_cond_inputs, n_hid), nn.ReLU(),
                                    nn.Linear(n_hid, num_inputs))
        self.bias = nn.Sequential(nn.Linear(num_cond_inputs, n_hid), nn.ReLU(),
                                  nn.Linear(n_hid, num_inputs))

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        weight = self.weight(cond_inputs)
        bias = self.bias(cond_inputs)

        if mode == 'direct':
            return (inputs - bias) * torch.exp(weight), weight
        else:
            return inputs * torch.exp(- weight) + bias, - weight


###################################################################
# Non-linear activation
###################################################################
class Tanh(nn.Module):
    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            y = torch.tanh(inputs)
            return y, torch.log((1 - y ** 2))
        else:
            return torch.atanh(inputs), -torch.log(1 - inputs ** 2)


class Tanh_inverse(nn.Module):
    def __init__(self):
        super(Tanh_inverse, self).__init__()

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            return torch.atanh(inputs), -torch.log(1 - inputs ** 2)
        else:
            y = torch.tanh(inputs)
            return y, torch.log((1 - y ** 2))


class Sigmoid_inverse(nn.Module):
    def __init__(self):
        super(Sigmoid_inverse, self).__init__()

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            return torch.log(inputs / (1 - inputs)), -torch.log(inputs - inputs ** 2)
        else:
            y = torch.sigmoid(inputs)
            return y, torch.log(y * (1 - y))


class Sigmoid(nn.Module):
    def __init__(self):
        super(Sigmoid, self).__init__()
        # self.lower = -0.1
        # self.upper = 1.1

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            y = torch.sigmoid(inputs)
            scale_y = y
            return scale_y, torch.log(y * (1 - y))
        else:
            x = inputs
            return torch.log(x / (1 - x)), -torch.log((inputs - inputs ** 2))
