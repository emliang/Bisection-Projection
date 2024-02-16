import torch
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import cKDTree
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

#################################################################
# Constraint for 2-dim toy example
#################################################################
class Complex_Constraints:
    def __init__(self):
        """
        xTx <= 0
        """
        self.A = torch.tensor(np.array([[1., -1.], [-1., -1.], [1., 1.], [-1., 1.]])).permute(1, 0).to(device=device)
        # self.bias = 2
        # self.b = torch.tensor(np.array([self.bias] * 4)).view(1, 4).to(device=device)
        # self.t_test = [ [1.5, 1.5, 0.5, 1.5, 1.5, 0.5],
        #                 [1.5, 0.5, 1.5, 1.5, 0.5, 0.5],
        #                 [0.5, 1.5, 0.5, 0.5, 1.5, 0.5],
        #                 [0.5, 0.5, 1.5, 1.5, 1.5, 0.5]]
        # self.sampling_range = np.array([[-0.01,-0.01,1,1,1,1,-1,-1], [2,2,1,1,1,1,1,1]])
        self.sampling_range = [-0.01,2]
        self.c_dim = 6 + 2
        self.n_dim = 2
        self.n_case = 8
        self.t_test = np.random.uniform(low=self.sampling_range[0],
                                        high=self.sampling_range[1],
                                        size=[self.n_case,self.c_dim])

    def cal_penalty(self, c, x):
        res = self.cal_res(c, x)
        return torch.clamp(res, min=0)
    
    def cal_barrier(self, c, x):
        res = self.cal_res(c, x)
        return torch.clamp(res, max=0)

    def cal_res(self, c, x):
        """
        y <= 0.5*x^2 + [0,2]
        y >= -0.5x^2 - [0,2]
        """
        bias = c[:,2:6]
        violation = torch.matmul(x, self.A) - (bias)
        violation_1 = (-c[:,[6]] * (x[:, [0]]) ** 2 + x[:, [1]] - c[:, [0]])
        violation_2 = (-c[:,[7]] * (x[:, [0]]) ** 2 - x[:, [1]] - c[:, [1]])
        violation = torch.cat([violation, violation_1, violation_2], dim=-1)
        return violation

    def check_feasibility(self, c, x):
        return self.cal_penalty(c, x)
    
    def scale(self, c, x):
        return x

    def complete_partial(self, c, x):
        return x

    def sampling_boudanry(self, c):
        ### fill area
        x_sample = np.linspace(-3, 3, 700)
        y_sample = np.linspace(-3, 3, 700)
        samples = [[x,y] for x in x_sample for y in y_sample]
        c_extend = np.concatenate([c for i in range(len(samples))], axis=0)
        c_tensor = torch.as_tensor(c_extend).to(device)
        x_tensor = torch.as_tensor(samples).to(device)
        penalty = self.cal_penalty(c_tensor, x_tensor).sum(-1)
        feasible_sample = (x_tensor[penalty<=0]).cpu().numpy()
        # plt.scatter(feasible_sample[:,0],feasible_sample[:,1], marker='.', alpha=0.9,
        #             zorder=0, c='whitesmoke', linewidths=0)

        ### plot the boudanry
        c = np.reshape(c, -1)
        x = np.linspace(-3, 3, 1000)
        all_boudanry = np.array([[x,  x - c[2]], [x, -x - c[3]], [x, -x + c[4]], [x,  x + c[5]],
                        [x, c[6] * x ** 2 + c[0]], [x, -c[7] * x ** 2 - c[1]]])
        all_boudanry = all_boudanry.transpose(0,2,1)
        all_boudanry = np.reshape(all_boudanry, [-1,2])

        tree1 = cKDTree(all_boudanry)
        tree2 = cKDTree(feasible_sample)
        threshold = 1e-2
        indices = tree2.query_ball_tree(tree1, r=threshold)
        # Flatten the list of indices and remove duplicates
        indices = set([i for sublist in indices for i in sublist])
        # Extract the matching points
        boundary = all_boudanry[list(indices)]
        # plt.scatter(boundary[:,0], boundary[:,1], alpha=0.7,
        #             zorder=0, c='cornflowerblue', marker='.', s=3, linewidths=0)
        return boundary

    def plot_boundary(self, c):
        x = np.linspace(-3, 3, 1000)
        # plt.plot(x,  x - c[2], linewidth=1.5, alpha=0.7, c='cornflowerblue')
        # plt.plot(x, -x - c[3], linewidth=1.5, alpha=0.7, c='cornflowerblue')
        # plt.plot(x, -x + c[4], linewidth=1.5, alpha=0.7, c='cornflowerblue')
        # plt.plot(x,  x + c[5], linewidth=1.5, alpha=0.7, c='cornflowerblue')
        # plt.plot(x, c[6] * x ** 2 + c[0], linewidth=1.5, alpha=0.7, c='cornflowerblue')
        # plt.plot(x, -c[7] * x ** 2 - c[1], linewidth=1.5, alpha=0.7, c='cornflowerblue')


        ### fill area
        x_sample = np.linspace(-3, 3, 700)
        y_sample = np.linspace(-3, 3, 700)
        samples = [[x,y] for x in x_sample for y in y_sample]
        c_extend = np.concatenate([c for i in range(len(samples))], axis=0)
        c_tensor = torch.as_tensor(c_extend).to(device)
        x_tensor = torch.as_tensor(samples).to(device)
        penalty = self.cal_penalty(c_tensor, x_tensor).sum(-1)
        feasible_sample = (x_tensor[penalty<=0]).cpu().numpy()
        plt.scatter(feasible_sample[:,0],feasible_sample[:,1], marker='.', alpha=0.9,
                    zorder=0, c='whitesmoke', linewidths=0)

        ### plot the boudanry
        c = np.reshape(c, -1)
        x = np.linspace(-3, 3, 1000)
        all_boudanry = np.array([[x,  x - c[2]], [x, -x - c[3]], [x, -x + c[4]], [x,  x + c[5]],
                        [x, c[6] * x ** 2 + c[0]], [x, -c[7] * x ** 2 - c[1]]])
        all_boudanry = all_boudanry.transpose(0,2,1)
        all_boudanry = np.reshape(all_boudanry, [-1,2])

        tree1 = cKDTree(all_boudanry)
        tree2 = cKDTree(feasible_sample)
        threshold = 1e-2
        indices = tree2.query_ball_tree(tree1, r=threshold)
        # Flatten the list of indices and remove duplicates
        indices = set([i for sublist in indices for i in sublist])
        # Extract the matching points
        boundary = all_boudanry[list(indices)]
        plt.scatter(boundary[:,0], boundary[:,1], alpha=0.7,
                    zorder=0, c='cornflowerblue', marker='.', s=3, linewidths=0)

class Convex_Constraints:
    def __init__(self):
        """
        Ax - b <= 0
        """
        self.A = torch.tensor(np.array([[1, -1], [-1, -1], [1, 1], [-1, 1]])).permute(1, 0).to(device=device)
        self.b = torch.tensor(np.array([2, 2, 2, 2])).view(1, 4).to(device=device)

        self.t_test = [[0.3, 0.3, 0.3, 0.3], [0.3, 0.3, 1.7, 1.7], [1.7, 1.7, 0.3, 1.7], [1.7, 1.7, 1.7, 1.7]]
        self.sampling_range = [0, 2]
        self.c_dim = 4

    def forward(self, x):
        violation = torch.matmul(x, self.A) - (self.b)

        return violation

    def cal_penalty(self, c, x):
        violation = torch.matmul(x, self.A) - c
        return torch.clamp(violation, 0)

    def check_feasibility(self, c, x):
        return self.cal_penalty(c, x)

    def scale(self, c, x):
        return x

    def complete_partial(self, c, x):
        return x

    def plot_boundary(self, c):
        x = np.linspace(-2, 2, 100)
        boundary, _, _, _ = plt.plot(x, x - c[0], x, -x - c[1], x, -x + c[2], x, x + c[3],
                                     linewidth=2, alpha=0.9, c='cornflowerblue', )
        return boundary

class Non_Convex_Constraints:
    def __init__(self):
        """
        A[x,y] - b + t <= 0
        """
        self.A = torch.tensor(np.array([[1, -1], [-1, -1], [1, 1], [-1, 1]])).permute(1, 0).to(device=device)
        self.bias = 2
        self.b = torch.tensor(np.array([self.bias] * 4)).view(1, 4).to(device=device)

        self.t_test = [[1.5, 1.5], [1.5, 0.2], [0.2, 1.5], [0.2, 0.2]]
        self.sampling_range = [0, 2]
        self.c_dim = 2

    def cal_penalty(self, c, x):
        """
        y <= 0.5*x^2 + [0,2]
        y >= -0.5x^2 - [0,2]
        """
        violation = torch.matmul(x, self.A) - (self.b)
        violation_1 = (-0.5 * (x[:, [0]]) ** 2 + x[:, [1]] - c[:, [0]])
        violation_2 = (-0.5 * (x[:, [0]]) ** 2 - x[:, [1]] - c[:, [1]])
        violation = torch.cat([violation, violation_1, violation_2], dim=-1)
        return violation

    def check_feasibility(self, c, x):
        return self.cal_penalty(c, x)

    def scale(self, c, x):
        return x

    def complete_partial(self, c, x):
        return x

    def plot_boundary(self, c):
        tt = self.bias
        x = np.linspace(-2, 2, 100)
        boundary, _, _ = plt.plot([0, tt, 0, -tt, 0], [tt, 0, -tt, 0, tt],
                                  x, 0.5 * x ** 2 + c[0],
                                  x, -0.5 * x ** 2 - c[1], linewidth=2, alpha=0.9, c='cornflowerblue', )
        return boundary

class Box_Constraints:
    def __init__(self):
        """
        -t1 < x < t1
        -t2 < y < t2
        """
        self.t_test = [[0.2, 1.8], [0.8, 1.2], [1.2, 0.8], [1.8, 0.2]]
        self.sampling_range = [0, 2]
        self.c_dim = 2

    def cal_penalty(self, c, x):
        violation = [torch.relu(x[:, [0]] - c[:, [0]]),
                     torch.relu(x[:, [1]] - c[:, [1]]),
                     torch.relu(-x[:, [0]] - c[:, [0]]),
                     torch.relu(-x[:, [1]] - c[:, [1]])]
        violation = torch.cat(violation, dim=-1)
        return violation

    def check_feasibility(self, c, x):
        return self.cal_penalty(c, x)

    def scale(self, c, x):
        return x

    def complete_partial(self, c, x):
        return x

    def plot_boundary(self, c):
        c = np.reshape(c, -1)
        t1 = c[0]
        t2 = c[1]
        ### fill area
        x_sample = np.linspace(-t1, t1, 700)
        y_sample = np.linspace(-t2, t2, 700)
        feasible_sample = [[x,y] for x in x_sample for y in y_sample]
        feasible_sample = np.reshape(feasible_sample, [-1,2])
        plt.scatter(feasible_sample[:,0],feasible_sample[:,1], marker='.', alpha=0.9,
                    zorder=0, c='whitesmoke', linewidths=0)

        boundary = plt.plot([t1, t1, -t1, -t1, t1], [t2, -t2, -t2, t2, t2], linewidth=2, alpha=0.7, c='cornflowerblue')

        return boundary


