import torch
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import cKDTree

device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
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
        self.fix_c = np.array([[0.5, 0.5, 2, 2, 2, 2, 1, 1]])
        self.sampling_range = [0.01,2]
        self.c_dim = 6 + 2
        self.n_dim = 2
        self.n_case = 8
        self.t_test = np.random.uniform(low=self.sampling_range[0],
                                        high=self.sampling_range[1],
                                        size=[self.n_case,self.c_dim])

    def cal_penalty(self, c, x):
        res = self.cal_res(c, x)
        return torch.clamp(res, min=0)


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

    def sampling_infeasible(self, c, err, density):
        # def offset_sampling(data,c,err,density):
        c = c.as_tensor(device=device)
        c_offset=c+torch.tensor([[err,err,np.sqrt(2)*err,np.sqrt(2)*err,np.sqrt(2)*err,np.sqrt(2)*err,0,0]]).to(device)
        offset=self.sampling_boudanry(c_offset.cpu().numpy())
        quadrant_1 = offset[((offset[:,0] > 0) & (offset[:,1] > 0))]
        quadrant_2 = offset[((offset[:,0] < 0) & (offset[:,1] > 0))]
        quadrant_3 = offset[((offset[:,0] < 0) & (offset[:,1] < 0))]
        quadrant_4 = offset[((offset[:,0] > 0) & (offset[:,1] < 0))]
        quadrant_1=quadrant_1[np.argsort(quadrant_1[:, 1]/quadrant_1[:, 0])]
        quadrant_2=quadrant_2[np.argsort(quadrant_2[:, 1]/quadrant_2[:, 0])]
        quadrant_3=quadrant_3[np.argsort(quadrant_3[:, 1]/quadrant_3[:, 0])]
        quadrant_4=quadrant_4[np.argsort(quadrant_4[:, 1]/quadrant_4[:, 0])]
        offset=np.concatenate((quadrant_1,quadrant_2,quadrant_3,quadrant_4),axis=0)
        samples=offset[::density,:]
        samples=torch.tensor(samples).to(device)
        return samples        



    def plot_boundary(self, c):
        c = np.reshape(c, -1)
        x = np.linspace(-3, 3, 1000)
        plt.plot(x,  x - c[2], linewidth=1.5, alpha=0.7, c='cornflowerblue')
        plt.plot(x, -x - c[3], linewidth=1.5, alpha=0.7, c='cornflowerblue')
        plt.plot(x, -x + c[4], linewidth=1.5, alpha=0.7, c='cornflowerblue')
        plt.plot(x,  x + c[5], linewidth=1.5, alpha=0.7, c='cornflowerblue')
        plt.plot(x, c[6] * x ** 2 + c[0], linewidth=1.5, alpha=0.7, c='cornflowerblue')
        plt.plot(x, -c[7] * x ** 2 - c[1], linewidth=1.5, alpha=0.7, c='cornflowerblue')

    def fill_constraint(self, c):
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



#################################################################
# Constraint for 4 disconnected balls
#################################################################
class Disconnected_Ball:
    def __init__(self, n_ball=4):
        """
        center, radius
        """
        self.sampling_range = [0.01,2]
        self.center_range = [-1, 1]
        self.radius_range = [0.3,0.7]
        self.n_ball = n_ball
        self.c_dim = self.n_ball * 3 #[center, radius]
        self.n_dim = 2
        self.fix_c = np.array([[1, 1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1]])
        self.sampling_range = np.array([[0,0,-1, 0,-1,-1, 0,-1, 0.4, 0.4, 0.4, 0.4] ,
                                        [1,1, 0, 1, 0, 0, 1, 0, 0.7, 0.7, 0.7, 0.7]])
        # self.sampling_range = np.array([[-1,-1,-1, -1,-1,-1, -1,-1, 0.1,0.1,0.1,0.1] ,
        #                                 [1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    def cal_penalty(self, c, x):
        res = self.cal_res(c, x)
        return torch.clamp(res, min=0)


    def cal_res(self, c, x):
        batch = c.shape[0]
        center = c[:, : self.n_ball * 2].view(batch, self.n_ball, -1) 
        radius = c[:, self.n_ball * 2 :].view(batch, self.n_ball) 
        x = x.view(batch, 1, -1)
        residual = torch.norm(x - center, dim=-1, p=2) - radius
        residual = torch.min(residual, dim=1, keepdim=True)[0] # batch * 1
        return residual

    def check_feasibility(self, c, x):
        return self.cal_penalty(c, x)
    
    def scale(self, c, x):
        return x

    def complete_partial(self, c, x):
        return x


    def sampling_infeasible(self, c, err=0.2, density=10):
        c = np.reshape(c, -1)
        center = c[: self.n_ball * 2]
        radius = c[self.n_ball * 2 :]

        theta = np.linspace(0, np.pi*2, density)
        x_list = []
        y_list = []
        for i in range(self.n_ball):
            center_i = center[i*2 : (i+1)*2]
            xi = center_i[0] + np.cos(theta) * (radius[i] + err)
            yi = center_i[1] + np.sin(theta) * (radius[i] + err)
            x_list.append(xi)
            y_list.append(yi)
        x = np.reshape(np.concatenate(x_list, axis=0), (-1,1))
        y = np.reshape(np.concatenate(y_list, axis=0), (-1,1))
        samples = np.concatenate([x, y], axis=1)
        samples = torch.as_tensor(samples).to(device)
        return samples


    def plot_boundary(self, c):
        c = np.reshape(c, -1)
        center = c[: self.n_ball * 2]
        radius = c[self.n_ball * 2 :]

        theta = np.linspace(0, np.pi*2, 100)
        for i in range(self.n_ball):
            center_i = center[i*2 : (i+1)*2]
            xi = center_i[0] + np.cos(theta) * radius[i]
            yi = center_i[1] + np.sin(theta) * radius[i]
            plt.scatter(xi,  yi, marker='.', alpha=0.9, c='lightgray', zorder=0)

    def fill_constraint(self, c):
        c = np.reshape(c, -1)
        center = c[: self.n_ball * 2]
        radius = c[self.n_ball * 2 :]
        theta = np.linspace(0, np.pi*2, 250)
        ### fill area
        for i in range(self.n_ball):
            rt = np.linspace(0, radius[i], 50)
            center_i = center[i*2 : (i+1)*2]
            samples = [[center_i[0] + np.cos(x) * y,
                        center_i[1] + np.sin(x) * y] for x in theta for y in rt]
            samples = np.array(samples)
            plt.scatter(samples[:,0],samples[:,1], marker='o',
                        alpha=0.5, edgecolors='gainsboro',
                        zorder=0, c='gainsboro', s=2)

        ### plot the boudanry
        # for i in range(self.n_ball):
        #     center_i = center[i*2 : (i+1)*2]
        #     xi = center_i[0] + np.cos(theta) * radius[i]
        #     yi = center_i[1] + np.sin(theta) * radius[i]
        #     plt.scatter(xi,  yi, alpha=0.01,
        #             zorder=0, c='cornflowerblue', marker='.', s=1)
        # plt.show()
        # plt.close()


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


