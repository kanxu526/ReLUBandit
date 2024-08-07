import numpy as np
import scipy as sp
import torch
import torch.nn as nn
import torch.optim as optim


class OFUL:
    """
    OFUL (Abbasi-Yadkori et al. 2011).
    """
    def __init__(self, d, lam, sig=0.05, delta=0.01, S=1):
        self.d = d # context dimension
        self.lam = lam # regularization hyperparameter
        self.sig = sig # std of noise
        self.delta = delta # high prob
        self.S = S # L2 norm upper bound of theta
        self.theta = np.zeros(d) # model parameter
        self.V_inv = np.eye(d) * 1/lam # V inverse
        self.B = 0 # X^TY
        self.logdetV = 0 # log det(V)
        self.cfrad = 0 # confidence width
        self.maxrcd = 0

    def choose_action(self, t, X):
        # X is N by d action set (arm set is unit sphere)
        ucbs = np.dot(X, self.theta) + self.cfrad * np.sqrt(np.sum(np.dot(X, self.V_inv) * X, 1))
        max_est = np.amax(ucbs)
        a = np.random.choice(np.argwhere(ucbs == max_est).flatten())
        self.action = X[a]
        # remove
        self.maxrcd = self.cfrad * np.sqrt(np.sum(np.dot(X, self.V_inv) * X, 1))[a]
        return self.action

    def update_model(self, rwd):
        xt = self.action
        self.logdetV += np.log(1 + np.dot(np.dot(self.V_inv, xt), xt))
        self.cfrad = self.sig * np.sqrt(self.logdetV - self.d * np.log(self.lam) + np.log(1/(self.delta**2))) + np.sqrt(self.lam) * self.S 
        self.V_inv -= self.V_inv @ np.outer(xt, xt) @ self.V_inv / (1 + xt @ self.V_inv @ xt) # Update V inverse using Sherman-Morrison formula
        self.B += xt * rwd
        self.theta = self.V_inv @ self.B


class Net(torch.nn.Module):
    def __init__(self, d, k):
        super(Net, self).__init__()
        self.fc_in = torch.nn.Linear(d, k, bias=False)
        self.fc_out = torch.sum
        self.activation = torch.nn.functional.relu

    def forward(self, x):
        x = self.activation(self.fc_in(x))
        return self.fc_out(x, dim=1)

def ReLUest(X, Y, d, k, learning_rate=1e-2, steps=2000):
    model = Net(d, k)
    loss_fn = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)

    for t in range(steps):
        loss = loss_fn(model(X), Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model.fc_in.weight.data.numpy() # k by d

class OFUReLU:
    """
    OFUReLU (Xu et al., 2023).
    """
    def __init__(self, d, k, lam, te, sig=0.05, delta=0.01, S=1):
        self.d = d # context dimension
        self.k = k # number of neurons
        self.lam = lam # regularization hyperparameter
        self.te = te # exploration phase
        self.sig = sig # std of noise
        self.delta = delta # high prob
        self.S = S # L2 norm upper bound of theta
        self.theta0 = np.zeros((k, d)) # true parameter
        self.theta = np.zeros(2 * d * k) # parameter in OFUL stage
        self.V_inv = np.eye(2 * d * k) * 1/lam # V inverse
        self.B = 0 # X^TY
        self.logdetV = 0 # log det(V)
        self.cfrad = 0 # confidence width
        self.Ye = np.zeros(self.te) # exploration Ys
        self.Xe = np.zeros((self.te, self.d)) # exploration Xs

    def choose_action(self, t, X):
        # X is N by d action set
        if t < self.te:
            ran_ind = np.random.choice(len(X))
            self.action = X[ran_ind]
        else:
            X_trans = np.apply_along_axis(self.xtrans, 1, X, self.theta0)
            ucbs = np.dot(X_trans, self.theta) + self.cfrad * np.sqrt(np.sum(np.dot(X_trans, self.V_inv) * X_trans, 1))
            max_est = np.amax(ucbs)
            self.action = X[np.random.choice(np.argwhere(ucbs == max_est).flatten())]
        return self.action

    def update_model(self, t, rwd):
        if t < self.te:
            self.Xe[t] = self.action
            self.Ye[t] = rwd
            if t == self.te-1:
                self.theta0 = ReLUest(self.Xe, self.Ye, self.d, self.k) # k by d matrix
                for i in range(t):
                    x = self.Xe[i]
                    y = self.Ye[i]
                    xt = self.xtrans(x, self.theta0)
                    self.B += xt * y
                    self.logdetV += np.log(1 + np.dot(np.dot(self.V_inv, xt), xt))
                    self.V_inv -= self.V_inv @ np.outer(xt, xt) @ self.V_inv / (1 + xt @ self.V_inv @ xt)
                    self.theta = self.V_inv @ self.B
        else:
            xt = self.xtrans(self.action, self.theta0)
            self.B += xt * rwd
            self.logdetV += np.log(1 + np.dot(np.dot(self.V_inv, xt), xt))
            self.cfrad = self.sig * np.sqrt(self.logdetV - 2*self.d*self.k * np.log(self.lam) + np.log(1/(self.delta**2))) + np.sqrt(self.lam) * self.S * np.sqrt(3*self.k) # confidence width
            self.V_inv -= self.V_inv @ np.outer(xt, xt) @ self.V_inv / (1 + xt @ self.V_inv @ xt)
            self.theta = self.V_inv @ self.B

    def xtrans(self, x, theta):
        # theta is k by d matrix
        # x is d-dim vector
        neuron_act = theta @ x > 0
        return np.concatenate((np.kron(neuron_act, x), np.kron(1/2-neuron_act, x)))


class Network(nn.Module):
    def __init__(self, dim, layer=1, hidden_size=100):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_size, bias=False)
        self.activate = nn.ReLU()
        self.layer = layer
        if layer==1:
            self.fc2 = torch.sum
        else:
            self.fc2 = nn.Linear(hidden_size, 1)
    def forward(self, x):
        if self.layer==1:
            return self.fc2(self.activate(self.fc1(x)), dim=1)
        else:
            return self.fc2(self.activate(self.fc1(x)))

class NeuralUCBDiag:
    """
    Neural UCB (Zhou et al., 2020). See https://github.com/uclaml/NeuralUCB/tree/master.
    """
    def __init__(self, dim, lamdba=1, nu=1, layer=1, hidden=100):
        self.func = Network(dim, layer=layer, hidden_size=hidden).cuda()
        self.context_list = []
        self.reward = []
        self.lamdba = lamdba
        self.total_param = sum(p.numel() for p in self.func.parameters() if p.requires_grad)
        self.U = lamdba * torch.ones((self.total_param,)).cuda()
        self.nu = nu

    def select(self, context):
        tensor = torch.from_numpy(context).float().cuda()
        mu = self.func(tensor)
        g_list = []
        sampled = []
        ave_sigma = 0
        ave_rew = 0
        for fx in mu:
            self.func.zero_grad()
            fx.backward(retain_graph=True)
            g = torch.cat([p.grad.flatten().detach() for p in self.func.parameters()])
            g_list.append(g)
            sigma2 = self.lamdba * self.nu * g * g / self.U
            sigma = torch.sqrt(torch.sum(sigma2))

            sample_r = fx.item() + sigma.item()

            sampled.append(sample_r)
            ave_sigma += sigma.item()
            ave_rew += sample_r
        arm = np.argmax(sampled)
        self.U += g_list[arm] * g_list[arm]
        return arm, g_list[arm].norm().item(), ave_sigma, ave_rew

    def train(self, context, reward):
        self.context_list.append(torch.from_numpy(context.reshape(1, -1)).float())
        self.reward.append(reward)
        optimizer = optim.SGD(self.func.parameters(), lr=1e-2, weight_decay=self.lamdba)
        length = len(self.reward)
        index = np.arange(length)
        np.random.shuffle(index)
        cnt = 0
        tot_loss = 0
        while True:
            batch_loss = 0
            for idx in index:
                c = self.context_list[idx]
                r = self.reward[idx]
                optimizer.zero_grad()
                delta = self.func(c.cuda()) - r
                loss = delta * delta
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
                tot_loss += loss.item()
                cnt += 1
                if cnt >= 1000:
                    return tot_loss / 1000
            if batch_loss / length <= 1e-3:
                return batch_loss / length