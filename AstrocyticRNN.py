import numpy as np
import torch as torch
from scipy import sparse
from IPython import display
from IPython.display import display, clear_output
import math
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import random
import os
import time
import tempfile
from tqdm import tqdm

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))
print(torch.cuda.device_count())  # returns 1 in my case
dtype = torch.cuda.FloatTensor

'''
RNN
-----
Giacomo Vedovati
g.vedovati@wustl.edu

Braindynamics and Control Group, Dr. Ching.
Washington University in St. Louis
Department of Electrical & System Enginerring
ese.wustl.edu
'''

class RNN():

    def __init__(self, Nx, Ny, Nu, dt, alpha, Rseed=7):
        '''
        Initialization method for the RNN
        '''
        self.Nx = Nx
        self.Ny = Ny
        self.Nu = Nu
        self.dt = dt
        self.alpha = alpha

        # Reproducibility of the code
        self.Rseed = Rseed
        gen0 = torch.Generator()
        self.gen0 = gen0.manual_seed(self.Rseed)
        np.random.seed(self.Rseed)
        self.rng = np.random.RandomState(self.Rseed)

        # RNN shape parameters
        self.Nx_Nx = (self.Nx, self.Nx,)
        self.Nx_Nu = (self.Nx, self.Nu,)
        self.Nx_Ny = (self.Nx, self.Ny,)
        self.Nx_1 = (self.Nx, 1,)
        self.Ny_1 = (self.Ny, 1,)
        self.Nu_1 = (self.Nu, 1,)

        # RNN initialization
        pg = 0.1
        std = 1/math.sqrt(pg * self.Nx)
        g = 1.5

        self.P = 1*torch.eye(self.Nx).type(dtype)

        ANN_J = g * std * sparse.random(self.Nx, self.Nx, density=pg, random_state=self.rng,
                                        data_rvs=self.rng.randn).toarray()  # connectivity matrix
        self.W = (torch.as_tensor(ANN_J).float()).type(dtype)

        self.x = (0.5 * torch.randn(self.Nx_1, generator=self.gen0)).type(dtype)         # State vector
        
        # Feedback matrix
        self.W_f = (
            2 * (torch.randn(self.Nx_Ny, generator=self.gen0)-0.5)).type(dtype)

        # Input 1 and 2
        self.W_in = (
            2 * (torch.randn(self.Nx_Nu, generator=self.gen0)-0.5)).type(dtype)        

        # Network output
        self.Phi = (torch.zeros(self.Nx_Ny)
                        ).type(dtype)                                  # Decoder
        self.r = torch.tanh(self.x).type(dtype)
        self.y = torch.matmul(self.Phi.T, self.r).type(dtype)
        self.y = self.y.reshape(self.y.shape[0],).type(dtype)

        
    def fpass(self, u, feedb, regmat):
        '''
        'Forward pass' for a RNN
        '''
        if len(u.size()) == 0:
            u = 0

        if len(feedb.size()) == 0:
            feedb = self.y

        u = u.reshape(self.Nu_1).type(dtype)
        regmat = regmat.type(dtype)
        feedb = feedb.reshape(self.Ny_1).float().type(dtype)

        self.dxx = - self.x + torch.matmul(self.W + regmat, self.r) + \
            torch.matmul(self.W_f, feedb) + \
            torch.matmul(self.W_in, u) 

        self.x += self.dxx * self.dt
        self.r = torch.tanh(self.x)
        self.y = (torch.matmul(self.Phi.T, self.r)
                  ).reshape(self.Ny,)

        return self.y, self.r

def Force_Learning(err, Pw, r, Phi):
    '''
    Force learning for RNN
    '''
    err = err.reshape((1, err.shape[0])).type(dtype)
    PP_r = torch.matmul(Pw, r)
    P_Force_upd = torch.matmul(PP_r, PP_r.T) / \
        (1 + torch.matmul(r.T, PP_r))
    Pw = Pw - P_Force_upd
    Phi = Phi - torch.matmul(Pw, r).matmul(err)

    return Pw, Phi


if __name__ == "__main__":
    '''
    '''
    Nr = 5000  # Reservoir
    dt = 0.04
    t = 0
    output_dim = 1
    input_dim = 1
    #max_it = 10000
    trials = 50

    RNN_ = RNN(Nx=Nr,
            Ny=output_dim,
            Nu=input_dim,
            dt=dt,
            alpha=0.7)
    output = []
    true_sig = []
    accuracy = []
    normC = []
    itC = []
    countRNN = 0
    test_it = 0
    freq = [2, 1]
    ii1, qq1 = 1, 1

    gen0 = torch.Generator()
    reg = 0.001*torch.zeros_like(RNN_.r).reshape(-1)
    v1 = torch.rand(Nr, generator=gen0)
    v1[1000:] = 0
    v2 = torch.rand(Nr, generator=gen0)
    v2[2000:] = 0
    v11 = torch.rand(Nr, generator=gen0)
    v11[3000:] = 0    
    v21 = torch.rand(Nr, generator=gen0)
    
    regmat1 = 0.01*torch.outer(v1, v2) #  # .type(dtype)
    regmat2 = 0.001*torch.outer(v11, v21) #  # .type(dtype)
    
    phiP = 0
    omegaT1 = math.pi*(1/2)*dt
    omegaT2 = math.pi*(1/1.25)*dt
    
    for timSep in range(trials):
        indx = timSep+1
        if (indx % 2) == 0:
            l = 1
            e = 1
            c = 1 
            delta_f = freq[0]
            regmat = regmat1
            
        else:
            l = 2
            e = 2
            c = 2
            delta_f = freq[1]
            regmat = regmat2
        max_it = torch.randint(100, 250, (1,)).item()
        itC.append(max_it)

        for idx in range(max_it):  # tqdm(range(max_it)):
            if (indx % 2) == 0:
                phiP = phiP + omegaT1
            else:
                phiP = phiP + omegaT2
            t += dt
            Phi1 = RNN_.Phi
            input = torch.ones(input_dim)
            
            y_true = torch.sin(torch.tensor(phiP))
            true_sig.append(y_true.item())
            RNN_.fpass(input, RNN_.y, regmat)
            output.append(RNN_.y.item())

            err = RNN_.y - y_true
            RNN_.P, RNN_.Phi = Force_Learning(
                err, RNN_.P, RNN_.r, RNN_.Phi)

            test_it += 1
            countRNN += torch.linalg.norm(RNN_.y - y_true)
            normC.append(torch.linalg.norm(Phi1 - RNN_.Phi, ord='fro').item())

        #print("Network accuracy: " + str(countRNN))
        accuracy.append(countRNN)
    print("Average accuracy: " + str(sum(accuracy) / test_it))
    torch.save(RNN_.Phi, 'FinPhi.pt')
    torch.save(regmat1, 'regmat1.pt')
    torch.save(regmat2, 'regmat2.pt')
    torch.save(output, 'output.pt')
    torch.save(true_sig, 'true_sig.pt')
    
    fig, axs = plt.subplots(2, 1, sharex='col', figsize=(20, 8))
    axs[0].plot(output)
    axs[0].plot(true_sig)
    axs[0].set_ylabel('Tracking')
    for nit in range(trials):
        axs[0].axvline(x=sum(itC[:nit]), color='r', linestyle='--', ymin=-1.5, ymax=1.5)
    axs[0].grid(True)
    axs[1].plot(normC)
    axs[1].set_ylabel('Frobenius Norm')
    axs[1].grid(True)
    fig.tight_layout()
    plt.show()
    fig.savefig('temp_l.pdf', bbox_inches='tight')
    
    itC2 = []
    true_sig2 = []
    output2 = []
    trials2 = 40
    for timSep in range(trials2):
        indx = timSep+1
        if (indx % 2) == 0:
            regmat = regmat1
            
        else:
            regmat = regmat2
        max_it = torch.randint(100, 250, (1,)).item()
        itC2.append(max_it)
        #print(max_it)
        for idx in range(max_it):  # tqdm(range(max_it)):
            if (indx % 2) == 0:
                phiP = phiP + omegaT1
            else:
                phiP = phiP + omegaT2
            t += dt
            Phi1 = RNN_.Phi
            input = torch.ones(input_dim)
            y_true = torch.sin(torch.tensor(phiP))
            true_sig2.append(y_true.item())
            RNN_.fpass(input, RNN_.y, regmat)
            output2.append(RNN_.y.item())

            test_it += 1
            countRNN += torch.linalg.norm(RNN_.y - y_true)
            normC.append(torch.linalg.norm(Phi1 - RNN_.Phi, ord='fro').item())
        accuracy.append(countRNN)
    print("Average accuracy: " + str(sum(accuracy) / test_it))
    
    torch.save(output2, 'output_trained.pt')
    torch.save(true_sig2, 'true_trained.pt')
    
    plt.figure(figsize=(14, 6))
    plt.plot(output2)
    plt.plot(true_sig2)
    plt.grid(True)
    plt.show()
    plt.savefig('output_l.pdf', bbox_inches='tight')
