
from mmap import ACCESS_COPY
import time
from numpy.lib.nanfunctions import nanprod

import torch
import torch.nn as nn
from torch.nn.modules import module
from torch.utils.data import DataLoader
from normpdf1D2D import guasspdf
from get_train_data import train_data
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.integrate as integrate

class sys(object):
    def __init__(self):
        #system parameters
        parse = argparse.ArgumentParser()
        parse.add_argument('--sigma', type=float, default=0.50)
        parse.add_argument('--a', type=float, default=0.3)
        parse.add_argument('--b', type=float, default=0.50)
        parse.add_argument('--lr', type=float, default=0.01)
        parse.add_argument('--batch_size', type=int, default=6000)
        parse.add_argument('--num_epoch', type=int, default=3000)
        parse.add_argument('--num_comp', type=int, default=13)
        parse.add_argument('--dx', type=float, default=1e-3)
        parse.add_argument('--xl', type=float, default=-3)
        parse.add_argument('--xr', type=float, default=3)
        self.params = parse.parse_args()
    def fx(self,x):
        return self.params.a*x-self.params.b*x**3
    def dfx(self,x):
        return self.params.a-3.*self.params.b*x**2

class NeuralNet(nn.Module):
    def __init__(self,num_comp):
        super(NeuralNet, self).__init__()
        # initialization of NeuralNet for GMM density [weights, means, covs]
        kc = torch.tensor([num_comp], dtype=torch.int16)

        weights = torch.ones(kc)/kc
        means = torch.linspace(-2.0, 2.0, int(kc))
        covs = torch.ones(kc) * 0.2
        self.kc = kc
        self.ws = nn.Parameter(data=weights,requires_grad=True)
        self.ms = nn.Parameter(data=means,requires_grad=True)
        self.cs = nn.Parameter(data=covs,requires_grad=True)
  
    def forward(self, x):
        dim = 1
        # obtain system parameters
        eq = sys()
        #computing pdf and FP_operation
        FP = torch.zeros_like(x)
        pdf = torch.zeros_like(x)
        for k in range(self.kc):
            pdf_i = guasspdf(x, self.ms[k], self.cs[k], dim)
            cov_i_inv = 1.0 / (self.cs[k])
            df1 = cov_i_inv * (x - self.ms[k])
            df1px = -pdf_i * df1
            df2 = cov_i_inv * (x - self.ms[k]) * ((x - self.ms[k])) * cov_i_inv - cov_i_inv
            df2px = pdf_i * df2
            FP_pi = -df1px * eq.fx(x) - pdf_i * eq.dfx(x) + (eq.params.sigma ** 2) / 2 * df2px
            FP = FP + torch.abs(self.ws[k]) * FP_pi
            pdf = pdf + torch.abs(self.ws[k]) * pdf_i
        return FP,pdf

def train():
    accp_list = []
    loss_list = []
    # loss1_list=[]
    eq = sys()
    # generate training data
    train_x = train_data(eq.params.xl,eq.params.xr,eq.params.dx)
    train_loader = DataLoader(dataset=train_x, batch_size=eq.params.batch_size, shuffle=True)
    # NeuralNet training parameters
    model = NeuralNet(num_comp=eq.params.num_comp)
    optimizer = torch.optim.AdamW(model.parameters(),lr=eq.params.lr)
    # beginning train Neural Network
    for epoch in tqdm(range(eq.params.num_epoch)):
        for x in train_loader:
            FP_op,pdf = model(x)
            with torch.no_grad():
                accp=accl(model)
            loss1 = (FP_op**2).mean()
            loss2 = 1.0/len(model.ws)*(torch.abs(model.ws).sum() -1.0)**2
            # loss2=((pdf*eq.params.dx).sum()-1)**2
            loss = loss1 + loss2
            # 1. setting grad to zero
            optimizer.zero_grad()
            # 2. backward and getting grad
            loss.backward()
            # 3. next step
            optimizer.step()
            # recording training process
        if (epoch % 20) == 0:
            print('epoch:{},accp:{},loss:{}'.format(epoch,accp.item(),loss.item()))   
        loss_list.append(loss.item())
        accp_list.append(accp.item())
    return model,loss_list,accp_list
    
def accl(model):
    x1 = np.linspace(eq.params.xl,eq.params.xr,3000)
    pg1=pdf_gmm(x1,model)
    pa1=pdf_anay(x1)
    ma1 = np.linalg.norm(pa1-pg1)
    mb1 = np.linalg.norm(pa1)
    accp2=1-ma1/mb1
    return accp2

def pdf_anay(xdata):
    eq = sys()
    px = lambda x: np.exp((2 * eq.params.a * x ** 2 - eq.params.b * x ** 4) / (2 * (eq.params.sigma ** 2)))
    con = integrate.quad(px, xdata[0], xdata[-1])[0]
    y_anay = px(xdata) / con
    return y_anay

def pdf_gmm(xdata,model):
    x_t = torch.from_numpy(xdata)
    _,pdf = model(x_t)
    return torch.detach(pdf.data).numpy()

if __name__ == '__main__':
    eq = sys()

    model, loss, accp= train()
    print('train over !')

    torch.save(model, f'model_{eq.params.num_comp}.pkl')
    torch.save(loss, f'loss_{eq.params.num_comp}.txt')
    torch.save(accp, f'accp_{eq.params.num_comp}RMSprop.txt')
 
    model = torch.load(f'model_{eq.params.num_comp}.pkl')
    loss = torch.load(f'loss_{eq.params.num_comp}.txt')

    # print('========GMM Parameters ==================')
    print('weights={}'.format(model.ws.data))
    print('means={}'.format(model.ms.data))
    print('covs={}'.format(model.cs.data))

    xi = np.linspace(eq.params.xl,eq.params.xr,3000)
    pa = pdf_anay(xi) # analytical pdf
    pg = pdf_gmm(xi,model)
    maxerror=(np.array(pa-pg)).max()
    print('maxerror:' '%.2e' % maxerror)

  

    ma = np.linalg.norm(pa-pg)
    mb = np.linalg.norm(pa)
    accl=1-ma/mb
    print('accl: {:.2%}'.format(accl))

    # accl2=[]
    # for i in range(0,3000):
    #     ma1 = np.linalg.norm((np.array(pa))-(np.array(pg)))
    #     mb1 = np.linalg.norm((np.array(pa)))
    #     accl2.append(1-ma1/mb1)
    # print(accl2)
    # torch.save(accl2, f'accl2_{eq.params.num_comp}.txt')

    # draw pdf
    plt.ion()
    fig = plt.figure(10)
    ax = fig.add_subplot(111)
    ax.axis([-2.5, 2.5, -0.05, 0.6])
    ax.plot(xi, pa,color='black',label = 'Exact PDF',linewidth = 3)
    ax.plot(xi, pg,color='red',label = 'ML-AGMM PDF',linestyle='dashed',linewidth = 3)
    plt.xlabel('$X$')
    plt.ylabel('$PDF$')
    plt.legend()
    plt.savefig("10.png",dpi=600,bbox_inches = 'tight')

    # loss figure
    fig123 = plt.figure(123)
    ax123 = fig123.add_subplot(111)
    loss1=np.log10(loss)
    ax123.plot(loss1)
    plt.xlabel('epoch')
    plt.ylabel('$log_{10}(loss \quad function)$')
    plt.savefig("123.png",dpi=600,bbox_inches = 'tight')
    print('minimumloss:','%.2e' % (np.array(loss)).min())
    
    plt.ioff()
    plt.show()

