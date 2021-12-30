
import torch
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

#输入的x为矩阵，行数表示维数，列数表示样本数
# x ：m*n
def guasspdf(x,mean,cov,dim):
    if dim == 1:
        std = torch.sqrt(cov)
        dis1D = Normal(loc=mean, scale=std)
        lp1 = dis1D.log_prob(x)
        pdf = torch.exp(lp1)
        return pdf
    elif dim == 2:
        pass 
        return pdf
    else:
        print('dim > 2 !')
        raise NotImplemented

def drawpdf(dim):
    fig = plt.figure(10)
    if dim==1:
        x = torch.linspace(-5, 5, 100).reshape(-1,1)
        me = torch.tensor([0.]).reshape(-1,1)
        cov = torch.tensor([1.]).reshape(-1,1)
        pdf = guasspdf(x,mean=me,cov=cov,dim=1)
        ax = fig.add_subplot(111)
        ax.plot(x.data,pdf.data)
        plt.show()
    else:
        print("dim > 2 !")
        raise NotImplemented

if __name__=='__main__':
    dim = 1
    drawpdf(dim=dim)
