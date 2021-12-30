
import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np
from normpdf1D2D import guasspdf


def gen_train_data(xl,xr,dx):
    set_d = np.arange(xl, xr, dx).reshape(1,-1)
    data1 = np.array(set_d)
    return data1

class FP_dataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def train_data(xlow,xup,dx):
    data = gen_train_data(xlow,xup,dx)
    train_data_d = FP_dataset(data=data)
    return train_data_d

if __name__ == '__main__':
    dx = 0.01
    x_data = train_data(-2.2,2.2,dx)
    print(x_data)
    me = torch.tensor([0.2])
    cov = torch.tensor([0.1])
    train_loader = DataLoader(dataset=x_data, batch_size=20, shuffle=True)
    for x in train_loader:
        print(x.shape)
        pdf_i = guasspdf(x,me,cov,1)
        print(pdf_i.shape)