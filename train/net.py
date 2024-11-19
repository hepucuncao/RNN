import torch
from torch import nn

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RNN(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,num_classes):
        super(RNN, self).__init__()
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.lstm=nn.LSTM(input_size,hidden_size,num_layers,batch_first=True)
        self.fc=nn.Linear(hidden_size,num_classes)

    def forward(self,x):
        #set initial hidden and cell states
        h0=torch.zeros(self.num_layers,x.size(0),self.hidden_size).to(device)
        c0=torch.zeros(self.num_layers,x.size(0),self.hidden_size).to(device)

        #forward propagate LSTM
        out,_=self.lstm(x,(h0,c0))#out：tensor of shape(batch_size,seq_length,hidden_size)
        #decode the hidden state of the last time step
        out=self.fc(out[:,-1,:])
        return out

'''
import torch
from torch import nn
class RNNCell(nn.Module):
    def __init__(self,input_dim,output_dim):
        super().__init__()
        self.U=nn.Parameter(torch.randn(size=(input_dim,output_dim),dtype=torch.float32))
        self.W=nn.Parameter(torch.randn(size=(output_dim,output_dim),dtype=torch.float32))
        self.V=nn.Parameter(torch.randn(size=(output_dim,output_dim),dtype=torch.float32))
        self.b1=nn.Parameter(torch.randn(size=(1,output_dim),dtype=torch.float32))
        self.b2=nn.Parameter(torch.randn(size=(1,output_dim),dtype=torch.float32))
        self.act1=nn.Tanh()
        self.act2=nn.Sigmoid()
    def forward(self,input,h0):
        h=self.act1(torch.matmul(input,self.U)+torch.matmul(h0,self.W)+self.b1+self.b2)
        output=self.act2(torch.matmul(h,self.V))
        return output,h
class RNN(RNNCell):
    def __init__(self,input_dim,output_dim):
        RNNCell.__init__(self,input_dim,output_dim)
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.rnncell=RNNCell(self.input_dim,self.output_dim)
    def forward(self,input):
        b,l,h=input.shape
        h0=torch.zeros(size=(b,self.output_dim),dtype=torch.float32)
        input1=input[:,0,:]
        out1,h1=self.rnncell(input1,h0)
        output=[]
        output.append(out1)
        for i in range(l-1):
            out,h2=self.rnncell(input[:,i+1,:],h1)
            h1=h2
            output.append(out)
        output=torch.stack([i for i in output]).permute(1,0,2)
        return output,h1
if __name__=='__main__':
    input=torch.randint(0,1,size=(5,3,2),dtype=torch.float32) #batch size=5,seq=3,dim=2
    output=torch.randint(0,1,size=(5,3,4),dtype=torch.float32) #batch size=5,seq=3,dim=4
    rnn=RNN(input_dim=2,output_dim=4)
    optim=torch.optim.Adam(params=rnn.parameters())
    Loss=nn.MSELoss()
    out=rnn(input)
    for i in range(100):
        yp=rnn(input)[0]
        loss=Loss(yp,output)
        optim.zero_grad()
        loss.backward()
        optim.step()
        print(loss)
'''