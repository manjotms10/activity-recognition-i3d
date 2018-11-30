
# coding: utf-8

# In[1]:


import torch
import numpy as np
from torch.autograd import Variable
from model.I3D_Pytorch import I3D
import torch.nn as nn
import torch.nn.functional as F
import glob
import matplotlib.pyplot as plt
import itertools


# In[2]:


_CHECKPOINT_PATHS = {
    'rgb': 'data/pytorch_checkpoints/rgb_scratch.pkl',
    'flow': 'data/pytorch_checkpoints/flow_scratch.pkl',
    'rgb_imagenet': 'data/pytorch_checkpoints/rgb_imagenet.pkl',
    'flow_imagenet': 'data/pytorch_checkpoints/flow_imagenet.pkl',
}

_IMAGE_SIZE = 224
_NUM_CLASSES = 400

_SAMPLE_VIDEO_FRAMES = 79
_SAMPLE_PATHS = {
    'rgb': 'data/MSR_data/a01_s01_e02_rgb_frames.npy',
    'flow': 'data/MSR_data/a01_s01_e02_rgb_flow.npy',
}


# In[3]:


model = I3D(input_channel=3)
model.eval()
state_dict = torch.load(_CHECKPOINT_PATHS['rgb_imagenet'])
model.load_state_dict(state_dict)
print('RGB checkpoint restored')


# In[4]:


flow_i3d = I3D(input_channel=2)
flow_i3d.eval()
state_dict = torch.load(_CHECKPOINT_PATHS['flow_imagenet'])
flow_i3d.load_state_dict(state_dict)
print('FLOW checkpoint restored')


# In[5]:


for name, param in model.named_parameters():
    param.requires_grad = False
    #print name, param.requires_grad
    


# In[6]:


for name, param in flow_i3d.named_parameters():
    param.requires_grad = False
    #print name, param.requires_grad


# In[7]:


model.features[-1] = nn.Conv3d(1024, 16, kernel_size=(1, 1, 1), stride=(1, 1, 1))
model.features[-1].requires_grad = True
flow_i3d.features[-1] = nn.Conv3d(1024, 16, kernel_size=(1, 1, 1), stride=(1, 1, 1))
flow_i3d.features[-1].requires_grad = True

print(model)


# In[8]:


def data_loader(batch_size=8):
    
    files = glob.glob('./data/MSR_data/*frames.npy')
    flow = glob.glob('./data/MSR_data/*flow.npy')
    print(flow[0], len(flow))
    while True:
        x_train, y_train, z_train = [], [], []
        idx = np.random.choice(range(320), batch_size)
        for i in idx:
            npx = np.load(files[i])
            npz = np.load(flow[i])
            if npx.shape[1]==79 and npz.shape[1]==79:
                x_train.append(npx)
                y_train.append(int(str(files[i].split('/')[-1]).split('_')[0][1:]) - 1)
                z_train.append(npz)
            else:
                pass
                #print("Code fata")
        
        x_train = np.array(x_train)
        x_train = np.reshape(x_train, (-1, 79, 224, 224, 3))
        rgb_sample = torch.from_numpy(x_train)
        rgb_sample = Variable(rgb_sample.permute(0, 4, 1, 2 ,3))
        
        z_train = np.array(z_train)
        z_train = np.reshape(z_train, (-1, 79, 224, 224, 2))
        flow_sample = torch.from_numpy(z_train)
        flow_sample = Variable(flow_sample.permute(0, 4, 1, 2 ,3))
        
        y = torch.from_numpy(np.array(y_train))
        yield rgb_sample, y, flow_sample


# In[9]:


data_gen = data_loader(32)


# In[11]:


model = nn.DataParallel(model, range(2))
flow_i3d = nn.DataParallel(flow_i3d, range(2))

model = model.cuda()
flow_i3d = flow_i3d.cuda()


# In[12]:


optim = torch.optim.Adam(itertools.chain(model.parameters(), flow_i3d.parameters()), 0.0002, betas=(0.99, 0.5))
loss_fn = nn.CrossEntropyLoss()
cost = []


# In[ ]:


num_epochs = 3000
for e in range(num_epochs):
    x, y, z = next(data_gen)
    xtrain, ytrain, ztrain = Variable(x), Variable(y), Variable(z)
    xtrain, ytrain, ztrain = xtrain.cuda(), ytrain.cuda(), ztrain.cuda()
    score_rgb, logits_rgb = model(xtrain)
    score_flow, logits_flow = flow_i3d(ztrain)
    
    ypred = (logits_rgb + logits_flow).squeeze(0)
    ypred = F.softmax(ypred)
    
    optim.zero_grad()
    
    #print(torch.argmax(ypred, 1))
    loss = loss_fn(ypred, Variable(ytrain))
    #loss = Variable(loss, requires_grad = True)
    
    cost.append(loss.item())
    
    print("Epoch [%d / %d] Loss: %.3f"%(e, num_epochs, loss.item()))
    
    loss.backward()
    optim.step()


# In[ ]:


model.save_state_dict('rgb.pt')
flow_i3d.save_state_dict('flow.pt')

