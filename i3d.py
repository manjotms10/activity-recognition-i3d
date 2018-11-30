import torch
import numpy as np
from torch.autograd import Variable
from model.I3D_Pytorch import I3D

model = I3D(input_channel=3)

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
    'rgb': 'data/MSR/a01_s01_e02_rgb_frames.npy',
    'flow': 'data/MSR/a01_s01_e02_rgb_flow.npy',
}

state_dict = torch.load(_CHECKPOINT_PATHS['rgb_imagenet'])
model.load_state_dict(state_dict)
print('RGB checkpoint restored')

#for name, param in model.named_parameters():
    #print name, param.requires_grad
    
print(model.features[-1])
model.features[-1] = nn.Conv3d(1024, 16, kernel_size=(1, 1, 1), stride=(1, 1, 1))

rgb_np = np.load(_SAMPLE_PATHS['rgb'])
rgb_sample = torch.from_numpy(rgb_np)
rgb_sample = Variable(rgb_sample.permute(0, 4, 1, 2 ,3))
print('RGB data loaded, shape=' + str(rgb_sample.data.size()))
rbg_score, rgb_logits = model(rgb_sample)

print(rbg_score)