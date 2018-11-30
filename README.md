# I3D models trained on Kinetics Pytorch

this repo implements the network of I3D with Pytorch, pre-trained model weights are converted from tensorflow. 

### Sample code

you can evaluate sample 

```
python evaluate.py

```

The sample video can be found in /data.
Use optical_flow.py to preprocess data to fed for inference

```
python optical_flow.py path_to_video
```
This will create two .npy files namely flow.npy and frames.npy, which can be used by the model to classify the video.

The original labels from the Kinetics dataset are in /data.

Reference:

[kinetics-i3d](https://github.com/deepmind/kinetics-i3d)  
[tensorflow-model-zoo.torch](https://github.com/Cadene/tensorflow-model-zoo.torch)
