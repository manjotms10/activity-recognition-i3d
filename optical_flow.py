import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import sys
cv2 = cv

def process_data(path_to_video):
	cap = cv.VideoCapture(path_to_video)
	ret, frame1 = cap.read()
	frame1 = cv2.resize(frame1, (224, 224))
	length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

	frame_width = int(cap.get(3))
	frame_height = int(cap.get(4))
	 
	# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
	out = cv2.VideoWriter('flow.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (224,224))

	prvs = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
	hsv = np.zeros_like(frame1)
	hsv[...,1] = 255
	i=1

	rgbframes = []
	flowframes = []

	while(i < 80):
	    i += 1
	    ret, frame2 = cap.read()
	    frame2 = cv2.resize(frame2, (224, 224))
	    rgbframes.append(frame2)
	    next = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)
	    flow = cv.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
	    flowframes.append(flow)
	    mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
	    hsv[...,0] = ang*180/np.pi/2
	    hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
	    bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
	    out.write(bgr)
	    prvs = next

	frames = np.expand_dims(np.array(rgbframes), 0)
	flow = np.expand_dims(np.array(flowframes), 0)
	cap.release()
	out.release()

	np.save('flow.npy', flow)
	np.save('frames.npy', frames)
	print("Done!")

if __name__=='__main__':
    if len(sys.argv)>1:
        process_data(sys.argv[1])
    else:
    	print("Usage: python optical_flow.py path_to_video")


