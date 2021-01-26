#!/bin/bash
# Download weights for vanilla YOLOv3
wget -c https://pjreddie.com/media/files/yolov3.weights
# Download weights for tiny YOLOv3
wget -c https://pjreddie.com/media/files/yolov3-tiny.weights
## Download weights for backbone network
#wget -c https://pjreddie.com/media/files/darknet53.conv.74

print "#############################################################"
print "######## Weights for HRNet Pose Estimation need to ##########"
print "######## be downloaded manually from here:         ##########"
print "######## https://drive.google.com/drive/folders/1nzM_OBV9LbAEA7HClC0chEyf_7ECDXYA"
print "######## Files: pose_hrnet_*.pth                   ##########"
print "#############################################################"
