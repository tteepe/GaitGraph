#!/bin/bash
# Download weights for vanilla YOLOv3
curl -O https://pjreddie.com/media/files/yolov3.weights
# Download weights for tiny YOLOv3
curl -O https://pjreddie.com/media/files/yolov3-tiny.weights
## Download weights for backbone network
#curl -O https://pjreddie.com/media/files/darknet53.conv.74

# Download pre-trained weights
curl -LJO https://github.com/tteepe/GaitGraph/releases/download/v0.1/gaitgraph_resgcn-n39-r8_coco_seq_60.pth

print "#############################################################"
print "######## Weights for HRNet Pose Estimation need to ##########"
print "######## be downloaded manually from here:         ##########"
print "######## https://drive.google.com/drive/folders/1nzM_OBV9LbAEA7HClC0chEyf_7ECDXYA"
print "######## Files: pose_hrnet_*.pth                   ##########"
print "#############################################################"
