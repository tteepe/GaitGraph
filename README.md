# GaitGraph
This repository contains the PyTorch code for:

__GaitGraph: Graph Convolutional Network for Skeleton-Based Gait Recognition__

[Torben Teepe](https://github.com/tteepe), Ali Khan, Johannes Gilg, [Fabian Herzog](https://github.com/fubel),
Stefan Hörmann 

[[`arXiv`](https://arxiv.org/abs/2101.11228)] [[`BibTeX`](#CitingGaitGraph)]

![Pipeline](images/pipeline.png)

## Quick Start

### Prerequisites
- Python >= 3.6
- CUDA >= 10

First, create a virtual environment or install dependencies directly with:
```shell
pip3 install -r requirements.txt
```

### Data preparation
The extraction of the pose data from CASIA-B can either run the commands bellow or download the preprocessed data using:
```shell
cd data
sh ./download_data.sh
```

Optional:
If you choose to run the preprocessing, [download](http://www.cbsr.ia.ac.cn/english/Gait%20Databases.asp) the dataset and run the following commands.
```shell
# Download required weights
cd models
sh ./download_weights.sh

# <PATH_TO_CASIA-B> should be something like: /home/ ... /datasets/CASIA/casia-b/all_frames/
cd <PATH_TO_CASIA-B>
find . -type f -regex ".*\.jpg" > casia-b_all_frames.csv
cp casia-b_all_frames.csv <PATH_TO_REPO>/data

cd <PATH_TO_REPO>/src
export PYTHONPATH=${PWD}:$PYTHONPATH

cd preparation
python3 prepare_detection.py <PATH_TO_CASIA-B> ../../data/casia-b_all_frames.csv ../../data/casia-b_detections.csv
python3 prepare_pose_estimation.py  <PATH_TO_CASIA-B> ../../data/casia-b_detections.csv ../../data/casia-b_pose_coco.csv
python3 split_casia-b.py ../../data/casia-b_pose_coco.csv --output_dir ../../data
```

### Train
To train the model you can run the `train.py` script. To see all options run:
```shell
cd src
export PYTHONPATH=${PWD}:$PYTHONPATH

python3 train.py --help
```

Check `experiments/1_train_*.sh` to see the configurations used in the paper. 

Optionally start the tensorboard with: 
```shell
tensorboard --logdir=save/casia-b_tensorboard 
```

### Evaluation
Evaluate the models using `evaluate.py` script. To see all options run:
```shell
python3 evaluate.py --help
```


## Main Results
Top-1 Accuracy per probe angle excluding identical-view cases for the provided models on 
[CASIA-B](http://www.cbsr.ia.ac.cn/english/Gait%20Databases.asp) dataset.

|        |    0 |   18 |   36 |   54 |   72 |   90 |   108 |   126 |   144 |   162 |   180 |   mean |
|:-------|-----:|-----:|-----:|-----:|-----:|-----:|------:|------:|------:|------:|------:|-------:|
| NM#5-6 | 85.3 | 88.5 | 91   | 92.5 | 87.2 | 86.5 |  88.4 |  89.2 |  87.9 |  85.9 |  81.9 |   87.7 |
| BG#1-2 | 75.8 | 76.7 | 75.9 | 76.1 | 71.4 | 73.9 |  78   |  74.7 |  75.4 |  75.4 |  69.2 |   74.8 |
| CL#1-2 | 69.6 | 66.1 | 68.8 | 67.2 | 64.5 | 62   |  69.5 |  65.6 |  65.7 |  66.1 |  64.3 |   66.3 |

The pre-trained model is available [here](https://github.com/tteepe/GaitGraph/releases/tag/v0.1).

## Licence & Acknowledgement
GaitPose itself is released under the MIT License (see LICENSE).

The following parts of the code are borrowed from other projects. Thanks for their wonderful work!
- Object Detector: [eriklindernoren/PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3)
- Pose Estimator: [HRNet/HRNet-Human-Pose-Estimation](https://github.com/HRNet/HRNet-Human-Pose-Estimation)
- ST-GCN Model: [yysijie/st-gcn](https://github.com/yysijie/st-gcn)
- ResGCNv1 Model: [yfsong0709/ResGCNv1](https://github.com/yfsong0709/ResGCNv1)
- SupCon Loss: [HobbitLong/SupContrast](https://github.com/HobbitLong/SupContrast)

## <a name="CitingGaitGraph"></a>Citing GaitGraph
If you use Densepose, please use the following BibTeX entry.

```
@misc{teepe2021gaitgraph,
      title={GaitGraph: Graph Convolutional Network for Skeleton-Based Gait Recognition}, 
      author={Torben Teepe and Ali Khan and Johannes Gilg and Fabian Herzog and Stefan H\"ormann and Gerhard Rigoll},
      year={2021},
      eprint={2101.11228},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
