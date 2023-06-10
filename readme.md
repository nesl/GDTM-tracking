

# GDTM-Tracking

**GDTM** is a new multi-hour dataset collected with a network of multimodal sensors for the indoor geospatial tracking problem. It features time-synchronized steoreo-vision camera, LiDAR camera, mmWave radar, and microphone arrays, as well as ground truth data containing the position and orientations of the sensing target (remote controlled cars on a indoor race track) and the sensor nodes. For details of the dataset please refer to [GitHub](https://github.com/nesl/GDTM) and PDF (Still under review).

This repository contains our baseline applications described in PDF (Still under review) built to process the GTDM data. It features two architectures (early fusion and late fusion and two choices of sensor sets (camera only and all-modalities) to track the locations of a target RC car.


## Installation Instuctions

### Environment
The code is tested with:
Ubuntu 20.04
Anaconda 22.9.0 (for virtual python environment)
NVIDIA-driver 525.105.17
The code should be compatible with most Anaconda, NVIDIA-driver, and Ubuntu versions available around 2023/06.

### Code Repository Structure
Please select the desired branch for your desired data/model combinations. Details described in **Baseline 1** section of PDF (Still under review).

 - master: early fusion, all modalities 
 - early-cam: early fusion, camera only 
 - late-cam: late fusion, camera only
 - late-all: late fusion, allmodalities

Additional, this repository contains our efforts towards building a model resilient to the placement locations and orientations of its sensor nodes.  Details described in **Baseline 2** section of PDF (Still under review).
- multi-cam: late fusion 3D, camera only, multiple viewpoints (limited tracking performance)
- multi-camdepth: late fusion 3D, camera + LiDAR camera depth, multiple viewpoints

As step one, please clone the desired branch using terminal
```
cd ~/Desktop
git clone https://github.com/nesl/GDTM-tracking.git
```
or
```
cd ~/Desktop
git clone --branch <branchname> https://github.com/nesl/GDTM-tracking.git
```

### Install Dependencies
First, place the repository folder on Desktop and rename it "mmtracking".
```
mv <path-to-cloned-repository> ~/Desktop/mmtracking
```
Create a new conda environment using
```
cd ~/Desktop/mmtracking
conda create -n iobt python=3.9
conda activate iobt
```
Install a few torch and mmcv using pip:
```
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.0/index.html
```
Install other dependencies:
```
pip  install  -r  requirements/build.txt
```
Install a few local packages (the terminal should still be in ~/Desktop/mmtracking):
```
pip install -e src/cad
pip install -e src/mmdetection
pip install -e src/TrackEval
pip install -e src/mmclassification
pip install -e src/resource_constrained_tracking
pip install -e src/yolov7
pip install  -v  -e  .
```

## Data Preparation
#### Sample dataset
Please visit the [data repository](https://drive.google.com/drive/folders/1oj75d7x11Y51g2lkLlLWog-bJZGKDPgG?usp=sharing) for a sample data to test this repository.

Note that the configuration of the data need to be the same as the branch of your choice. For example, the late-all branch code is configured as (single view, late fusion, all modalities). If you would like to test this code in good lighting condition data, you should go to
["SampleData/SingleView/Good Lighting Conditions/dataset_singleview3.zip"](https://drive.google.com/file/d/1c72LTUa9Uz90r_2iO9LBxz9bHoS-V-fi/view?usp=sharing)

#### Full dataset
We are going to release the full dataset on a later date. Check for updates at [GitHub](https://github.com/nesl/GDTM).

#### Unzip the data

Please unzip the data, rename it to **"mcp-sample-dataset/"** and put it on the Desktop.
The final data structure should be like following:
```
└── Desktop/mcp-sample-dataset/
	├── test/
	│   ├── node1/       
	│   │   ├── mmwave.hdf5
	│   │   ├── realsense.hdf5
	│   │   ├── respeaker.hdf5
	│   │   └── zed.hdf5
	│   ├── node2/
	│   │   └── same as node 1
	│   ├── node3/
	│   │   └── same as node 1  
	│   └── mocap.hdf5
	├── train/
	│   └── same sa test
	└── val/
	    └── same as test
```
Note that you only need test/ if you are running test from checkpoints only.
#### Specify Filepath
Open mmtracking/configs/\_base\_/datasets/ucla_sample_1car.py
In Line 75, Line 114, and Line 153, change the data_root to absolute path:
e.g. ~/Desktop/... -> /home/USER_NAME/Desktop/...

## Code Usage
### Make Inference Using A Checkpoint (Testing)
Please download the pretrained checkpoints [Here](https://drive.google.com/drive/folders/1Bm6PZZYlW0uiAYI7mUdmd0P-SEFDTnvq?usp=sharing).

Note that for single-view case (Baseline 1 in the paper), please make sure to use the checkpoints corresponding to the code and data of your choice. 

For example, if we use view 3 data (single view, good lighting condition) and late-all branch code (single view, late fusion, all modalities), we should download
["Checkpoints/Single-view/Late Fusion/All Modalities/logs_sal_goodlight_view3.zip"](https://drive.google.com/file/d/16JjY2d2J0JnWJNI1Mpy7om7m1MKzF8Qj/view?usp=sharing)

After downloading the checkpoint, please rename it to **logs/** and put it under "mmtracking" folder using this hierachy. 

```
└── Desktop/mmtracking/
    └── logs/
        ├── val
        ├── epoch_xx.pth
        └── latest.pth
```
where the "latest.pth" above is created by
```
ln -s epoch_40.pth latest.pth
```

Then, you could run the evaluations by running (still in terminal under ~/Desktop/mmtracking)
```
bash ./tools/test_from_config_nll_local.sh ./configs/mocap/ucla_sample_1car_zed_nodes123_r50.py 1
```
---
**Warning**: This script will cache the dataset in system memory (/dev/shm)
If the dataset loading operation was not successful, or you have changed the dataset in "~/Desktop/mcp-sample-dataset", please make sure to run this line **before** the "test_from_config_nll_local.sh" above:
```
rm -r /dev/shm/cache_*
```
---


The visualization results will apprear in 
```
mmtracking/logs/ucla_sample_1car_zed_nodes123_r50/test_nll/latest_vid.mp4
```
and numerical results appears at the last two lines of 
```
mmtracking/logs/ucla_sample_1car_zed_nodes123_r50/test_nll/mean.txt
```

If you would like to train a model from scratch instead , please refer to the “training” and “scaling” sections down below.

### Training
Set up the data as instructed by previous sections, and run
```
bash ./tools/train_from_config_local.sh ./configs/mocap/ucla_sample_1car_zed_nodes123_r50.py 1
```
where the last digit indicate the number of GPU you have for training.

### Scaling
After training, some additional data is required to perform a post-hoc model recalibration as described in the paper to better capture model prediction uncertainties. More specifically, We apply an affine transformation Σ′ = aΣ + bI to the output covariance matrix Σ with parameters a and b that minimize the calibration data’s NLL. 

Instructions for scaling:
```
bash ./tools/val_from_config_local.sh ./configs/mocap/ucla_sample_1car_zed_nodes123_r50.py 1
```
The last digit must be "1". Scaling with multiple GPU will cause an error.


## Troubleshooting

Here we list a few files to change in case some error happens during your configurations.
#### Data not found error
This is where the filepath are stored
mmtracking/configs/\_base\_/datasets/ucla_sample_1car.py

Don't forget to do "rm -r /dev/shm/cache_*" after you fix this error. Otherwise a "List out of range" error will pop up.

#### GPU OOM Error, Number of Epoches, Inteval of checkpoints
mmtracking/configs/mocap/ucla_sample_1car_zed_nodes123_r50.py
Reduce "samples_per_gpu" near Line 161 helps with OOM error.
Line 170+ changes the training configurations.

This configuration also defines (1) the valid modalities (2) backbone, adapter, and output head architecture hyperparameters

#### Something wrong with dataset caching
mmtracking/mmtrack/datasets/mocap/cacher.py

#### Something wrong with model training/inferences
mmtracking/mmtrack/models/mocap/kfdetr.py
Function forward_train() for training
Fuction forward_track() for testing

#### Something wrong with final visualzations
mmtracking/mmtrack/datasets/mocap/hdf5_dataset.py
in function write_videos()

#### Backbone definitions
mmtracking/mmtrack/models/backbones/tv_r50.py

## Citation and Acknowledgements

If you find this project useful in your research, please consider cite:
```
@inproceedings{wang2023gdtm,
    title={GTDM: An Indoor Geospatial Tracking Dataset with Distributed Multimodal Sensors},
    author={Jeong, Ho Lyun and Wang, Ziqi and Samplawski, Colin and Wu, Jason and Fang, Shiwei and Kaplan, Lance and Ganesan, Deepak and Marlin, Benjamin and Srivastava, Mani},
    booktitle={submission to the Thirty-seventh Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
    year={2023}
}
```
The architecture of this work is adapted from/inspired by:

```
@inproceedings{samplawski2020heteroskedastic,
  title={Heteroskedastic Geospatial Tracking with Distributed Camera Networks},
  author={Samplawski, Colin and Fang, Shiwei and Wang, Ziqi and Ganesan, Deepak and Srivastava, Mani and Marlin, Benjamin},
  booktitle={Uncertainty in Artificial Intelligence},
  year={2023},
  organization={PMLR}
}

@misc{mmtrack2020,
    title={{MMTracking: OpenMMLab} video perception toolbox and benchmark},
    author={MMTracking Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmtracking}},
    year={2020}
}
```

