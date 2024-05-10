# TetrisNet: Tetris Kernels for Sketch Recognition and Beyond



This is the official implementation of "TetrisNet: Tetris Kernels for Sketch Recognition and Beyond"

## Folder Overview

**CAN_yang**: Codes and data for Handwritten Mathematical Expression Recognition on CROHME dataset.

**Swin-Transformer-Object-Detection**: Codes and data for Instance Segmentation on COCO dataset.

**MMSegmentation:** Codes and data for vessel segmentation on DRIVE dataset.

**TetrisNet:** Codes and data for sketch recognition on QuickDraw-414k and TU-Berlin datasets.

## Supported Datasets



-  QuickDraw-414k
-  Tuberlin
-  CIFAR

## Installation



### Prerequisites



The code is built with following libraries:

- Python >= 3.6, <3.8
- [PyTorch](https://github.com/pytorch/pytorch) >= 1.6
- [tqdm](https://github.com/tqdm/tqdm)
- [torchpack](https://github.com/mit-han-lab/torchpack)
- [torchsparse](https://github.com/mit-han-lab/torchsparse)
- [numba](http://numba.pydata.org/)
- [cv2](https://github.com/opencv/opencv)

## Training



You can modify the config (e.g. ***configs/swin_image.yaml***) to choose or define your model for training

Supported Distributed Training

```
CUDA_VISIBLE_DEVICES=0,1,2,3 torchpack dist-run -np 4 python train_img_single.py configs/swin_image.yaml --run-dir nbenchmark/swin_sce # 4 gpus
```



```
CUDA_VISIBLE_DEVICES=2,3 torchpack dist-run -np 2 python train_img2.py configs/quickdraw/sd3b1_image_stroke.yaml --run-dir nbenchmark/trans/resnet50_quickdraw_image_stroke_sd3b1_norm/
```



Single GPU Training

```
python train_img_single.py configs/swin_image.yaml --run-dir nbenchmark/swin_sce --distributed False
```



```
python train_img2.py configs/quickdraw/sd3b1_image_stroke.yaml --run-dir nbenchmark/trans/resnet50_quickdraw_image_stroke_sd3b1_norm/ --distributed False
```



# Citation



> 

# Issues



If you have any problems, feel free to reach out to me in the issue.
