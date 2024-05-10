# TetrisNet: Tetris Kernels for Sketch Recognition and Beyond



This is the official implementation of "TetrisNet: Tetris Kernels for Sketch Recognition and Beyond"

## Folder Overview



**CAN_yang**: Codes and data for Handwritten Mathematical Expression Recognition on CROHME dataset.

**Swin-Transformer-Object-Detection**: Codes and data for Instance Segmentation on COCO dataset.

**MMSegmentation:** Codes and data for vessel segmentation on DRIVE dataset.

**TetrisNet:** Codes and data for sketch recognition on QuickDraw-414k and TU-Berlin datasets.

### Prerequisites

The code is built with following libraries:

- Python >= 3.6, <3.8
- [PyTorch](https://github.com/pytorch/pytorch) >= 1.6
- [tqdm](https://github.com/tqdm/tqdm)
- [torchpack](https://github.com/mit-han-lab/torchpack)
- [torchsparse](https://github.com/mit-han-lab/torchsparse)
- [numba](http://numba.pydata.org/)
- [cv2](https://github.com/opencv/opencv)
- mmcv-full==1.4.0
- [spconv](https://github.com/traveller59/spconv)

## CAN_yang

### Datasets

Download the CROHME dataset from [BaiduYun](https://pan.baidu.com/s/1qUVQLZh5aPT6d7-m6il6Rg) (downloading code: 1234) and put it in `datasets/`.

The HME100K dataset can be download from the official website [HME100K](https://ai.100tal.com/dataset).

### Training

Check the config file `config.yaml` and train with the CROHME dataset:

```
python train.py --dataset CROHME
```

By default the `batch size` is set to 8 and you may need to use a GPU with 32GB RAM to train your model.

### Testing

Fill in the `checkpoint` (pretrained model path) in the config file `config.yaml` and test with the CROHME dataset:

```
python inference.py --dataset CROHME
```

Note that the testing dataset path is set in the `inference.py`.

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