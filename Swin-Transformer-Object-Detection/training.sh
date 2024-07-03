CUDA_VISIBLE_DEVICES=0,1,3,4 ./tools/dist_train.sh configs/swin/cascade_mask_rcnn_swin_small_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_cocov2.py 4 --work-dir tetrisnet_bench/tetrisnet
