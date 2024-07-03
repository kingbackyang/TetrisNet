CUDA_VISIBLE_DEVICES=6 python tools/test.py configs/swin/cascade_mask_rcnn_swin_small_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_cocov2_conv.py tetrisnet_bench/conv/epoch_1.pth --out out.pkl --eval bbox segm --show-dir tetrisnet_bench/conv/vis

CUDA_VISIBLE_DEVICES=6 python tools/test.py configs/swin/cascade_mask_rcnn_swin_small_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_cocov2_dcn.py tetrisnet_bench/dcn/epoch_1.pth --out out.pkl --eval bbox segm --show-dir tetrisnet_bench/dcn/vis

CUDA_VISIBLE_DEVICES=6 python tools/test.py configs/swin/cascade_mask_rcnn_swin_small_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_cocov2_dsc.py tetrisnet_bench/dsc/epoch_1.pth --out out.pkl --eval bbox segm --show-dir tetrisnet_bench/dsc/vis

CUDA_VISIBLE_DEVICES=6 python tools/test.py configs/swin/cascade_mask_rcnn_swin_small_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_cocov2_scc.py tetrisnet_bench/scc/epoch_7.pth --out out.pkl --eval bbox segm --show-dir tetrisnet_bench/scc/vis

CUDA_VISIBLE_DEVICES=6 python tools/test.py configs/swin/cascade_mask_rcnn_swin_small_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_cocov2_vit.py tetrisnet_bench/vit/epoch_1.pth --out out.pkl --eval bbox segm --show-dir tetrisnet_bench/vit/vis

CUDA_VISIBLE_DEVICES=6 python tools/test.py configs/swin/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py cascade_mask_rcnn_swin_tiny_patch4_window7.pth --out out.pkl --eval bbox segm --show-dir tetrisnet_bench/ori/vis







