import cv2 as cv
import numpy as np
import torch
import glob
from mvn.models.pose_resnet import PoseResNet, Bottleneck, BasicBlock
from mvn.utils.cfg import load_config

resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}

if __name__ == '__main__':
    # quick and dirty hardcoded inference

    # step 0 load the sample data
    image_names = glob.glob('/home/colin/panoptic/images/*.png')
    images = [cv.imread(image_name) for image_name in image_names]
    # step 1 load the 2D backbone model and do inference on the images
    config = load_config('/home/colin/vpose/learnable-triangulation/pytorch/experiments/human36m/eval/test_vol.yaml')
    num_layers = 152
    resnet_weight_path = '/home/colin/vpose/pose_resnet_4.5_pixels_human36m.pth'
    resnet_weights = torch.load(resnet_weight_path, map_location=torch.device('cpu'))
    block_class, layers = resnet_spec[num_layers]
    model_2d = PoseResNet(block_class, layers, 17,
                          num_input_channels=3,
                          deconv_with_bias=False,
                          num_deconv_layers=3,
                          num_deconv_filters=(256, 256, 256),
                          num_deconv_kernels=(4, 4, 4),
                          final_conv_kernel=1
                          )
    model_2d.load_state_dict(resnet_weights)
    print('here')
