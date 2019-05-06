import os
import numpy as np
import types
import torch.nn as nn
from PIL import Image
from PIL import ImageDraw
from estimator import ResEstimator
from networks import *
from network import CoordRegressionNetwork


def postprocess(self, keypoints: torch.Tensor, input_img: Image, visualize: bool = False):
    """Converts pytorch tensor into PIL Image

    Handles all the steps for postprocessing of the raw output of the model.
    Depending on the rocket family there might be additional options.

    Args:
        x (Tensor): Output Tensor to postprocess
    """
    img = np.array(input_img)
    img_height, img_width, _ = img.shape

    # The amount of padding that was added
    pad_x = max(img_height - img_width, 0) * (224 / max(img.shape))
    pad_y = max(img_width - img_height, 0) * (224 / max(img.shape))
    # Image height and width after padding is removed
    unpad_h = 224 - pad_y
    unpad_w = 224 - pad_x

    keypoints = keypoints[0].detach().numpy()
    keypoints = (((keypoints.reshape([-1,2])+np.array([1.0,1.0]))/2.0*np.array((224, 224))-[pad_x, pad_y]) * 1.0 /np.array([unpad_w, unpad_h])*np.array([img_width,img_height]))

    if visualize:
        image_h, image_w = img.shape[:2]
        centers = {}

        colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255]]

        pairs = [[8,9],[11,12],[11,10],[2,1],[1,0],[13,14],[14,15],[3,4],[4,5],[8,7],[7,6],[6,2],[6,3],[8,12],[8,13]]
        colors_skeleton = ['r', 'y', 'y', 'g', 'g', 'y', 'y', 'g', 'g', 'm', 'm', 'g', 'g', 'y','y']
        colors_skeleton = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255]]

        for idx in range(len(colors)):
            cv2.circle(img, (keypoints[idx,0], keypoints[idx,1]), 3, colors[idx], thickness=3, lineType=8, shift=0)
        for idx in range(len(colors_skeleton)):
            img = cv2.line(img, (keypoints[pairs[idx][0],0], keypoints[pairs[idx][0],1]), (keypoints[pairs[idx][1],0], keypoints[pairs[idx][1],1]), colors_skeleton[idx], 3)

        return Image.fromarray(npimg)
    return keypoints


def preprocess(self, img: Image) -> torch.Tensor, np.Array:
    """Converts PIL Image or Array into pytorch tensor specific to this model

    Handles all the necessary steps for preprocessing such as resizing, normalization.
    Works with both single images and list/batch of images. Input image file is expected
    to be a `PIL.Image` object with 3 color channels.
    Labels must have the following format: `x1, y1, x2, y2, category_id`

    Args:
        img (PIL.Image): input image
        labels (list): list of bounding boxes and class labels
    """

    h, w, c = img.shape
    dim_diff = np.abs(h - w)
    # Upper (left) and lower (right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2

    # Determine padding
    pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
    # Add padding
    input_img = np.pad(img, pad, 'constant', constant_values=128) / 255.
    padded_h, padded_w, _ = input_img.shape

    # Resize and normalize
    input_img = Image.fromarray(np.uint8(input_img*255), 'RGB')
    input_img.thumbnail((224, 224), resample=Image.BICUBIC)
    input_img = np.array(input_img)
    input_img = input_img / 255.0

    # Channels-first
    input_img = np.transpose(input_img, (2, 0, 1))
    # As pytorch tensor
    input_img = torch.from_numpy(input_img).float()
    return input_img


def build():
    model_path = os.path.join(os.path.realpath(os.path.dirname(__file__)), "resnet18_224.t7")
    net = CoordRegressionNetwork(n_locations=16, backbone="resnet18").to("cpu")
    model = ResEstimator(model_path, net, 224)

    model.postprocess = types.MethodType(postprocess, model)
    model.preprocess = types.MethodType(preprocess, model)

    return model