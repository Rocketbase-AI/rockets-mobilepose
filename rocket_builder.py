import os
import numpy as np
import types
import torch.nn as nn
import cv2
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
    img = img[: ,:, ::-1]
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
            cv2.circle(img, (keypoints[idx,0].astype(int), keypoints[idx,1].astype(int)), 3, colors[idx], thickness=3, lineType=8, shift=0)
        for idx in range(len(colors_skeleton)):
            img = cv2.line(img, (keypoints[pairs[idx][0],0].astype(int), keypoints[pairs[idx][0],1].astype(int)), (keypoints[pairs[idx][1],0].astype(int), keypoints[pairs[idx][1],1].astype(int)), colors_skeleton[idx], 3)
        #img = img[: ,:, ::-1]
        return Image.fromarray(img.get())
    return keypoints


def preprocess(self, img: Image) -> torch.Tensor:
    """Converts PIL Image or Array into pytorch tensor specific to this model

    Handles all the necessary steps for preprocessing such as resizing, normalization.
    Works with both single images and list/batch of images. Input image file is expected
    to be a `PIL.Image` object with 3 color channels.
    Labels must have the following format: `x1, y1, x2, y2, category_id`

    Args:
        img (PIL.Image): input image
        labels (list): list of bounding boxes and class labels
    """
    image = np.array(img)
    image = image[: ,:, ::-1]
    image_ = image/256.0
    h, w = image_.shape[:2]
    im_scale = min(float(224) / float(h), float(224) / float(w))
    new_h = int(image_.shape[0] * im_scale)
    new_w = int(image_.shape[1] * im_scale)
    image = cv2.resize(image_, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    left_pad =int( (224 - new_w) / 2.0)
    top_pad = int((224 - new_h) / 2.0)
    mean=np.array([0.485, 0.456, 0.406])
    pad = ((top_pad, top_pad), (left_pad, left_pad))
    image = np.stack([np.pad(image[:,:,c], pad, mode='constant', constant_values=mean[c])for c in range(3)], axis=2)
    # As pytorch tensor
    mean=np.array([0.485, 0.456, 0.406])
    std=np.array([0.229, 0.224, 0.225])    
    image = torch.from_numpy(((image-mean)/std).transpose((2, 0, 1))).float()
    image = image.unsqueeze(0)
    return image


def build():
    model_path = os.path.join(os.path.realpath(os.path.dirname(__file__)), "resnet18_224.t7")
    model = CoordRegressionNetwork(n_locations=16, backbone="resnet18").to("cpu")

    model.postprocess = types.MethodType(postprocess, model)
    model.preprocess = types.MethodType(preprocess, model)

    return model