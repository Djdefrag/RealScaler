import collections
import ctypes
import math
import multiprocessing
import os
import os.path
import platform
import queue
import shutil
import sys
import threading
import time
import tkinter as tk
import tkinter.font as tkFont
import warnings
import webbrowser
from itertools import repeat
from timeit import default_timer as timer
from tkinter import PhotoImage, ttk

import cv2
import numpy as np
import sv_ttk
import tkinterDnD
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from moviepy.audio.AudioClip import CompositeAudioClip
from moviepy.audio.io import AudioFileClip
from moviepy.video.io import ImageSequenceClip, VideoFileClip
from PIL import Image, ImageDraw, ImageFont
from split_image import reverse_split, split_image
from win32mica import MICAMODE, ApplyMica

pay      = True
version  = "v. 6.0"

if not pay:
    version = version + ".f"

global app_name
app_name = "ReSR.Scaler"

windows_subversion    = int(platform.version().split('.')[2])

image_path            = "no file"
AI_model              = "RealESRGAN_x4plus"
device                = "dml"
input_video_path      = ""
tiles_resolution      = 700
single_file           = False
multiple_files        = False
video_files           = False
multi_img_list        = []
video_frames_list     = []
video_frames_upscaled_list = []
target_file_extension = ".png"
half_precision        = True
denoise_strength      = 1
multiplier_num_tiles  = 3
resize_algorithm      = Image.LINEAR

paypalme           = "https://www.paypal.com/paypalme/jjstd/5"
githubme           = "https://github.com/Djdefrag/ReSR.Scaler"
itchme             = "https://jangystudio.itch.io/resrscaler"

default_font          = 'Segoe UI'
background_color      = "#181818"
window_width          = 1300
window_height         = 750
left_bar_width        = 410
left_bar_height       = window_height
drag_drop_width       = window_width - left_bar_width
drag_drop_height      = window_height
show_image_width      = drag_drop_width * 0.8
show_image_height     = drag_drop_width * 0.6
image_text_width      = drag_drop_width * 0.8
support_button_height = 95 
button_1_y            = 210
button_2_y            = 313
button_3_y            = 416
button_4_y            = 519
text_color            = "#DCDCDC"
selected_button_color = "#ffbf00"

supported_file_list     = ['.jpg', '.jpeg', '.JPG', '.JPEG',
                            '.png', '.PNG',
                            '.webp', '.WEBP',
                            '.bmp', '.BMP',
                            '.tif', '.tiff', '.TIF', '.TIFF',
                            '.mp4', '.MP4',
                            '.webm', '.WEBM',
                            '.mkv', '.MKV',
                            '.flv', '.FLV',
                            '.gif', '.GIF',
                            '.m4v', ',M4V',
                            '.avi', '.AVI',
                            '.mov', '.MOV',
                            '.qt', '.3gp', '.mpg', '.mpeg']

supported_video_list    = ['.mp4', '.MP4',
                            '.webm', '.WEBM',
                            '.mkv', '.MKV',
                            '.flv', '.FLV',
                            '.gif', '.GIF',
                            '.m4v', ',M4V',
                            '.avi', '.AVI',
                            '.mov', '.MOV',
                            '.qt',
                            '.3gp', '.mpg', '.mpeg']

not_supported_file_list = ['.txt', '.exe', '.xls', '.xlsx', '.pdf',
                           '.odt', '.html', '.htm', '.doc', '.docx',
                           '.ods', '.ppt', '.pptx', '.aiff', '.aif',
                           '.au', '.bat', '.java', '.class',
                           '.csv', '.cvs', '.dbf', '.dif', '.eps',
                           '.fm3', '.psd', '.psp', '.qxd',
                           '.ra', '.rtf', '.sit', '.tar', '.zip',
                           '.7zip', '.wav', '.mp3', '.rar', '.aac',
                           '.adt', '.adts', '.bin', '.dll', '.dot',
                           '.eml', '.iso', '.jar', '.py',
                           '.m4a', '.msi', '.ini', '.pps', '.potx',
                           '.ppam', '.ppsx', '.pptm', '.pst', '.pub',
                           '.sys', '.tmp', '.xlt', '.avif']

ctypes.windll.shcore.SetProcessDpiAwareness(True)
scaleFactor = ctypes.windll.shcore.GetScaleFactorForDevice(0) / 100
font_scale = round(1/scaleFactor, 1)


# ---------------------- /Dimensions ----------------------

# ---------------------- Functions ----------------------

# ------------------------ Utils ------------------------

def openpaypal():
    webbrowser.open(paypalme, new=1)

def opengithub():
    webbrowser.open(githubme, new=1)

def openitch():
    webbrowser.open(itchme, new=1)

def create_temp_dir(name_dir):
    if os.path.exists(name_dir):
        shutil.rmtree(name_dir)

    if not os.path.exists(name_dir):
        os.makedirs(name_dir)

def find_by_relative_path(relative_path):
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(
        os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

def adapt_image_to_show(image_to_prepare):
    old_image     = cv2.imread(image_to_prepare)
    actual_width  = old_image.shape[1]
    actual_height = old_image.shape[0]

    if actual_width >= actual_height:
        max_val = actual_width
        max_photo_resolution = show_image_width
    else:
        max_val = actual_height
        max_photo_resolution = show_image_height

    if max_val >= max_photo_resolution:
        downscale_factor = max_val/max_photo_resolution
        new_width        = round(old_image.shape[1]/downscale_factor)
        new_height       = round(old_image.shape[0]/downscale_factor)
        resized_image    = cv2.resize(old_image,
                                   (new_width, new_height),
                                   interpolation = cv2.INTER_NEAREST )
        cv2.imwrite("temp.png", resized_image)
        return "temp.png"
    else:
        new_width        = round(old_image.shape[1])
        new_height       = round(old_image.shape[0])
        resized_image    = cv2.resize(old_image,
                                   (new_width, new_height),
                                   interpolation = cv2.INTER_NEAREST)
        cv2.imwrite("temp.png", resized_image)
        return "temp.png"

def prepare_output_filename(img, AI_model, target_file_extension):
    result_path = (img.replace("_resized" + target_file_extension, "").replace(target_file_extension, "") 
                    + "_"  + AI_model  
                    + target_file_extension)
    return result_path

def delete_list_of_files(list_to_delete):
    if len(list_to_delete) > 0:
        for to_delete in list_to_delete:
            if os.path.exists(to_delete):
                os.remove(to_delete)

def convert_image_list(image_list, target_file_extension):
    converted_images = []
    for image in image_list:
        image = image.strip()
        converted_img = convert_image_and_save(image, target_file_extension)
        converted_images.append(converted_img)

    return converted_images

def convert_image_and_save(image_to_prepare, target_file_extension):
    image_to_prepare = image_to_prepare.replace("{", "").replace("}", "")
    new_image_path = image_to_prepare

    for file_type in supported_file_list:
        new_image_path = new_image_path.replace(file_type, target_file_extension)

    cv2.imwrite(new_image_path, cv2.imread(image_to_prepare))
    return new_image_path

def write_in_log_file(text_to_insert):
    log_file_name   = app_name + ".log"
    with open(log_file_name,'w') as log_file:
        log_file.write(text_to_insert) 
    log_file.close()

def read_log_file():
    log_file_name   = app_name + ".log"
    with open(log_file_name,'r') as log_file:
        step = log_file.readline()
    log_file.close()
    return step


# IMAGE

def resize_image(image_path, resize_factor):
    new_image_path = image_path.replace(target_file_extension, 
                                        "_resized" + target_file_extension)

    old_image = Image.open(image_path)

    new_width, new_height = old_image.size
    new_width = int(new_width * resize_factor)
    new_height = int(new_height * resize_factor)

    max_height_or_width = int(max(new_height, new_width))

    resized_image = old_image.resize((new_width, new_height), 
                                        resample = resize_algorithm)

    if not pay:
        font = ImageFont.truetype(find_by_relative_path("Assets" 
                                                        + os.sep 
                                                        + "arial.ttf"), 
                                    int(max_height_or_width/40))
        img_text = ImageDraw.Draw(resized_image)
        img_text.text((20, 20), 
                        "Upscaled with " + app_name 
                        + "\To avoid watermarks leave a tip here:"
                        + "\n" + itchme , 
                        font = font, 
                        fill = (250, 250, 250))
                                    
    resized_image.save(new_image_path)

def resize_image_list(image_list, resize_factor):
    files_to_delete   = []
    downscaled_images = []
    how_much_images = len(image_list)

    index = 1
    for image in image_list:
        resized_image_path = image.replace(target_file_extension, 
                                            "_resized" + target_file_extension)
        
        resize_image(image, resize_factor)
        write_in_log_file("Resizing image " + str(index) + "/" + str(how_much_images)) 

        downscaled_images.append(resized_image_path)
        files_to_delete.append(resized_image_path)

        index += 1

    return downscaled_images, files_to_delete


#VIDEO

def extract_frames_from_video(video_path):
    cap          = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    extr_frame   = 0
    video_frames_list = []
    
    # extract video frames
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        extr_frame += 1
        result_path = app_name + "_temp" + os.sep + "frame_" + str(extr_frame) + target_file_extension
        
        cv2.imwrite(result_path, frame)
        video_frames_list.append(result_path)

        write_in_log_file("Extracted frames " + str(extr_frame) + "/" + str(total_frames))
    cap.release()

    # extract audio from video
    try:
        video = VideoFileClip.VideoFileClip(video_path)
        audio = video.audio
        audio.write_audiofile(app_name + "_temp" + os.sep + "audio.mp3")
    except:
        pass

    return video_frames_list

def video_reconstruction_by_frames(input_video_path, video_frames_upscaled_list, AI_model):
    cap          = cv2.VideoCapture(input_video_path)
    frame_rate   = int(cap.get(cv2.CAP_PROP_FPS))
    path_as_list = input_video_path.split("/")
    video_name   = str(path_as_list[-1])
    only_path    = input_video_path.replace(video_name, "")
    cap.release()

    for video_type in supported_video_list:
        video_name = video_name.replace(video_type, "")

    upscaled_video_path = (only_path + video_name + "_" + AI_model + ".mp4")

    temp_video_path = app_name + "_temp" + os.sep + "tempVideo.mp4"
    clip = ImageSequenceClip.ImageSequenceClip(video_frames_upscaled_list, 
                                                                fps=frame_rate)
    clip.write_videofile(temp_video_path)

    video = VideoFileClip.VideoFileClip(temp_video_path)
    
    # audio
    try:
        audio = AudioFileClip.AudioFileClip(app_name + "_temp" + os.sep + "audio.mp3")
        new_audioclip = CompositeAudioClip([audio])
        video.audio = new_audioclip
    except:
        pass
    
    video.write_videofile(upscaled_video_path)

def resize_frame(image_path, new_width, new_height, max_height_or_width):
    new_image_path = image_path.replace(target_file_extension, 
                                        "_resized" + target_file_extension)

    old_image = Image.open(image_path)

    resized_image = old_image.resize((new_width, new_height), 
                                        resample = resize_algorithm)

    if not pay:
        font = ImageFont.truetype(find_by_relative_path("Assets" 
                                                        + os.sep 
                                                        + "arial.ttf"), 
                                    int(max_height_or_width/40))
        img_text = ImageDraw.Draw(resized_image)
        img_text.text((20, 20), 
                        "Upscaled with " + app_name 
                        + "\To avoid watermarks leave a tip here:"
                        + "\n" + itchme , 
                        font = font, 
                        fill = (250, 250, 250))
                                    
    resized_image.save(new_image_path)

def resize_frame_list(image_list, resize_factor):
    downscaled_images = []
    how_much_images = len(image_list)    

    old_image = Image.open(image_list[1])
    new_width, new_height = old_image.size
    new_width = int(new_width * resize_factor)
    new_height = int(new_height * resize_factor)

    max_height_or_width = int(max(new_height, new_width))
    
    index = 1

    for image in image_list:
        resized_image_path = image.replace(target_file_extension, 
                                            "_resized" + target_file_extension)
        
        resize_frame(image, new_width, new_height, max_height_or_width)
        write_in_log_file("Resizing frame " + str(index) + "/" + str(how_much_images)) 

        downscaled_images.append(resized_image_path)

        index += 1

    return downscaled_images


# ----------------------- /Utils ------------------------


# ------------------ Neural Net related ------------------


## ---------------------- AI related ----------------------

@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.BatchNorm2d): # booo
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)

def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)

class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale

class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)

def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True):
    """Warp an image or feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.

    Returns:
        Tensor: Warped image or feature map.
    """
    assert x.size()[-2:] == flow.size()[1:3]
    _, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h).type_as(x), torch.arange(0, w).type_as(x))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False

    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode, align_corners=align_corners)

    # TODO, what if align_corners=False
    return output

def resize_flow(flow, size_type, sizes, interp_mode='bilinear', align_corners=False):
    """Resize a flow according to ratio or shape.

    Args:
        flow (Tensor): Precomputed flow. shape [N, 2, H, W].
        size_type (str): 'ratio' or 'shape'.
        sizes (list[int | float]): the ratio for resizing or the final output
            shape.
            1) The order of ratio should be [ratio_h, ratio_w]. For
            downsampling, the ratio should be smaller than 1.0 (i.e., ratio
            < 1.0). For upsampling, the ratio should be larger than 1.0 (i.e.,
            ratio > 1.0).
            2) The order of output_size should be [out_h, out_w].
        interp_mode (str): The mode of interpolation for resizing.
            Default: 'bilinear'.
        align_corners (bool): Whether align corners. Default: False.

    Returns:
        Tensor: Resized flow.
    """
    _, _, flow_h, flow_w = flow.size()
    if size_type == 'ratio':
        output_h, output_w = int(flow_h * sizes[0]), int(flow_w * sizes[1])
    elif size_type == 'shape':
        output_h, output_w = sizes[0], sizes[1]
    else:
        raise ValueError(f'Size type should be ratio or shape, but got type {size_type}.')

    input_flow = flow.clone()
    ratio_h = output_h / flow_h
    ratio_w = output_w / flow_w
    input_flow[:, 0, :, :] *= ratio_w
    input_flow[:, 1, :, :] *= ratio_h
    resized_flow = F.interpolate(
        input=input_flow, size=(output_h, output_w), mode=interp_mode, align_corners=align_corners)
    return resized_flow

def pixel_unshuffle(x, scale):

    b, c, hh, hw = x.size()
    out_channel = c * (scale**2)
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)

    def forward(self, x, feat):
        out = self.conv_offset(feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        offset_absmean = torch.mean(torch.abs(offset))
        if offset_absmean > 50:
            logger = get_root_logger()
            logger.warning(f'Offset abs mean is {offset_absmean}, larger than 50.')

        if LooseVersion(torchvision.__version__) >= LooseVersion('0.9.0'):
            return torchvision.ops.deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding,
                                                    self.dilation, mask)
        else:
            return modulated_deform_conv(x, offset, mask, self.weight, self.bias, self.stride, self.padding,
                                            self.dilation, self.groups, self.deformable_groups)

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/weight_init.py
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            'mean is more than 2 std from [a, b] in nn.init.trunc_normal_. '
            'The distribution of values may be incorrect.',
            stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        low = norm_cdf((a - mean) / std)
        up = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [low, up], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * low - 1, 2 * up - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution.

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/weight_init.py

    The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

def _ntuple(n):

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse

to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple

class ResidualDenseBlock(nn.Module):
    """Residual Dense Block.

    Used in RRDB block in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Empirically, we use 0.2 to scale the residual for better performance
        return x5 * 0.2 + x

class RRDB(nn.Module):
    """Residual in Residual Dense Block.

    Used in RRDB-Net in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Empirically, we use 0.2 to scale the residual for better performance
        return out * 0.2 + x

class RRDBNet(nn.Module):
    """Networks consisting of Residual in Residual Dense Block, which is used
    in ESRGAN.

    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.

    We extend ESRGAN for scale x2 and scale x1.
    Note: This is one option for scale 1, scale 2 in RRDBNet.
    We first employ the pixel-unshuffle (an inverse operation of pixelshuffle to reduce the spatial size
    and enlarge the channel size before feeding inputs into the main ESRGAN architecture.

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64
        num_block (int): Block number in the trunk network. Defaults: 23
        num_grow_ch (int): Channels for each growth. Default: 32.
    """

    def __init__(self, num_in_ch, num_out_ch, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super(RRDBNet, self).__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        # upsample
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out

class RealESRGANer():
    """A helper class for upsampling images with RealESRGAN.

    Args:
        scale (int): Upsampling scale factor used in the networks. It is usually 2 or 4.
        model_path (str): The path to the pretrained model. It can be urls (will first download it automatically).
        model (nn.Module): The defined network. Default: None.
        tile (int): As too large images result in the out of GPU memory issue, so this tile option will first crop
            input images into tiles, and then process each of them. Finally, they will be merged into one image.
            0 denotes for do not use tile. Default: 0.
        tile_pad (int): The pad size for each tile, to remove border artifacts. Default: 10.
        pre_pad (int): Pad the input images to avoid border artifacts. Default: 10.
        half (float): Whether to use half precision during inference. Default: False.
    """

    def __init__(self,
                 scale,
                 model_path,
                 dni_weight=None,
                 model=None,
                 tile=0,
                 tile_pad=10,
                 pre_pad=10,
                 half=False,
                 device=None,
                 gpu_id=None):
        self.scale = scale
        self.tile_size = tile
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.mod_scale = None
        self.half = half

        # initialize model
        self.device = torch.device(device)

        if isinstance(model_path, list):
            # dni
            assert len(model_path) == len(dni_weight), 'model_path and dni_weight should have the save length.'
            loadnet = self.dni(model_path[0], model_path[1], dni_weight)
        else:
            loadnet = torch.load(model_path, map_location=torch.device('cpu'))

        # prefer to use params_ema
        if 'params_ema' in loadnet:
            keyname = 'params_ema'
        else:
            keyname = 'params'
        model.load_state_dict(loadnet[keyname], strict=True)

        model.eval()
        self.model = model.to(self.device)
        if self.half:
            self.model = self.model.half()

    def dni(self, net_a, net_b, dni_weight, key='params', loc='cpu'):
        """Deep network interpolation.

        ``Paper: Deep Network Interpolation for Continuous Imagery Effect Transition``
        """
        net_a = torch.load(net_a, map_location=torch.device(loc))
        net_b = torch.load(net_b, map_location=torch.device(loc))
        for k, v_a in net_a[key].items():
            net_a[key][k] = dni_weight[0] * v_a + dni_weight[1] * net_b[key][k]
        return net_a

    def pre_process(self, img):
        """Pre-process, such as pre-pad and mod pad, so that the images can be divisible
        """
        img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        self.img = img.unsqueeze(0).to(self.device)
        if self.half:
            self.img = self.img.half()

        # pre_pad
        if self.pre_pad != 0:
            self.img = F.pad(self.img, (0, self.pre_pad, 0, self.pre_pad), 'reflect')
        # mod pad for divisible borders
        if self.scale == 2:
            self.mod_scale = 2
        elif self.scale == 1:
            self.mod_scale = 4
        if self.mod_scale is not None:
            self.mod_pad_h, self.mod_pad_w = 0, 0
            _, _, h, w = self.img.size()
            if (h % self.mod_scale != 0):
                self.mod_pad_h = (self.mod_scale - h % self.mod_scale)
            if (w % self.mod_scale != 0):
                self.mod_pad_w = (self.mod_scale - w % self.mod_scale)
            self.img = F.pad(self.img, (0, self.mod_pad_w, 0, self.mod_pad_h), 'reflect')

    def process(self):
        # model inference
        self.output = self.model(self.img)

    def tile_process(self):
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.

        Modified from: https://github.com/ata4/esrgan-launcher
        """
        batch, channel, height, width = self.img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        self.output = self.img.new_zeros(output_shape)
        tiles_x = math.ceil(width / self.tile_size)
        tiles_y = math.ceil(height / self.tile_size)

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * self.tile_size
                ofs_y = y * self.tile_size
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + self.tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + self.tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - self.tile_pad, 0)
                input_end_x_pad = min(input_end_x + self.tile_pad, width)
                input_start_y_pad = max(input_start_y - self.tile_pad, 0)
                input_end_y_pad = min(input_end_y + self.tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = self.img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # upscale tile
                try:
                    with torch.no_grad():
                        output_tile = self.model(input_tile)
                except RuntimeError as error:
                    print('Error', error)
                print(f'\tTile {tile_idx}/{tiles_x * tiles_y}')

                # output tile area on total image
                output_start_x = input_start_x * self.scale
                output_end_x = input_end_x * self.scale
                output_start_y = input_start_y * self.scale
                output_end_y = input_end_y * self.scale

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale
                output_end_x_tile = output_start_x_tile + input_tile_width * self.scale
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale
                output_end_y_tile = output_start_y_tile + input_tile_height * self.scale

                # put tile into output image
                self.output[:, :, output_start_y:output_end_y,
                            output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                                                       output_start_x_tile:output_end_x_tile]

    def post_process(self):
        # remove extra pad
        if self.mod_scale is not None:
            _, _, h, w = self.output.size()
            self.output = self.output[:, :, 0:h - self.mod_pad_h * self.scale, 0:w - self.mod_pad_w * self.scale]
        # remove prepad
        if self.pre_pad != 0:
            _, _, h, w = self.output.size()
            self.output = self.output[:, :, 0:h - self.pre_pad * self.scale, 0:w - self.pre_pad * self.scale]
        return self.output

    @torch.no_grad()
    def enhance(self, img, outscale=None, alpha_upsampler='realesrgan'):
        h_input, w_input = img.shape[0:2]
        # img: numpy
        img = img.astype(np.float32)
        if np.max(img) > 256:  # 16-bit image
            max_range = 65535
            print('\tInput is a 16-bit image')
        else:
            max_range = 255
        img = img / max_range
        if len(img.shape) == 2:  # gray image
            img_mode = 'L'
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:  # RGBA image with alpha channel
            img_mode = 'RGBA'
            alpha = img[:, :, 3]
            img = img[:, :, 0:3]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if alpha_upsampler == 'realesrgan':
                alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2RGB)
        else:
            img_mode = 'RGB'
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ------------------- process image (without the alpha channel) ------------------- #
        self.pre_process(img)
        if self.tile_size > 0:
            self.tile_process()
        else:
            self.process()
        output_img = self.post_process()
        output_img = output_img.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output_img = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0))
        if img_mode == 'L':
            output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)

        # ------------------- process the alpha channel if necessary ------------------- #
        if img_mode == 'RGBA':
            if alpha_upsampler == 'realesrgan':
                self.pre_process(alpha)
                if self.tile_size > 0:
                    self.tile_process()
                else:
                    self.process()
                output_alpha = self.post_process()
                output_alpha = output_alpha.data.squeeze().float().cpu().clamp_(0, 1).numpy()
                output_alpha = np.transpose(output_alpha[[2, 1, 0], :, :], (1, 2, 0))
                output_alpha = cv2.cvtColor(output_alpha, cv2.COLOR_BGR2GRAY)
            else:  # use the cv2 resize for alpha channel
                h, w = alpha.shape[0:2]
                output_alpha = cv2.resize(alpha, (w * self.scale, h * self.scale), interpolation=cv2.INTER_LINEAR)

            # merge the alpha channel
            output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2BGRA)
            output_img[:, :, 3] = output_alpha

        # ------------------------------ return ------------------------------ #
        if max_range == 65535:  # 16-bit image
            output = (output_img * 65535.0).round().astype(np.uint16)
        else:
            output = (output_img * 255.0).round().astype(np.uint8)

        if outscale is not None and outscale != float(self.scale):
            output = cv2.resize(
                output, (
                    int(w_input * outscale),
                    int(h_input * outscale),
                ), interpolation=cv2.INTER_LANCZOS4)

        return output, img_mode

class PrefetchReader(threading.Thread):
    """Prefetch images.

    Args:
        img_list (list[str]): A image list of image paths to be read.
        num_prefetch_queue (int): Number of prefetch queue.
    """

    def __init__(self, img_list, num_prefetch_queue):
        super().__init__()
        self.que = queue.Queue(num_prefetch_queue)
        self.img_list = img_list

    def run(self):
        for img_path in self.img_list:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            self.que.put(img)

        self.que.put(None)

    def __next__(self):
        next_item = self.que.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __iter__(self):
        return self

class IOConsumer(threading.Thread):

    def __init__(self, opt, que, qid):
        super().__init__()
        self._queue = que
        self.qid = qid
        self.opt = opt

    def run(self):
        while True:
            msg = self._queue.get()
            if isinstance(msg, str) and msg == 'quit':
                break

            output = msg['output']
            save_path = msg['save_path']
            cv2.imwrite(save_path, output)
        print(f'IO worker {self.qid} is done.')

class SRVGGNetCompact(nn.Module):

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu'):
        super(SRVGGNetCompact, self).__init__()
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_conv = num_conv
        self.upscale = upscale
        self.act_type = act_type

        self.body = nn.ModuleList()
        # the first conv
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        # the first activation
        if act_type == 'relu':
            activation = nn.ReLU(inplace=True)
        elif act_type == 'prelu':
            activation = nn.PReLU(num_parameters=num_feat)
        elif act_type == 'leakyrelu':
            activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.body.append(activation)

        # the body structure
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            # activation
            if act_type == 'relu':
                activation = nn.ReLU(inplace=True)
            elif act_type == 'prelu':
                activation = nn.PReLU(num_parameters=num_feat)
            elif act_type == 'leakyrelu':
                activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
            self.body.append(activation)

        # the last conv
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        # upsample
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for i in range(0, len(self.body)):
            out = self.body[i](out)

        out = self.upsampler(out)
        # add the nearest upsampled image, so that the network learns the residual
        base = F.interpolate(x, scale_factor=self.upscale, mode='nearest')
        out += base
        return out


# ------------------ /Neural Net related ------------------


# ----------------------- Core ------------------------


def file_drop_event(event):
    global image_path
    global multiple_files
    global multi_img_list
    global video_files
    global single_file
    global input_video_path

    supported_file_dropped_number, not_supported_file_dropped_number, supported_video_dropped_number = count_files_dropped(event)
    all_supported, single_file, multiple_files, video_files, more_than_one_video = check_compatibility(supported_file_dropped_number, not_supported_file_dropped_number, supported_video_dropped_number)

    if video_files:
        # video section
        if not all_supported:
            info_string.set("Some files are not supported")
            return
        elif all_supported:
            if multiple_files:
                info_string.set("Only one video supported")
                return
            elif not multiple_files:
                if not more_than_one_video:
                    input_video_path = str(event.data).replace("{", "").replace("}", "")
                    
                    show_video_info_with_drag_drop(input_video_path)

                    # reset variable
                    image_path = "no file"
                    multi_img_list = []

                elif more_than_one_video:
                    info_string.set("Only one video supported")
                    return
    else:
        # image section
        if not all_supported:
            if multiple_files:
                info_string.set("Some files are not supported")
                return
            elif single_file:
                info_string.set("This file is not supported")
                return
        elif all_supported:
            if multiple_files:
                image_list_dropped = drop_event_to_image_list(event)

                show_list_images_in_GUI(image_list_dropped)
                
                multi_img_list = image_list_dropped

                place_clean_button()

                # reset variable
                image_path = "no file"
                video_frames_list = []

            elif single_file:
                image_list_dropped = drop_event_to_image_list(event)

                # convert images to target file extension
                show_single_image_inGUI = threading.Thread(target = show_image_in_GUI,
                                                         args=(str(image_list_dropped[0]), 1),
                                                         daemon=True)
                show_single_image_inGUI.start()

                multi_img_list = image_list_dropped

                # reset variable
                image_path = "no file"
                video_frames_list = []

def check_compatibility(supported_file_dropped_number, not_supported_file_dropped_number, supported_video_dropped_number):
    all_supported  = True
    single_file    = False
    multiple_files = False
    video_files    = False
    more_than_one_video = False

    if not_supported_file_dropped_number > 0:
        all_supported = False

    if supported_file_dropped_number + not_supported_file_dropped_number == 1:
        single_file = True
    elif supported_file_dropped_number + not_supported_file_dropped_number > 1:
        multiple_files = True

    if supported_video_dropped_number == 1:
        video_files = True
        more_than_one_video = False
    elif supported_video_dropped_number > 1:
        video_files = True
        more_than_one_video = True

    return all_supported, single_file, multiple_files, video_files, more_than_one_video

def count_files_dropped(event):
    supported_file_dropped_number = 0
    not_supported_file_dropped_number = 0
    supported_video_dropped_number = 0

    # count compatible images files
    for file_type in supported_file_list:
        supported_file_dropped_number = supported_file_dropped_number + \
            str(event.data).count(file_type)

    # count compatible video files
    for file_type in supported_video_list:
        supported_video_dropped_number = supported_video_dropped_number + \
            str(event.data).count(file_type)

    # count not supported files
    for file_type in not_supported_file_list:
        not_supported_file_dropped_number = not_supported_file_dropped_number + \
            str(event.data).count(file_type)

    return supported_file_dropped_number, not_supported_file_dropped_number, supported_video_dropped_number

def thread_check_steps_for_images( not_used_var, not_used_var2 ):
    time.sleep(3)
    try:
        while True:
            step = read_log_file()
            if "Upscale completed" in step or "Error while upscaling" in step or "Stopped upscaling" in step:
                
                if  "Error while upscaling" in step:
                    info_string.set("Error while upscaling | check .log file")
                else:
                    info_string.set(step)

                stop = 1 + "x"
            info_string.set(step)
            time.sleep(1)
    except:
        place_upscale_button()

def thread_check_steps_for_videos( not_used_var, not_used_var2 ):
    time.sleep(3)
    try:
        while True:
            step = read_log_file()
            if "Upscale video completed" in step or "Error while upscaling" in step or "Stopped upscaling" in step:
                
                if  "Error while upscaling" in step:
                    info_string.set("Error while upscaling | check .log file")
                else:
                    info_string.set(step)

                stop = 1 + "x"
            info_string.set(step)
            time.sleep(1)
    except:
        place_upscale_button()

def drop_event_to_image_list(event):
    image_list = str(event.data).replace("{", "").replace("}", "")

    for file_type in supported_file_list:
        image_list = image_list.replace(file_type, file_type+"\n")

    image_list = image_list.split("\n")
    image_list.pop() 

    return image_list

def optimize_torch():
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)



def prepare_AI_model(AI_model, device):
    backend = torch.device(device)

    model_path = find_by_relative_path("AI" + os.sep + AI_model + ".pth")

    dni_weight = None

    if 'RealESRGAN_x4plus' in AI_model: 
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
    elif 'realesr-general-x4v3' in AI_model: 
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        netscale = 4
        if denoise_strength != 1:
            wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
            model_path = [model_path, wdn_model_path]
            dni_weight = [denoise_strength, 1 - denoise_strength]

    model = RealESRGANer( scale = netscale, model_path = model_path, dni_weight = dni_weight,
                            model = model, tile = 0, tile_pad = 10, pre_pad = 0,
                            half  = half_precision, gpu_id = None, device = device)
        
    return model

def upscale_image_and_save(img, model, result_path, device, tiles_resolution):
    img_tmp          = cv2.imread(img)
    image_resolution = max(img_tmp.shape[1], img_tmp.shape[0])
    num_tiles        = image_resolution/tiles_resolution

    if num_tiles <= 1:
        with torch.no_grad():
            img_adapted  = cv2.imread(img, cv2.IMREAD_UNCHANGED)
            img_upscaled, _ = model.enhance(img_adapted, outscale=4)
            cv2.imwrite(result_path, img_upscaled)
    else:
        num_tiles = round(num_tiles)
        if (num_tiles % 2) != 0: num_tiles += 1
        num_tiles = round(num_tiles * multiplier_num_tiles)

        num_tiles_applied = int(num_tiles/2)
        how_many_tiles = int(pow(num_tiles/2, 2))

        split_image(img, num_tiles_applied, num_tiles_applied, False, should_cleanup = True, should_quiet=True, output_dir=None)

        tiles = []
        for index in range(how_many_tiles):
            tiles.append(img.replace(".png", "_" + str(index) + ".png"))

        with torch.no_grad():
            for tile in tiles:
                tile_adapted  = cv2.imread(tile, cv2.IMREAD_UNCHANGED)
                tile_upscaled, _ = model.enhance(tile_adapted, outscale=4)
                cv2.imwrite(tile, tile_upscaled)

        reverse_split(tiles, num_tiles_applied, num_tiles_applied, result_path, should_cleanup=True, should_quiet=True)

def process_upscale_multiple_images_qualityscaler(image_list, AI_model, resize_factor, device, tiles_resolution, target_file_extension):
    try:
        start = timer()
        
        write_in_log_file('...')

        optimize_torch()

        write_in_log_file('Resizing images')
        image_list = convert_image_list(image_list, target_file_extension)
        image_list, files_to_delete = resize_image_list(image_list, resize_factor)

        how_many_images = len(image_list)
        done_images     = 0

        write_in_log_file('Upscaling...')
        for img in image_list:
            model = prepare_AI_model(AI_model, device)
            result_path = prepare_output_filename(img, AI_model, target_file_extension)
            upscale_image_and_save(img, model, 
                                    result_path, device,    
                                    tiles_resolution)

            done_images += 1
            write_in_log_file("Upscaled images " + str(done_images) + "/" + str(how_many_images))
                
        write_in_log_file("Upscale completed [" + str(round(timer() - start)) + " sec.]")

        delete_list_of_files(files_to_delete)
    except Exception as e:
        write_in_log_file('Error while upscaling' + '\n\n' + str(e)) 

def process_upscale_video_frames_qualityscaler(input_video_path, AI_model, resize_factor, device, tiles_resolution, target_file_extension):
    try:
        start = timer()

        create_temp_dir(app_name + "_temp")

        write_in_log_file('...')
      
        optimize_torch()

        write_in_log_file('Extracting video frames')
        image_list = extract_frames_from_video(input_video_path)
        
        write_in_log_file('Resizing video frames')
        image_list  = resize_frame_list(image_list, resize_factor)

        write_in_log_file('Upscaling...')
        how_many_images = len(image_list)
        done_images     = 0
        video_frames_upscaled_list = []

        model = prepare_AI_model(AI_model, device)

        for img in image_list:
            result_path = prepare_output_filename(img, AI_model, target_file_extension)
            video_frames_upscaled_list.append(result_path)
            upscale_image_and_save(img, model, 
                                    result_path, device,    
                                    tiles_resolution)
            done_images += 1
            write_in_log_file("Upscaled frame " + str(done_images) + "/" + str(how_many_images))

        write_in_log_file("Processing upscaled video")
        
        video_reconstruction_by_frames(input_video_path, video_frames_upscaled_list, AI_model)

        write_in_log_file("Upscale video completed [" + str(round(timer() - start)) + " sec.]")

        create_temp_dir(app_name + "_temp")
    except Exception as e:
        write_in_log_file('Error while upscaling' + '\n\n' + str(e)) 
    

# ----------------------- /Core ------------------------

# ---------------------- GUI related ----------------------


def upscale_button_command():
    global image_path
    global multiple_files
    global process_upscale
    global thread_wait
    global resize_factor
    global video_frames_list
    global video_files
    global video_frames_upscaled_list
    global input_video_path
    global device

    try:
        resize_factor = int(float(str(selected_resize_factor.get())))
    except:
        info_string.set("Resize input must be a numeric value")
        return

    if resize_factor > 0 and resize_factor <= 100:
        resize_factor = resize_factor/100
        pass    
    else:
        info_string.set("Resize must be in range 1 - 100")
        return

    info_string.set("...")

    if video_files:
        place_stop_button()

        process_upscale = multiprocessing.Process(target = process_upscale_video_frames_qualityscaler,
                                                  args   = (input_video_path, 
                                                            AI_model, 
                                                            resize_factor, 
                                                            device,
                                                            tiles_resolution,
                                                            target_file_extension))
        process_upscale.start()

        thread_wait = threading.Thread(target = thread_check_steps_for_videos,
                                       args   = (1, 2), 
                                       daemon = True)
        thread_wait.start()

    elif multiple_files:
        place_stop_button()
        
        process_upscale = multiprocessing.Process(target = process_upscale_multiple_images_qualityscaler,
                                                    args   = (multi_img_list, 
                                                             AI_model, 
                                                             resize_factor, 
                                                             device,
                                                             tiles_resolution,
                                                             target_file_extension))
        process_upscale.start()

        thread_wait = threading.Thread(target = thread_check_steps_for_images,
                                        args   = (1, 2), daemon = True)
        thread_wait.start()

    elif single_file:
        place_stop_button()

        process_upscale = multiprocessing.Process(target = process_upscale_multiple_images_qualityscaler,
                                                    args   = (multi_img_list, 
                                                             AI_model, 
                                                             resize_factor, 
                                                             device,
                                                             tiles_resolution,
                                                             target_file_extension))
        process_upscale.start()

        thread_wait = threading.Thread(target = thread_check_steps_for_images,
                                       args   = (1, 2), daemon = True)
        thread_wait.start()

    elif "no file" in image_path:
        info_string.set("No file selected")
  
def stop_button_command():
    global process_upscale
    process_upscale.terminate()
    
    # this will stop thread that check upscaling steps
    write_in_log_file("Stopped upscaling") 

def clear_input_variables():
    global image_path
    global multi_img_list
    global video_frames_list
    global single_file
    global multiple_files
    global video_files

    # reset variable
    image_path        = "no file"
    multi_img_list    = []
    video_frames_list = []
    single_file       = False
    multiple_files    = False
    video_files       = False
    multi_img_list    = []
    video_frames_list = []

def clear_app_background():
    drag_drop = ttk.Label(root,
                          ondrop = file_drop_event,
                          relief = "flat",
                          background = background_color,
                          foreground = text_color)
    drag_drop.place(x = left_bar_width + 50, y=0,
                    width = drag_drop_width, height = drag_drop_height)

def show_video_info_with_drag_drop(video_path):
    global image
    
    fist_frame = "temp.jpg"
    
    clear_app_background()
    place_clean_button()

    cap          = cv2.VideoCapture(video_path)
    width        = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate   = cap.get(cv2.CAP_PROP_FPS)
    duration     = num_frames/frame_rate
    minutes      = int(duration/60)
    seconds      = duration % 60
    path_as_list = video_path.split("/")
    video_name   = str(path_as_list[-1])
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        cv2.imwrite(fist_frame, frame)
        break
    cap.release()

    image_to_show_resized = adapt_image_to_show(fist_frame)

    image = tk.PhotoImage(file = image_to_show_resized)
    drag_drop_and_images = ttk.Label(root,
                                     image   = image,
                                     ondrop  = file_drop_event,
                                     anchor  = "center",
                                     relief  = "flat",
                                     justify = "center",
                                     background = background_color,
                                     foreground = "#202020")
    drag_drop_and_images.place(x = 30 + left_bar_width + drag_drop_width/2 - show_image_width/2,
                               y = drag_drop_height/2 - show_image_height/2 - 15,
                               width  = show_image_width,
                               height = show_image_height)

    os.remove(fist_frame)

    file_description = ( video_name + "\n" + "[" + str(width) + "x" + str(height) + "]" + " | " + str(minutes) + 'm:' + str(round(seconds)) + "s | " + str(num_frames) + "frames | " + str(round(frame_rate)) + "fps")

    video_info_width = drag_drop_width * 0.8

    video_info_space = ttk.Label(root,
                                 text    = file_description,
                                 ondrop  = file_drop_event,
                                 font    = (default_font, round(11 * font_scale), "bold"),
                                 anchor  = "center",
                                 relief  = "flat",
                                 justify = "center",
                                 background = background_color,
                                 foreground = "#D3D3D3",
                                 wraplength = video_info_width * 0.95)
    video_info_space.place(x = 30 + left_bar_width + drag_drop_width/2 - video_info_width/2,
                           y = drag_drop_height - 85,
                           width  = video_info_width,
                           height = 65)

def show_list_images_in_GUI(image_list_prepared):
    clear_app_background()
    place_clean_button()

    final_string = "\n"
    counter_img = 0

    for elem in image_list_prepared:
        counter_img += 1
        if counter_img <= 8:
            # add first 8 files in list
            img     = cv2.imread(elem.strip())
            width   = round(img.shape[1])
            height  = round(img.shape[0])
            img_name = str(elem.split("/")[-1])

            final_string += (str(counter_img) 
                            + ".  " 
                            + img_name 
                            + " | [" + str(width) + "x" + str(height) + "]" + "\n\n")
        else:
            final_string += "and others... \n"
            break

    list_height = 420
    list_width  = 750

    images_list = ttk.Label(root,
                            text    = final_string,
                            ondrop  = file_drop_event,
                            font    = (default_font, round(12 * font_scale)),
                            anchor  = "n",
                            relief  = "flat",
                            justify = "left",
                            background = background_color,
                            foreground = "#D3D3D3",
                            wraplength = list_width)

    images_list.place(x = 30 + left_bar_width + drag_drop_width/2 - list_width/2,
                               y = drag_drop_height/2 - list_height/2 -25,
                               width  = list_width,
                               height = list_height)

    images_counter = ttk.Entry(root, 
                                foreground = text_color,
                                ondrop  = file_drop_event,
                                font    = (default_font, round(12 * font_scale), "bold"), 
                                justify = 'center')

    images_counter.insert(0, str(len(image_list_prepared)) + ' images')

    images_counter.configure(state='disabled')

    images_counter.place(x = left_bar_width + drag_drop_width/2 + 125,
                        y = drag_drop_height/2 + 250,
                        width  = 250,
                        height = 42)

def show_image_in_GUI(image_to_show, _ ):
    global image
    image_to_show = image_to_show.replace('{', '').replace('}', '')

    image_to_show_resized = adapt_image_to_show(image_to_show)

    clear_app_background()
    place_clean_button()

    image = tk.PhotoImage(file = image_to_show_resized)
    drag_drop_and_images = ttk.Label(root,
                                     text="",
                                     image   = image,
                                     ondrop  = file_drop_event,
                                     anchor  = "center",
                                     relief  = "flat",
                                     justify = "center",
                                     background = background_color,
                                     foreground = "#202020")
    drag_drop_and_images.place(x = 30 + left_bar_width + drag_drop_width/2 - show_image_width/2,
                               y = drag_drop_height/2 - show_image_height/2,
                               width  = show_image_width,
                               height = show_image_height)

    path_as_list = image_to_show.split("/")
    img_name     = str(path_as_list[-1])
    img          = cv2.imread(image_to_show)
    width        = round(img.shape[1])
    height       = round(img.shape[0])

    single_image_path = (img_name + " | [" + str(width) + "x" + str(height) + "]")
    single_image_info = ttk.Label(root,
                                  font =(default_font, round(11 * font_scale), "bold"),
                                  text = single_image_path,
                                  relief  = "flat",
                                  justify = "center",
                                  background = background_color,
                                  foreground = "#D3D3D3",
                                  anchor     = "center")

    single_image_info.place(x = 30 + left_bar_width + drag_drop_width/2 - image_text_width/2,
                            y = drag_drop_height - 70,
                            width  = image_text_width,
                            height = 40)

    place_clean_button()

def place_upscale_button():
    global play_icon
    play_icon = tk.PhotoImage(file = find_by_relative_path("Assets" 
                                                        + os.sep 
                                                        + "upscale_icon.png"))
    
    ft = tkFont.Font(family = default_font,
                    size   = round(11 * font_scale),
                    weight = 'bold')
    Upsc_Butt_Style = ttk.Style()
    Upsc_Butt_Style.configure("Bold.TButton", font = ft, foreground = text_color)

    Upscale_button = ttk.Button(root, 
                                text  = '  UPSCALE',
                                image = play_icon,
                                compound = tk.LEFT,
                                style    = 'Bold.TButton')

    Upscale_button.place(x      = 50 + left_bar_width/2 - 310/2,  
                         y      = left_bar_height - 95,
                         width  = 310,
                         height = 45)
    Upscale_button["command"] = lambda: upscale_button_command()

def place_stop_button():
    ft = tkFont.Font(family = default_font,
                    size   = round(11 * font_scale),
                    weight = 'bold')
    
    global stop_icon
    stop_icon = tk.PhotoImage(file = find_by_relative_path("Assets" 
                                                        + os.sep 
                                                        + "stop_icon.png"))
    
    Upsc_Butt_Style = ttk.Style()
    Upsc_Butt_Style.configure("Bold.TButton", font = ft)

    Stop_button = ttk.Button(root, 
                                text  = '  STOP UPSCALE ',
                                image = stop_icon,
                                compound = tk.LEFT,
                                style    = 'Bold.TButton')

    Stop_button.place(x      = 50 + left_bar_width/2 - 310/2,  
                      y      = left_bar_height - 95,
                      width  = 310,
                      height = 45)

    Stop_button["command"] = lambda: stop_button_command()

def combobox_AI_selection(event):
    global AI_model

    selected = str(selected_AI.get())

    AI_model = selected

    Combo_box_AI.set('')
    Combo_box_AI.set(selected)

def combobox_backend_selection(event):
    global device

    selected = str(selected_backend.get())
    if 'gpu' in selected:
        device = "dml"
    elif 'cpu' in selected:
        device = "cpu"

    Combo_box_backend.set('')
    Combo_box_backend.set(selected)

def combobox_VRAM_selection(event):
    global tiles_resolution

    selected = str(selected_VRAM.get())

    if 'Minimal(2GB)' == selected:
        tiles_resolution = 100
    if 'Medium(4GB)' == selected:
        tiles_resolution = 300
    if 'Normal(6GB)' == selected:
        tiles_resolution = 500
    if 'High(8GB)' == selected:
        tiles_resolution = 700
    if 'Ultra(12GB)' == selected:
        tiles_resolution = 900
    if 'Max(>16GB)' == selected:
        tiles_resolution = 1200

    Combo_box_VRAM.set('')
    Combo_box_VRAM.set(selected)




def place_drag_drop_widget():
    clear_input_variables()

    clear_app_background()

    ft = tkFont.Font(family = default_font,
                        size   = round(12 * font_scale),
                        weight = "bold")

    text_drop = (" DROP FILES HERE \n\n"
                + " ⥥ \n\n"
                + " IMAGE   - jpg png tif bmp webp \n\n"
                + " IMAGE LIST   - jpg png tif bmp webp \n\n"
                + " VIDEO   - mp4 webm mkv flv gif avi mov mpg qt 3gp \n\n")

    drag_drop = ttk.Notebook(root, ondrop  = file_drop_event)

    x_center = 30 + left_bar_width + drag_drop_width/2 - (drag_drop_width * 0.75)/2
    y_center = drag_drop_height/2 - (drag_drop_height * 0.75)/2

    drag_drop.place(x = x_center, 
                    y = y_center, 
                    width  = drag_drop_width * 0.75, 
                    height = drag_drop_height * 0.75)

    drag_drop_text = ttk.Label(root,
                            text    = text_drop,
                            ondrop  = file_drop_event,
                            font    = ft,
                            anchor  = "center",
                            relief  = 'flat',
                            justify = "center",
                            foreground = text_color)

    x_center = 30 + left_bar_width + drag_drop_width/2 - (drag_drop_width * 0.5)/2
    y_center = drag_drop_height/2 - (drag_drop_height * 0.5)/2
    
    drag_drop_text.place(x = x_center, 
                         y = y_center, 
                         width  = drag_drop_width * 0.50, 
                         height = drag_drop_height * 0.50)

def place_backend_combobox():
    ft = tkFont.Font(family = default_font,
                     size   = round(11 * font_scale),
                     weight = "bold")

    root.option_add("*TCombobox*Listbox*Background", background_color)
    root.option_add("*TCombobox*Listbox*Foreground", selected_button_color)
    root.option_add("*TCombobox*Listbox*Font",       ft)
    root.option_add('*TCombobox*Listbox.Justify',    'center')

    global Combo_box_backend
    Combo_box_backend = ttk.Combobox(root, 
                            textvariable = selected_backend, 
                            justify      = 'center',
                            foreground   = text_color,
                            values       = ['GPU', 'CPU'],
                            state        = 'readonly',
                            takefocus    = False,
                            font         = ft)
    Combo_box_backend.place(x = 50 + left_bar_width/2 - 285/2, 
                            y = button_3_y, 
                            width  = 290, 
                            height = 42)
    Combo_box_backend.bind('<<ComboboxSelected>>', combobox_backend_selection)
    Combo_box_backend.set('GPU')

def place_resize_factor_entrybox():
    ft = tkFont.Font(family = default_font,
                     size   = round(11 * font_scale),
                     weight = "bold")

    global Entry_box_resize_factor
    Entry_box_resize_factor = ttk.Entry(root, 
                                        textvariable = selected_resize_factor, 
                                        justify      = 'center',
                                        foreground   = text_color,
                                        takefocus    = False,
                                        font         = ft)
    Entry_box_resize_factor.place(x = 50 + left_bar_width/2 - 285/2, 
                                    y = button_2_y, 
                                    width  = 290 * 0.8, 
                                    height = 42)
    Entry_box_resize_factor.insert(0, '70')

    Label_percentage = ttk.Label(root,
                                 text       = "%",
                                 justify    = "center",
                                 font       = tkFont.Font(family = default_font,
                                            size   = round(13 * font_scale),
                                            weight = "bold"),
                                 foreground = text_color)
    Label_percentage.place(x = left_bar_width/2 + 160, 
                            y = button_2_y + 2, 
                            width  = 30, 
                            height = 42)
    
def place_AI_combobox():
    ft = tkFont.Font(family = default_font,
                     size   = round(11 * font_scale),
                     weight = "bold")

    models_array = [ 'RealESRGAN_x4plus', 'realesr-general-x4v3 (coming soon)' ]

    global Combo_box_AI
    Combo_box_AI = ttk.Combobox(root, 
                        textvariable = selected_AI, 
                        justify      = 'center',
                        foreground   = text_color,
                        values       = models_array,
                        state        = 'readonly',
                        takefocus    = False,
                        font         = ft)
    Combo_box_AI.place(x = 50 + left_bar_width/2 - 285/2,  
                       y = button_1_y, 
                       width  = 290, 
                       height = 42)
    Combo_box_AI.bind('<<ComboboxSelected>>', combobox_AI_selection)
    Combo_box_AI.set(AI_model)

def place_VRAM_combobox():
    ft = tkFont.Font(family = default_font,
                     size   = round(11 * font_scale),
                     weight = "bold")

    global Combo_box_VRAM
    Combo_box_VRAM = ttk.Combobox(root, 
                            textvariable = selected_VRAM, 
                            justify      = 'center',
                            foreground   = text_color,
                            values       = ['Minimal(2GB)', 'Medium(4GB)', 
                                            'Normal(6GB)', 'High(8GB)', 
                                            'Ultra(12GB)', 'Max(>16GB)' ],
                            state        = 'readonly',
                            takefocus    = False,
                            font         = ft)
    Combo_box_VRAM.place(x = 50 + left_bar_width/2 - 285/2, 
                         y = button_4_y, 
                         width  = 290, 
                         height = 42)
    Combo_box_VRAM.bind('<<ComboboxSelected>>', combobox_VRAM_selection)

    if tiles_resolution == 100:
        Combo_box_VRAM.set('Minimal(2GB)')
    if tiles_resolution == 300:
        Combo_box_VRAM.set('Medium(4GB)')
    if tiles_resolution == 500:
        Combo_box_VRAM.set('Normal(6GB)')
    if tiles_resolution == 700:
        Combo_box_VRAM.set('High(8GB)')
    if tiles_resolution == 900:
        Combo_box_VRAM.set('Ultra(12GB)')
    if tiles_resolution == 1200:
        Combo_box_VRAM.set('Max(>16GB)')

def place_clean_button():
    global clear_icon
    clear_icon = tk.PhotoImage(file = find_by_relative_path("Assets" 
                                                        + os.sep 
                                                        + "clear_icon.png"))

    clean_button = ttk.Button(root, 
                            text     = '  CLEAN',
                            image    = clear_icon,
                            compound = 'left',
                            style    = 'Bold.TButton')

    clean_button.place(x = 50 + left_bar_width + drag_drop_width/2 - 175/2,
                       y = 25,
                       width  = 175,
                       height = 40)
    clean_button["command"] = lambda: place_drag_drop_widget()

def place_background():
    Background = ttk.Label(root, background = background_color, relief = 'flat')
    Background.place(x = 0, 
                     y = 0, 
                     width  = window_width,
                     height = window_height)

def place_left_bar():
    Left_bar = ttk.Notebook(root)
    Left_bar.place(x = 50, 
                    y = 22, 
                    width  = left_bar_width,
                    height = left_bar_height * 0.76)

def place_app_title():
    Title = ttk.Label(root, 
                      font = (default_font, round(20 * font_scale), "bold"),
                      foreground = "#E97451", 
                      anchor     = 'w', 
                      text       = app_name)
    Title.place(x = 50 + 50,
                y = 70,
                width  = 300,
                height = 55)

def place_itch_button():
    global logo_itch
    logo_itch = PhotoImage(file = find_by_relative_path( "Assets" 
                                                        + os.sep 
                                                        + "itch_logo.png"))

    version_button = ttk.Button(root,
                               image = logo_itch,
                               padding = '0 0 0 0',
                               text    = " " + version,
                               compound = 'left',
                               style    = 'Bold.TButton')
    version_button.place(x = (left_bar_width + 50) - (125 + 25),
                        y = 47,
                        width  = 125,
                        height = 35)
    version_button["command"] = lambda: openitch()

def place_github_button():
    global logo_git
    logo_git = PhotoImage(file = find_by_relative_path( "Assets" 
                                                        + os.sep 
                                                        + "github_logo.png"))

    ft = tkFont.Font(family = default_font)
    Butt_Style = ttk.Style()
    Butt_Style.configure("Bold.TButton", font = ft)

    github_button = ttk.Button(root,
                               image = logo_git,
                               padding = '0 0 0 0',
                               text    = ' Github',
                               compound = 'left',
                               style    = 'Bold.TButton')
    github_button.place(x = (left_bar_width + 50) - (125 + 25),
                        y = 92,
                        width  = 125,
                        height = 35)
    github_button["command"] = lambda: opengithub()

def place_paypal_button():
    global logo_paypal
    logo_paypal = PhotoImage(file=find_by_relative_path("Assets" 
                                                        + os.sep 
                                                        + "paypal_logo.png"))

    ft = tkFont.Font(family = default_font)
    Butt_Style = ttk.Style()
    Butt_Style.configure("Bold.TButton", font = ft)

    paypal_button = ttk.Button(root,
                                image = logo_paypal,
                                padding = '0 0 0 0',
                                text = ' Paypal',
                                compound = 'left',
                                style    = 'Bold.TButton')
    paypal_button.place(x = (left_bar_width + 50) - (125 + 25),
                            y = 137,
                            width  = 125,
                            height = 35)
    paypal_button["command"] = lambda: openpaypal()

def place_AI_models_title():
    IA_selection_title = ttk.Label(root, 
                                   font = (default_font, round(12 * font_scale), "bold"), 
                                   foreground = text_color, 
                                   justify    = 'left', 
                                   relief     = 'flat', 
                                   text       = " ◪  AI model ")
    IA_selection_title.place(x = left_bar_width/2 - 115,
                             y = button_1_y - 45,
                             width  = 200,
                             height = 40)

def place_resize_factor_title():
    Upscale_fact_selection_title = ttk.Label(root, 
                                            font = (default_font, round(12 * font_scale), "bold"), 
                                            foreground = text_color, 
                                            justify    = 'left', 
                                            relief     = 'flat', 
                                            text       = " ⤮  Resize before upscaling ")
    Upscale_fact_selection_title.place(x = left_bar_width/2 - 115,
                                        y = button_2_y - 45,
                                        width  = 300,
                                        height = 40)

def place_backend_title():
    Upscale_backend_selection_title = ttk.Label(root, 
                                                font = (default_font, round(12 * font_scale), "bold"), 
                                                foreground = text_color, 
                                                justify    = 'left', 
                                        	    relief     = 'flat', 
                                                text       = " ⍚  AI backend ")
    Upscale_backend_selection_title.place(x = left_bar_width/2 - 115,
                                          y = button_3_y - 45,
                                          width  = 200,
                                          height = 40)

def place_VRAM_title():
    IA_selection_title = ttk.Label(root, 
                                   font = (default_font, round(12 * font_scale), "bold"), 
                                   foreground = text_color, 
                                   justify    = 'left', 
                                   relief     = 'flat', 
                                   text       = " ⋈  Gpu Vram + PC Ram ")
    IA_selection_title.place(x = left_bar_width/2 - 115,
                             y = button_4_y - 45,
                             width  = 250,
                             height = 40)

def place_message_box():
    message_label = ttk.Label(root,
                            font = (default_font, round(11 * font_scale), "bold"),
                            textvar    = info_string,
                            relief     = "flat",
                            justify    = "center",
                            background = background_color,
                            foreground = "#ffbf00",
                            anchor     = "center")
    message_label.place(x = 50 + left_bar_width/2 - left_bar_width/2,
                        y = 615,
                        width  = left_bar_width,
                        height = 30)

    info_string.set("...")


# ---------------------- /GUI related ----------------------

# ---------------------- /Functions ----------------------


def apply_windows_dark_bar(window_root):
    window_root.update()
    DWMWA_USE_IMMERSIVE_DARK_MODE = 20
    set_window_attribute          = ctypes.windll.dwmapi.DwmSetWindowAttribute
    get_parent                    = ctypes.windll.user32.GetParent
    hwnd                          = get_parent(window_root.winfo_id())
    rendering_policy              = DWMWA_USE_IMMERSIVE_DARK_MODE
    value                         = 2
    value                         = ctypes.c_int(value)
    set_window_attribute(hwnd, rendering_policy, ctypes.byref(value), ctypes.sizeof(value))    

    #Changes the window size
    window_root.geometry(str(window_root.winfo_width()+1) + "x" + str(window_root.winfo_height()+1))
    #Returns to original size
    window_root.geometry(str(window_root.winfo_width()-1) + "x" + str(window_root.winfo_height()-1))

def apply_windows_transparency_effect(window_root):
    window_root.wm_attributes("-transparent", background_color)
    hwnd = ctypes.windll.user32.GetParent(window_root.winfo_id())
    ApplyMica(hwnd, MICAMODE.DARK )

class App:
    def __init__(self, root):
        sv_ttk.use_dark_theme()
        
        root.title('')
        width        = window_width
        height       = window_height
        screenwidth  = root.winfo_screenwidth()
        screenheight = root.winfo_screenheight()
        alignstr     = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        root.geometry(alignstr)
        root.resizable(width=False, height=False)
        root.iconphoto(False, PhotoImage(file = find_by_relative_path("Assets" 
                                                                    + os.sep 
                                                                    + "logo.png")))

        if windows_subversion >= 22000: # Windows 11
            apply_windows_transparency_effect(root)
        apply_windows_dark_bar(root)

        place_background()               # Background
        place_left_bar()                 # Left bar background
        place_app_title()                # App title
        place_itch_button()
        place_github_button()
        place_paypal_button()
        place_AI_models_title()          # AI models title
        place_AI_combobox()              # AI models widget
        place_resize_factor_title()      # Upscale factor title
        place_resize_factor_entrybox()   # Upscale factor widget
        place_backend_title()            # Backend title
        place_backend_combobox()         # Backend widget
        place_VRAM_title()               # VRAM title
        place_VRAM_combobox()            # VRAM widget
        place_message_box()              # Message box
        place_upscale_button()           # Upscale button
        place_drag_drop_widget()         # Drag&Drop widget
        

if __name__ == "__main__":
    multiprocessing.freeze_support()

    root        = tkinterDnD.Tk()
    info_string = tk.StringVar()
    selected_AI = tk.StringVar()
    selected_resize_factor = tk.StringVar()
    selected_backend = tk.StringVar()
    selected_VRAM    = tk.StringVar()

    app = App(root)
    root.update()
    root.mainloop()
