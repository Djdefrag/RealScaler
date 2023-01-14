import collections
import ctypes
import itertools
import math
import multiprocessing
import os
import os.path
import platform
import shutil
import sys
import threading
import time
import tkinter as tk
import tkinter.font as tkFont
import warnings
import webbrowser
from itertools import repeat
from math import ceil, floor, sqrt
from multiprocessing.pool import ThreadPool
from timeit import default_timer as timer
from tkinter import PhotoImage, ttk

import cv2
import numpy as np
import tkinterDnD
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch_directml
from moviepy.editor import VideoFileClip
from moviepy.video.io import ImageSequenceClip
from PIL import Image
from win32mica import MICAMODE, ApplyMica

import sv_ttk

version  = "v. 10.0"

# totally reworked tiles/join function (when image/frame does not fit entirely in VRAM)
#   tiles cut and join is now faster for images and frames
# in addition to a general improvement of the tiles, for videos, the management of tiles has been completely revised 
#   this will especially help gpu with low vram
#   and it will accelerate the upscale of high-resolution video
# reworked upscaled video reconstruction
#   it is now multithreading based on value of "Cpu number" widget 
#   is now also faster and lighter
# reworked images/frames resize 
#   now faster
# fixed a bug when extracting video frames that led to 
#   the extraction of more frames than were actually present in the video
# GUI improvements
# code cleaning & removed unused code
# optimized packages imports
# updated libraries

global app_name
app_name = "RealESRScaler"

image_path            = "no file"
models_array          = [ 'RealESRGAN_plus_x4', 'RealESR_generalv3_x4' ]
AI_model              = models_array[0]
denoise_strength      = 1

device                = 0
input_video_path      = ""
target_file_extension = ".png"
file_extension_list   = [ '.png', '.jpg', '.jp2', '.bmp', '.tiff' ]
half_precision        = True
single_file           = False
multiple_files        = False
video_files           = False
multi_img_list        = []
video_frames_list     = []
frames_upscaled_list  = []
vram_multiplier       = 1
default_vram_limiter  = 8
multiplier_num_tiles  = 2
cpu_number            = 4
interpolation_mode    = cv2.INTER_AREA
windows_subversion    = int(platform.version().split('.')[2])
compatible_gpus       = torch_directml.device_count()

device_list_names     = []
device_list           = []

class Gpu:
    def __init__(self, name, index):
        self.name = name
        self.index = index

for index in range(compatible_gpus): 
    gpu = Gpu(name = torch_directml.device_name(index), index = index)
    device_list.append(gpu)
    device_list_names.append(gpu.name)

torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)

githubme           = "https://github.com/Djdefrag/ReSRScaler"
itchme             = "https://jangystudio.itch.io/realesrscaler"

default_font          = 'Segoe UI'
background_color      = "#181818"
window_width          = 1300
window_height         = 850
left_bar_width        = 410
left_bar_height       = window_height
drag_drop_width       = window_width - left_bar_width
drag_drop_height      = window_height
show_image_width      = drag_drop_width * 0.8
show_image_height     = drag_drop_width * 0.6
image_text_width      = drag_drop_width * 0.8
support_button_height = 95 
button1_y             = 200
button2_y             = button1_y + 90
button3_y             = button2_y + 90
button4_y             = button3_y + 90
button5_y             = button4_y + 90
button6_y             = button5_y + 90

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

# ------------------- Slice functions -------------------


class Tile(object):
    def __init__(self, image, number, position, coords, filename=None):
        self.image = image
        self.number = number
        self.position = position
        self.coords = coords
        self.filename = filename

    @property
    def row(self): return self.position[0]

    @property
    def column(self): return self.position[1]

    @property
    def basename(self): return get_basename(self.filename)

    def generate_filename(
        self, directory=os.getcwd(), prefix="tile", format="png", path=True
    ):
        filename = prefix + "_{col:02d}_{row:02d}.{ext}".format(
            col=self.column, row=self.row, ext=format.lower().replace("jpeg", "jpg")
        )
        if not path: return filename
        return os.path.join(directory, filename)

    def save(self, filename=None, format="png"):
        if not filename: filename = self.generate_filename(format=format)
        self.image.save(filename, format)
        self.filename = filename

    def __repr__(self):
        """Show tile number, and if saved to disk, filename."""
        if self.filename:
            return "<Tile #{} - {}>".format(
                self.number, os.path.basename(self.filename)
            )
        return "<Tile #{}>".format(self.number)

def get_basename(filename):
    return os.path.splitext(os.path.basename(filename))[0]

def calc_columns_rows(n):
    """
    Calculate the number of columns and rows required to divide an image
    into ``n`` parts.
    Return a tuple of integers in the format (num_columns, num_rows)
    """
    num_columns = int(ceil(sqrt(n)))
    num_rows = int(ceil(n / float(num_columns)))
    return (num_columns, num_rows)

def get_combined_size(tiles):
    """Calculate combined size of tiles."""
    # TODO: Refactor calculating layout to avoid repetition.
    columns, rows = calc_columns_rows(len(tiles))
    tile_size = tiles[0].image.size
    return (tile_size[0] * columns, tile_size[1] * rows)

def join(tiles, width=0, height=0):
    """
    @param ``tiles`` - Tuple of ``Image`` instances.
    @param ``width`` - Optional, width of combined image.
    @param ``height`` - Optional, height of combined image.
    @return ``Image`` instance.
    """
    # Don't calculate size if width and height are provided
    # this allows an application that knows what the
    # combined size should be to construct an image when
    # pieces are missing.

    if width > 0 and height > 0:
        im = Image.new("RGBA", (width, height), None)
    else:
        im = Image.new("RGBA", get_combined_size(tiles), None)
    columns, rows = calc_columns_rows(len(tiles))
    for tile in tiles:
        try:
            im.paste(tile.image, tile.coords)
        except IOError:
            # do nothing, blank out the image
            continue
    return im

def validate_image(image, number_tiles):
    """Basic sanity checks prior to performing a split."""
    TILE_LIMIT = 99 * 99

    try:
        number_tiles = int(number_tiles)
    except BaseException:
        raise ValueError("number_tiles could not be cast to integer.")

    if number_tiles > TILE_LIMIT or number_tiles < 2:
        raise ValueError(
            "Number of tiles must be between 2 and {} (you \
                          asked for {}).".format(
                TILE_LIMIT, number_tiles
            )
        )

def validate_image_col_row(image, col, row):
    """Basic checks for columns and rows values"""
    SPLIT_LIMIT = 99

    try:
        col = int(col)
        row = int(row)
    except BaseException:
        raise ValueError("columns and rows values could not be cast to integer.")

    if col < 1 or row < 1 or col > SPLIT_LIMIT or row > SPLIT_LIMIT:
        raise ValueError(
            f"Number of columns and rows must be between 1 and"
            f"{SPLIT_LIMIT} (you asked for rows: {row} and col: {col})."
        )
    if col == 1 and row == 1:
        raise ValueError("There is nothing to divide. You asked for the entire image.")

def img_cutter(filename, number_tiles=None, col=None, row=None, save=True, DecompressionBombWarning=True):
    if DecompressionBombWarning is False:
        Image.MAX_IMAGE_PIXELS = None

    im = Image.open(filename)
    im_w, im_h = im.size

    columns = 0
    rows = 0
    if number_tiles:
        validate_image(im, number_tiles)
        columns, rows = calc_columns_rows(number_tiles)
    else:
        validate_image_col_row(im, col, row)
        columns = col
        rows = row

    tile_w, tile_h = int(floor(im_w / columns)), int(floor(im_h / rows))

    tiles = []
    number = 1
    for pos_y in range(0, im_h - rows, tile_h):  # -rows for rounding error.
        for pos_x in range(0, im_w - columns, tile_w):  # as above.
            area = (pos_x, pos_y, pos_x + tile_w, pos_y + tile_h)
            image = im.crop(area)
            position = (int(floor(pos_x / tile_w)) + 1, int(floor(pos_y / tile_h)) + 1)
            coords = (pos_x, pos_y)
            tile = Tile(image, number, position, coords)
            tiles.append(tile)
            number += 1
    if save:
        save_tiles(tiles, prefix=get_basename(filename), directory=os.path.dirname(filename))
    return tiles

def save_tiles(tiles, prefix="", directory=os.getcwd(), format="png"):
    for tile in tiles:
        tile.save(
            filename=tile.generate_filename(
                prefix=prefix, directory=directory, format=format
            ),
            format=format,
        )
    return tuple(tiles)

def reunion_image(tiles):
    image = join(tiles)
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return image

# ------------------ / Slice functions ------------------

# ------------------------ Utils ------------------------


def opengithub(): webbrowser.open(githubme, new=1)

def openitch(): webbrowser.open(itchme, new=1)

def create_temp_dir(name_dir):
    if os.path.exists(name_dir): shutil.rmtree(name_dir)
    if not os.path.exists(name_dir): os.makedirs(name_dir)

def find_by_relative_path(relative_path):
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
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
                                   interpolation = interpolation_mode)
        cv2.imwrite("temp.png", resized_image)
        return "temp.png"
    else:
        new_width        = round(old_image.shape[1])
        new_height       = round(old_image.shape[0])
        resized_image    = cv2.resize(old_image,
                                   (new_width, new_height),
                                   interpolation = interpolation_mode)
        cv2.imwrite("temp.png", resized_image)
        return "temp.png"

def prepare_output_filename(img, AI_model, target_file_extension):
    result_path = (img.replace("_resized" + target_file_extension, "").replace(target_file_extension, "") 
                    + "_"  + AI_model + target_file_extension)
    return result_path

def delete_list_of_files(list_to_delete):
    if len(list_to_delete) > 0:
        for to_delete in list_to_delete:
            if os.path.exists(to_delete):
                os.remove(to_delete)

def write_in_log_file(text_to_insert):
    log_file_name   = app_name + ".log"
    with open(log_file_name,'w') as log_file: log_file.write(text_to_insert) 
    log_file.close()

def read_log_file():
    log_file_name   = app_name + ".log"
    with open(log_file_name,'r') as log_file: step = log_file.readline()
    log_file.close()
    return step


# IMAGE

def resize_image(image_path, resize_factor, target_file_extension):
    new_image_path = (os.path.splitext(image_path)[0] + "_resized" + target_file_extension).strip()

    old_image = cv2.imread(image_path.strip(), cv2.IMREAD_UNCHANGED)
    new_width = int(old_image.shape[1] * resize_factor)
    new_height = int(old_image.shape[0] * resize_factor)

    resized_image = cv2.resize(old_image, (new_width, new_height), 
                                interpolation = interpolation_mode)    
    cv2.imwrite(new_image_path, resized_image)

def resize_image_list(image_list, resize_factor, target_file_extension):
    files_to_delete   = []
    downscaled_images = []
    how_much_images = len(image_list)

    index = 1
    for image in image_list:
        resized_image_path = (os.path.splitext(image)[0] + "_resized" + target_file_extension).strip()
        
        resize_image(image.strip(), resize_factor, target_file_extension)
        write_in_log_file("Resizing image " + str(index) + "/" + str(how_much_images)) 

        downscaled_images.append(resized_image_path)
        files_to_delete.append(resized_image_path)

        index += 1

    return downscaled_images, files_to_delete


#VIDEO

def extract_frames_from_video(video_path):
    video_frames_list = []
    cap          = cv2.VideoCapture(video_path)
    frame_rate   = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    # extract frames
    video = VideoFileClip(video_path)
    img_sequence = app_name + "_temp" + os.sep + "frame_%01d" + '.jpg'
    video_frames_list = video.write_images_sequence(img_sequence, logger = 'bar', fps = frame_rate)
    
    # extract audio
    try:
        video.audio.write_audiofile(app_name + "_temp" + os.sep + "audio.mp3")
    except Exception as e:
        pass

    return video_frames_list

def video_reconstruction_by_frames(input_video_path, frames_upscaled_list, AI_model, cpu_number):
    cap          = cv2.VideoCapture(input_video_path)
    frame_rate   = int(cap.get(cv2.CAP_PROP_FPS))
    path_as_list = input_video_path.split("/")
    video_name   = str(path_as_list[-1])
    only_path    = input_video_path.replace(video_name, "")
    for video_type in supported_video_list: video_name = video_name.replace(video_type, "")
    upscaled_video_path = (only_path + video_name + "_" + AI_model + ".mp4")
    cap.release()

    audio_file = app_name + "_temp" + os.sep + "audio.mp3"

    clip = ImageSequenceClip.ImageSequenceClip(frames_upscaled_list, fps = frame_rate)
    if os.path.exists(audio_file):
        clip.write_videofile(upscaled_video_path,
                            fps = frame_rate,
                            audio   = audio_file,
                            threads = cpu_number)
    else:
        clip.write_videofile(upscaled_video_path,
                            fps = frame_rate,
                            threads = cpu_number)     

def resize_frame(image_path, new_width, new_height, target_file_extension):
    new_image_path = image_path.replace('.jpg', "" + target_file_extension)
    
    old_image = cv2.imread(image_path.strip(), cv2.IMREAD_UNCHANGED)

    resized_image = cv2.resize(old_image, (new_width, new_height), 
                                interpolation = interpolation_mode)    
    cv2.imwrite(new_image_path, resized_image)

def resize_frame_list(image_list, resize_factor, target_file_extension, cpu_number):
    downscaled_images = []

    old_image = Image.open(image_list[1])
    new_width, new_height = old_image.size
    new_width = int(new_width * resize_factor)
    new_height = int(new_height * resize_factor)
    
    with ThreadPool(cpu_number) as pool:
        pool.starmap(resize_frame, zip(image_list, 
                                    itertools.repeat(new_width), 
                                    itertools.repeat(new_height), 
                                    itertools.repeat(target_file_extension)))

    for image in image_list:
        resized_image_path = image.replace('.jpg', "" + target_file_extension)
        downscaled_images.append(resized_image_path)

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


# ------------------ /AI related ------------------


# ----------------------- Core ------------------------


def thread_check_steps_for_images( not_used_var, not_used_var2 ):
    time.sleep(3)
    try:
        while True:
            step = read_log_file()
            if "Upscale completed" in step or "Error while upscaling" in step or "Stopped upscaling" in step:
                info_string.set(step)
                stop = 1 + "x"
            info_string.set(step)
            time.sleep(2)
    except:
        place_upscale_button()

def thread_check_steps_for_videos( not_used_var, not_used_var2 ):
    time.sleep(3)
    try:
        while True:
            step = read_log_file()
            if "Upscale video completed" in step or "Error while upscaling" in step or "Stopped upscaling" in step:
                info_string.set(step)
                stop = 1 + "x"
            info_string.set(step)
            time.sleep(2)
    except:
        place_upscale_button()



def enhance(model, img, backend, half_precision):
    img = img.astype(np.float32)

    if np.max(img) > 256: max_range = 65535 # 16 bit images
    else: max_range = 255

    img = img / max_range
    if len(img.shape) == 2:  # gray image
        img_mode = 'L'
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:  # RGBA image with alpha channel
        img_mode = 'RGBA'
        alpha = img[:, :, 3]
        img = img[:, :, 0:3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2RGB)
    else:
        img_mode = 'RGB'
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ------------------- process image (without the alpha channel) ------------------- #
    
    img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
    img = img.unsqueeze(0).to(backend)
    if half_precision: img = img.half()

    output = model(img) ## model
    
    output_img = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output_img = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0))

    if img_mode == 'L':  output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)

    # ------------------- process the alpha channel if necessary ------------------- #
    
    if img_mode == 'RGBA':
        alpha = torch.from_numpy(np.transpose(alpha, (2, 0, 1))).float()
        alpha = alpha.unsqueeze(0).to(backend)
        if half_precision: alpha = alpha.half()

        output_alpha = model(alpha) ## model

        output_alpha = output_alpha.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output_alpha = np.transpose(output_alpha[[2, 1, 0], :, :], (1, 2, 0))
        output_alpha = cv2.cvtColor(output_alpha, cv2.COLOR_BGR2GRAY)

        # merge the alpha channel
        output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2BGRA)
        output_img[:, :, 3] = output_alpha

    # ------------------------------ return ------------------------------ #
    if max_range == 65535: output = (output_img * 65535.0).round().astype(np.uint16) # 16-bit image
    else: output = (output_img * 255.0).round().astype(np.uint8)

    return output, img_mode

def prepare_model(AI_model, device, half_precision):
    backend = torch.device(torch_directml.device(device))

    model_path = find_by_relative_path("AI" + os.sep + AI_model + ".pth")

    if 'RealESRGAN_plus_x4' in AI_model: 
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, 
                        num_block=23, num_grow_ch=32, scale=4)
    elif 'RealESR_generalv3_x4' in AI_model: 
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, 
                                num_conv=32, upscale=4, act_type='prelu')

    loadnet = torch.load(model_path, map_location = torch.device('cpu'))

    if 'params_ema' in loadnet: keyname = 'params_ema'
    else: keyname = 'params'
    model.load_state_dict(loadnet[keyname], strict=True)
    model.eval()
    
    model = model.to(backend)
    if half_precision: model = model.half()
        
    return model



def reverse_split_multiple_frames(list_of_tiles_list, frames_upscaled_list):
    for index in range(len(frames_upscaled_list)):
        cv2.imwrite(frames_upscaled_list[index], reunion_image(list_of_tiles_list[index]))              

def upscale_frame_and_save(frame, model, result_path, 
                            tiles_resolution, device, 
                            half_precision, list_of_tiles_list):

    used_tiles       = False
    backend          = torch.device(torch_directml.device(device))

    img_tmp          = cv2.imread(frame)
    image_resolution = max(img_tmp.shape[1], img_tmp.shape[0])
    num_tiles        = image_resolution/tiles_resolution

    if num_tiles <= 1:
        with torch.no_grad():
            img_adapted     = cv2.imread(frame, cv2.IMREAD_UNCHANGED)
            img_upscaled, _ = enhance(model, img_adapted, backend, half_precision)
            cv2.imwrite(result_path, img_upscaled)
    else:
        used_tiles = True

        num_tiles  = round(num_tiles)
        if (num_tiles % 2) != 0: num_tiles += 1
        num_tiles  = round(num_tiles * multiplier_num_tiles)

        tiles = img_cutter(frame, num_tiles)
        with torch.no_grad():
            for tile in tiles:
                tile_adapted  = cv2.imread(tile.filename, cv2.IMREAD_UNCHANGED)
                tile_upscaled, _ = enhance(model, tile_adapted, backend, half_precision)
                cv2.imwrite(tile.filename, tile_upscaled)
                tile.image = Image.open(tile.filename)
                tile.coords = (tile.coords[0] * 4, 
                                tile.coords[1] * 4)

        list_of_tiles_list.append(tiles)

    return list_of_tiles_list, used_tiles

def process_upscale_video_frames(input_video_path, AI_model, resize_factor, device,
                                tiles_resolution, target_file_extension, cpu_number,
                                half_precision):
    try:
        start = timer()

        write_in_log_file('...')
        if AI_model == "RealESR_generalv3_x4": half_precision = False
        create_temp_dir(app_name + "_temp")

        write_in_log_file('Extracting video frames...')
        frame_list = extract_frames_from_video(input_video_path)
        print("> Extracted frames: " + str(len(frame_list)))
        
        write_in_log_file('Resizing video frames...')
        frame_list  = resize_frame_list(frame_list, resize_factor, target_file_extension, cpu_number)
        print("> Resized frames: " + str(len(frame_list)))


        write_in_log_file('Upscaling...')
        how_many_images = len(frame_list)
        done_images     = 0
        frames_upscaled_list = []
        list_of_tiles_list   = []

        model = prepare_model(AI_model, device, half_precision)

        for frame in frame_list:
            result_path = prepare_output_filename(frame, AI_model, target_file_extension)
            frames_upscaled_list.append(result_path)

            list_of_tiles_list, used_tiles = upscale_frame_and_save(frame, 
                                                                    model, 
                                                                    result_path, 
                                                                    tiles_resolution, 
                                                                    device, 
                                                                    half_precision,
                                                                    list_of_tiles_list)
            done_images += 1
            write_in_log_file("Upscaled frame " + str(done_images) + "/" + str(how_many_images))

        if used_tiles: 
            write_in_log_file("Reconstructing frames from tiles...")
            reverse_split_multiple_frames(list_of_tiles_list, frames_upscaled_list)

        print("> Upscaled frames: " + str(len(frames_upscaled_list)))

        write_in_log_file("Processing upscaled video...")
        video_reconstruction_by_frames(input_video_path, frames_upscaled_list, AI_model, cpu_number)

        write_in_log_file("Upscale video completed [" + str(round(timer() - start)) + " sec.]")

        create_temp_dir(app_name + "_temp")

    except Exception as e:
        write_in_log_file('Error while upscaling' + '\n\n' + str(e)) 
        import tkinter as tk
        error_root = tk.Tk()
        error_root.withdraw()
        tk.messagebox.showerror(title   = 'Error', 
                                message = 'Upscale failed caused by:\n\n' +
                                           str(e) + '\n\n' +
                                          'Please report the error on Github.com or Itch.io.' +
                                          '\n\nThank you :)')



def upscale_image_and_save(img, model, result_path, 
                            tiles_resolution, 
                            upscale_factor, 
                            device, half_precision):
    img_tmp          = cv2.imread(img)
    image_resolution = max(img_tmp.shape[1], img_tmp.shape[0])
    num_tiles        = image_resolution/tiles_resolution
    backend          = torch.device(torch_directml.device(device))

    if num_tiles <= 1:
        with torch.no_grad():
            img_adapted     = cv2.imread(img, cv2.IMREAD_UNCHANGED)
            img_upscaled, _ = enhance(model, img_adapted, backend, half_precision)
            cv2.imwrite(result_path, img_upscaled)
    else:
        num_tiles = round(num_tiles)
        if (num_tiles % 2) != 0: num_tiles += 1
        num_tiles = round(num_tiles * multiplier_num_tiles)

        tiles = img_cutter(img, num_tiles)
        with torch.no_grad():
            for tile in tiles:
                tile_adapted  = cv2.imread(tile.filename, cv2.IMREAD_UNCHANGED)
                tile_upscaled, _ = enhance(model, tile_adapted, backend, half_precision)
                cv2.imwrite(tile.filename, tile_upscaled)
                tile.image = Image.open(tile.filename)
                tile.coords = (tile.coords[0] * upscale_factor, 
                                tile.coords[1] * upscale_factor)
    
        cv2.imwrite(result_path, reunion_image(tiles))

        to_delete = []
        for tile in tiles: to_delete.append(tile.filename)
        delete_list_of_files(to_delete)

def process_upscale_multiple_images(image_list, AI_model, resize_factor, device, 
                                    tiles_resolution, target_file_extension, 
                                    half_precision):
    try:
        start = timer()
        write_in_log_file('...')

        if "x2" in AI_model: upscale_factor = 2
        elif "x4" in AI_model: upscale_factor = 4 
        if AI_model == "RealESR_generalv3_x4": half_precision = False
 
        write_in_log_file('Resizing images...')
        image_list, files_to_delete = resize_image_list(image_list, resize_factor, target_file_extension)

        how_many_images = len(image_list)
        done_images     = 0

        write_in_log_file('Upscaling...')
        model = prepare_model(AI_model, device, half_precision)
        for img in image_list:
            result_path = prepare_output_filename(img, AI_model, target_file_extension)
            upscale_image_and_save(img, model, result_path, tiles_resolution, 
                                    upscale_factor, device, half_precision)
            done_images += 1
            write_in_log_file("Upscaled images " + str(done_images) + "/" + str(how_many_images))
                
        write_in_log_file("Upscale completed [" + str(round(timer() - start)) + " sec.]")

        delete_list_of_files(files_to_delete)
    except Exception as e:
        write_in_log_file('Error while upscaling' + '\n\n' + str(e)) 
        import tkinter as tk
        error_root = tk.Tk()
        error_root.withdraw()
        tk.messagebox.showerror(title   = 'Error', 
                                message = 'Upscale failed caused by:\n\n' +
                                           str(e) + '\n\n' +
                                          'Please report the error on Github.com or Itch.io.' +
                                          '\n\nThank you :)')

    
# ----------------------- /Core ------------------------


# ---------------------- GUI related ----------------------


def user_input_checks():
    global tiles_resolution
    global resize_factor
    global cpu_number

    is_ready = True

    if compatible_gpus == 0:
        tk.messagebox.showerror(title   = 'Error', 
                                message = 'Sorry, your gpu is not compatible with ReSRScaler :(')
        is_ready = False

    # resize factor
    try: resize_factor = int(float(str(selected_resize_factor.get())))
    except:
        info_string.set("Resize % must be a numeric value")
        is_ready = False

    if resize_factor > 0 and resize_factor <= 100: resize_factor = resize_factor/100
    else:
        info_string.set("Resize % must be in range 1 - 100")
        is_ready = False
    
    # vram limiter
    try: tiles_resolution = 100 * int(float(str(selected_VRAM_limiter.get())))
    except:
        info_string.set("VRAM/RAM value must be a numeric value")
        is_ready = False 

    if tiles_resolution > 0: tiles_resolution = 100 * (vram_multiplier * int(float(str(selected_VRAM_limiter.get()))))    
    else:
        info_string.set("VRAM/RAM value must be > 0")
        is_ready = False

    # cpu number
    try: cpu_number = int(float(str(selected_cpu_number.get())))
    except:
        info_string.set("Cpu number must be a numeric value")
        is_ready = False 

    if cpu_number <= 0:         
        info_string.set("Cpu number value must be > 0")
        is_ready = False
    elif cpu_number == 1: cpu_number = 1
    else: cpu_number = int(cpu_number)

    return is_ready


def upscale_button_command():
    global image_path
    global multiple_files
    global process_upscale
    global thread_wait
    global video_frames_list
    global video_files
    global frames_upscaled_list
    global input_video_path
    global device
    global tiles_resolution
    global target_file_extension
    global cpu_number
    global half_precision

    info_string.set("...")

    is_ready = user_input_checks()

    if is_ready:
        if video_files:
            place_stop_button()

            process_upscale = multiprocessing.Process(target = process_upscale_video_frames,
                                                    args   = (input_video_path, 
                                                                AI_model, 
                                                                resize_factor, 
                                                                device,
                                                                tiles_resolution,
                                                                target_file_extension,
                                                                cpu_number,
                                                                half_precision))
            process_upscale.start()

            thread_wait = threading.Thread(target = thread_check_steps_for_videos,
                                        args   = (1, 2), 
                                        daemon = True)
            thread_wait.start()

        elif multiple_files or single_file:
            place_stop_button()
            
            process_upscale = multiprocessing.Process(target = process_upscale_multiple_images,
                                                        args   = (multi_img_list, 
                                                                AI_model, 
                                                                resize_factor, 
                                                                device,
                                                                tiles_resolution,
                                                                target_file_extension,
                                                                half_precision))
            process_upscale.start()

            thread_wait = threading.Thread(target = thread_check_steps_for_images,
                                            args   = (1, 2), daemon = True)
            thread_wait.start()

        elif "none" in image_path:
            info_string.set("No file selected")
  

def stop_button_command():
    global process_upscale
    process_upscale.terminate()
    process_upscale.join()
    
    # this will stop thread that check upscaling steps
    write_in_log_file("Stopped upscaling") 



def drop_event_to_image_list(event):
    image_list = str(event.data).replace("{", "").replace("}", "")

    for file_type in supported_file_list: image_list = image_list.replace(file_type, file_type+"\n")

    image_list = image_list.split("\n")
    image_list.pop() 

    return image_list

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
                    show_video_in_GUI(input_video_path)

                    # reset variable
                    image_path = "none"
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
                # reset variable
                image_path = "none"
                video_frames_list = []

            elif single_file:
                image_list_dropped = drop_event_to_image_list(event)
                show_image_in_GUI(image_list_dropped[0])
                multi_img_list = image_list_dropped

                # reset variable
                image_path = "none"
                video_frames_list = []

def check_compatibility(supported_file_dropped_number, not_supported_file_dropped_number, 
                        supported_video_dropped_number):
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

def clear_input_variables():
    global image_path
    global multi_img_list
    global video_frames_list
    global single_file
    global multiple_files
    global video_files

    # reset variable
    image_path        = "none"
    multi_img_list    = []
    video_frames_list = []
    single_file       = False
    multiple_files    = False
    video_files       = False

def clear_app_background():
    drag_drop = ttk.Label(root,
                          ondrop = file_drop_event,
                          relief = "flat",
                          background = background_color,
                          foreground = text_color)
    drag_drop.place(x = left_bar_width + 50, y=0,
                    width = drag_drop_width, height = drag_drop_height)


def show_video_in_GUI(video_path):
    clear_app_background()
    
    fist_frame   = "temp.jpg"
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
        if ret == False: break
        cv2.imwrite(fist_frame, frame)
        break
    cap.release()

    resized_image_to_show = adapt_image_to_show(fist_frame)

    global image
    image = tk.PhotoImage(file = resized_image_to_show)
    resized_image_to_show_width = round(cv2.imread(resized_image_to_show).shape[1])
    resized_image_to_show_height = round(cv2.imread(resized_image_to_show).shape[0])
    image_x_center = 30 + left_bar_width + drag_drop_width/2 - resized_image_to_show_width/2
    image_y_center = drag_drop_height/2 - resized_image_to_show_height/2

    image_container = ttk.Notebook(root)
    image_container.place(x = image_x_center - 20, 
                            y = image_y_center - 20, 
                            width  = resized_image_to_show_width + 40,
                            height = resized_image_to_show_height + 40)

    image_ = ttk.Label(root,
                        text    = "",
                        image   = image,
                        ondrop  = file_drop_event,
                        anchor  = "center",
                        relief  = "flat",
                        justify = "center",
                        background = background_color,
                        foreground = "#202020")
    image_.place(x = image_x_center,
                y = image_y_center,
                width  = resized_image_to_show_width,
                height = resized_image_to_show_height)

    image_info_label = ttk.Label(root,
                                  font       = bold11,
                                  text       = ( video_name + "\n" + "[" + str(width) + "x" + str(height) + "]" + " | " + str(minutes) + 'm:' + str(round(seconds)) + "s | " + str(num_frames) + "frames | " + str(round(frame_rate)) + "fps" ),
                                  relief     = "flat",
                                  justify    = "center",
                                  background = background_color,
                                  foreground = "#D3D3D3",
                                  anchor     = "center")

    image_info_label.place(x = 30 + left_bar_width + drag_drop_width/2 - image_text_width/2,
                            y = drag_drop_height - 85,
                            width  = image_text_width,
                            height = 40)

    global clear_icon
    clear_icon = tk.PhotoImage(file = find_by_relative_path("Assets" + os.sep + "clear_icon.png"))

    clean_button = ttk.Button(root, 
                            text     = '  CLEAN',
                            image    = clear_icon,
                            compound = 'left',
                            style    = 'Bold.TButton')

    clean_button.place(x = image_x_center - 20,
                       y = image_y_center - 75,
                       width  = 175,
                       height = 40)
    clean_button["command"] = lambda: place_drag_drop_widget()          

    os.remove(fist_frame)
              
def show_list_images_in_GUI(image_list):
    clear_app_background()

    global clear_icon
    clear_icon = tk.PhotoImage(file = find_by_relative_path("Assets"  + os.sep  + "clear_icon.png"))
    clean_button = ttk.Button(root, 
                            text     = '  CLEAN',
                            image    = clear_icon,
                            compound = 'left',
                            style    = 'Bold.TButton')

    clean_button.place(x = left_bar_width + drag_drop_width/2 - 175/2,
                       y = 125,
                       width  = 175,
                       height = 40)
    clean_button["command"] = lambda: place_drag_drop_widget()

    final_string = "\n"
    counter_img = 0
    for elem in image_list:
        counter_img += 1
        if counter_img <= 8:
            img     = cv2.imread(elem.strip())
            width   = round(img.shape[1])
            height  = round(img.shape[0])
            img_name = str(elem.split("/")[-1])

            final_string += (str(counter_img) + ".  " + img_name + " | [" + str(width) + "x" + str(height) + "]" + "\n\n")
        else:
            final_string += "and others... \n"
            break

    list_height = 420
    list_width  = 750

    images_list_label = ttk.Label(root,
                            text    = final_string,
                            ondrop  = file_drop_event,
                            font    = bold12,
                            anchor  = "n",
                            relief  = "flat",
                            justify = "left",
                            background = background_color,
                            foreground = "#D3D3D3",
                            wraplength = list_width)

    images_list_label.place(x = left_bar_width + drag_drop_width/2 - list_width/2,
                               y = drag_drop_height/2 - list_height/2 -25,
                               width  = list_width,
                               height = list_height)

    images_counter = ttk.Entry(root, 
                                foreground = text_color,
                                ondrop     = file_drop_event,
                                font       = bold12, 
                                justify    = 'center')
    images_counter.insert(0, str(len(image_list)) + ' images')
    images_counter.configure(state='disabled')
    images_counter.place(x = left_bar_width + drag_drop_width/2 - 175/2,
                        y  = drag_drop_height/2 + 225,
                        width  = 200,
                        height = 42)

def show_image_in_GUI(original_image):
    clear_app_background()
    original_image = original_image.replace('{', '').replace('}', '')
    resized_image_to_show = adapt_image_to_show(original_image)

    global image
    image = tk.PhotoImage(file = resized_image_to_show)
    resized_image_to_show_width = round(cv2.imread(resized_image_to_show).shape[1])
    resized_image_to_show_height = round(cv2.imread(resized_image_to_show).shape[0])
    image_x_center = 30 + left_bar_width + drag_drop_width/2 - resized_image_to_show_width/2
    image_y_center = drag_drop_height/2 - resized_image_to_show_height/2

    image_container = ttk.Notebook(root)
    image_container.place(x = image_x_center - 20, 
                            y = image_y_center - 20, 
                            width  = resized_image_to_show_width + 40,
                            height = resized_image_to_show_height + 40)

    image_ = ttk.Label(root,
                        text    = "",
                        image   = image,
                        ondrop  = file_drop_event,
                        anchor  = "center",
                        relief  = "flat",
                        justify = "center",
                        background = background_color,
                        foreground = "#202020")
    image_.place(x = image_x_center,
                      y = image_y_center,
                      width  = resized_image_to_show_width,
                      height = resized_image_to_show_height)

    img_name     = str(original_image.split("/")[-1])
    width        = round(cv2.imread(original_image).shape[1])
    height       = round(cv2.imread(original_image).shape[0])

    image_info_label = ttk.Label(root,
                                  font       = bold11,
                                  text       = (img_name + " | [" + str(width) + "x" + str(height) + "]"),
                                  relief     = "flat",
                                  justify    = "center",
                                  background = background_color,
                                  foreground = "#D3D3D3",
                                  anchor     = "center")

    image_info_label.place(x = 30 + left_bar_width + drag_drop_width/2 - image_text_width/2,
                            y = drag_drop_height - 70,
                            width  = image_text_width,
                            height = 40)

    global clear_icon
    clear_icon = tk.PhotoImage(file = find_by_relative_path("Assets" + os.sep + "clear_icon.png"))

    clean_button = ttk.Button(root, 
                            text     = '  CLEAN',
                            image    = clear_icon,
                            compound = 'left',
                            style    = 'Bold.TButton')

    clean_button.place(x = image_x_center - 20,
                       y = image_y_center - 75,
                       width  = 175,
                       height = 40)
    clean_button["command"] = lambda: place_drag_drop_widget()


def place_drag_drop_widget():
    clear_input_variables()

    clear_app_background()

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
                            font    = bold12,
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


def combobox_AI_selection(event):
    global AI_model
    selected = str(selected_AI.get())
    AI_model = selected
    combo_box_AI.set('')
    combo_box_AI.set(selected)

def combobox_backend_selection(event):
    global device

    selected_option = str(selected_backend.get())
    combo_box_backend.set('')
    combo_box_backend.set(selected_option)

    for obj in device_list:
        if obj.name == selected_option:
            device = obj.index

def combobox_extension_selection(event):
    global target_file_extension
    selected = str(selected_file_extension.get()).strip()
    target_file_extension = selected
    combobox_file_extension.set('')
    combobox_file_extension.set(selected)


def place_AI_combobox():
    Ai_container = ttk.Notebook(root)
    Ai_container.place(x = 45 + left_bar_width/2 - 370/2, 
                        y = button1_y - 17, 
                        width  = 370,
                        height = 75)

    Ai_label = ttk.Label(root, 
                    font       = bold11, 
                    foreground = text_color, 
                    justify    = 'left', 
                    relief     = 'flat', 
                    text       = " AI model ")
    Ai_label.place(x = 90,
                    y = button1_y - 2,
                    width  = 130,
                    height = 42)

    global combo_box_AI
    combo_box_AI = ttk.Combobox(root, 
                        textvariable = selected_AI, 
                        justify      = 'center',
                        foreground   = text_color,
                        values       = models_array,
                        state        = 'readonly',
                        takefocus    = False,
                        font         = bold10)
    combo_box_AI.place(x = 10 + left_bar_width/2, 
                        y = button1_y, 
                        width  = 200, 
                        height = 40)
    combo_box_AI.bind('<<ComboboxSelected>>', combobox_AI_selection)
    combo_box_AI.set(AI_model)

def place_backend_combobox():
    backend_container = ttk.Notebook(root)
    backend_container.place(x = 45 + left_bar_width/2 - 370/2, 
                            y = button2_y - 17, 
                            width  = 370,
                            height = 75)

    backend_label = ttk.Label(root, 
                            font       = bold11, 
                            foreground = text_color, 
                            justify    = 'left', 
                            relief     = 'flat', 
                            text       = " AI backend  ")
    backend_label.place(x = 90,
                        y = button2_y - 2,
                        width  = 155,
                        height = 42)

    global combo_box_backend
    combo_box_backend = ttk.Combobox(root, 
                            textvariable = selected_backend, 
                            justify      = 'center',
                            foreground   = text_color,
                            values       = device_list_names,
                            state        = 'readonly',
                            takefocus    = False,
                            font         = bold10)
    combo_box_backend.place(x = 10 + left_bar_width/2, 
                            y = button2_y, 
                            width  = 200, 
                            height = 40)
    combo_box_backend.bind('<<ComboboxSelected>>', combobox_backend_selection)
    combo_box_backend.set(device_list_names[0])

def place_file_extension_combobox():
    file_extension_container = ttk.Notebook(root)
    file_extension_container.place(x = 45 + left_bar_width/2 - 370/2, 
                        y = button3_y - 17, 
                        width  = 370,
                        height = 75)

    file_extension_label = ttk.Label(root, 
                        font       = bold11, 
                        foreground = text_color, 
                        justify    = 'left', 
                        relief     = 'flat', 
                        text       = " AI output extension ")
    file_extension_label.place(x = 90,
                            y = button3_y - 2,
                            width  = 155,
                            height = 42)

    global combobox_file_extension
    combobox_file_extension = ttk.Combobox(root, 
                        textvariable = selected_file_extension, 
                        justify      = 'center',
                        foreground   = text_color,
                        values       = file_extension_list,
                        state        = 'readonly',
                        takefocus    = False,
                        font         = bold11)
    combobox_file_extension.place(x = 65 + left_bar_width/2, 
                        y = button3_y, 
                        width  = 145, 
                        height = 40)
    combobox_file_extension.bind('<<ComboboxSelected>>', combobox_extension_selection)
    combobox_file_extension.set('.png')


def place_resize_factor_spinbox():
    resize_factor_container = ttk.Notebook(root)
    resize_factor_container.place(x = 45 + left_bar_width/2 - 370/2, 
                               y = button4_y - 17, 
                               width  = 370,
                               height = 75)

    global spinbox_resize_factor
    spinbox_resize_factor = ttk.Spinbox(root,  
                                        from_        = 1, 
                                        to           = 100, 
                                        increment    = 1,
                                        textvariable = selected_resize_factor, 
                                        justify      = 'center',
                                        foreground   = text_color,
                                        takefocus    = False,
                                        font         = bold12)
    spinbox_resize_factor.place(x = 65 + left_bar_width/2, 
                                y = button4_y, 
                                width  = 145, 
                                height = 40)
    spinbox_resize_factor.insert(0, '70')

    resize_factor_label = ttk.Label(root, 
                                    font       = bold11, 
                                    foreground = text_color, 
                                    justify    = 'left', 
                                    relief     = 'flat', 
                                    text       = " Input resolution | % ")
    resize_factor_label.place(x = 90,
                            y = button4_y - 2,
                            width  = 155,
                            height = 42)

def place_VRAM_spinbox():
    vram_container = ttk.Notebook(root)
    vram_container.place(x = 45 + left_bar_width/2 - 370/2, 
                        y = button5_y - 17, 
                        width  = 370,
                        height = 75)

    global spinbox_VRAM
    spinbox_VRAM = ttk.Spinbox(root,  
                                from_     = 1, 
                                to        = 100, 
                                increment = 1,
                                textvariable = selected_VRAM_limiter, 
                                justify      = 'center',
                                foreground   = text_color,
                                takefocus    = False,
                                font         = bold12)
    spinbox_VRAM.place(x = 65 + left_bar_width/2, 
                        y = button5_y, 
                        width  = 145, 
                        height = 40)
    spinbox_VRAM.insert(0, str(default_vram_limiter))

    vram_label = ttk.Label(root, 
                            font       = bold11, 
                            foreground = text_color, 
                            justify    = 'left', 
                            relief     = 'flat', 
                            text       = " Vram limiter | GB ")
    vram_label.place(x = 90,
                    y = button5_y - 2,
                    width  = 155,
                    height = 42)

def place_cpu_number_spinbox():
    cpu_number_container = ttk.Notebook(root)
    cpu_number_container.place(x = 45 + left_bar_width/2 - 370/2, 
                        y = button6_y - 17, 
                        width  = 370,
                        height = 75)

    global spinbox_cpus
    spinbox_cpus = ttk.Spinbox(root,  
                                from_     = 1, 
                                to        = 100, 
                                increment = 1,
                                textvariable = selected_cpu_number, 
                                justify      = 'center',
                                foreground   = text_color,
                                takefocus    = False,
                                font         = bold12)
    spinbox_cpus.place(x = 65 + left_bar_width/2, 
                        y = button6_y, 
                        width  = 145, 
                        height = 40)
    spinbox_cpus.insert(0, str(cpu_number))

    cpus_label = ttk.Label(root, 
                            font       = bold11, 
                            foreground = text_color, 
                            justify    = 'left', 
                            relief     = 'flat', 
                            text       = " Cpu number ")
    cpus_label.place(x = 90,
                    y = button6_y - 2,
                    width  = 155,
                    height = 42)


def place_advanced_option_button():
    global advanced_settings_icon
    advanced_settings_icon = tk.PhotoImage(file = find_by_relative_path("Assets" 
                                                        + os.sep 
                                                        + "advanced_settings_icon.png"))

    advance_option_button = ttk.Button(root,
                               image = advanced_settings_icon,
                               padding = '0 0 0 0',
                               text    = " OPTIONS",
                               compound = 'left',
                               style    = 'Bold.TButton')
    advance_option_button.place(x = 65,
                                y = 130,
                                width  = 150,
                                height = 40)


def place_app_title():
    Title = ttk.Label(root, 
                      font       = bold20,
                      foreground = "#FFAC1C", 
                      background = background_color,
                      anchor     = 'w', 
                      text       = app_name)
    Title.place(x = 77,
                y = 40,
                width  = 300,
                height = 55)

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
    version_button.place(x = (left_bar_width + 45) - (125 + 30),
                        y = 30,
                        width  = 125,
                        height = 35)
    version_button["command"] = lambda: openitch()

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
    github_button.place(x = (left_bar_width + 45) - (125 + 30),
                        y = 75,
                        width  = 125,
                        height = 35)
    github_button["command"] = lambda: opengithub()

def place_message_box():
    message_label = ttk.Label(root,
                            font       = bold11,
                            textvar    = info_string,
                            relief     = "flat",
                            justify    = "center",
                            background = background_color,
                            foreground = "#ffbf00",
                            anchor     = "center")
    message_label.place(x = 45 + left_bar_width/2 - left_bar_width/2,
                        y = window_height - 120,
                        width  = left_bar_width,
                        height = 30)

def place_upscale_button():
    global play_icon
    play_icon = tk.PhotoImage(file = find_by_relative_path("Assets" 
                                                        + os.sep 
                                                        + "upscale_icon.png"))
    
    button_Style = ttk.Style()
    button_Style.configure("Bold.TButton", 
                                font = bold11, 
                                foreground = text_color)

    Upscale_button = ttk.Button(root, 
                                text  = ' UPSCALE',
                                image = play_icon,
                                compound = tk.LEFT,
                                style = "Bold.TButton")

    Upscale_button.place(x      = 45 + left_bar_width/2 - 275/2,  
                         y      = left_bar_height - 80,
                         width  = 280,
                         height = 47)
    Upscale_button["command"] = lambda: upscale_button_command()

def place_stop_button():
    global stop_icon
    stop_icon = tk.PhotoImage(file = find_by_relative_path("Assets" 
                                                        + os.sep 
                                                        + "stop_icon.png"))
    
    Upsc_Butt_Style = ttk.Style()
    Upsc_Butt_Style.configure("Bold.TButton", font = bold11)

    Stop_button = ttk.Button(root, 
                                text  = '  STOP UPSCALE ',
                                image = stop_icon,
                                compound = tk.LEFT,
                                style    = 'Bold.TButton')

    Stop_button.place(x      = 45 + left_bar_width/2 - 275/2,  
                      y      = left_bar_height - 80,
                      width  = 280,
                      height = 47)

    Stop_button["command"] = lambda: stop_button_command()

def place_background(root, width, height):
    Background = ttk.Label(root, background = background_color, relief = 'flat')
    Background.place(x = 0, 
                     y = 0, 
                     width  = width,
                     height = height)


# ---------------------- /GUI related ----------------------


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
        root.iconphoto(False, PhotoImage(file = find_by_relative_path("Assets"  + os.sep + "logo.png")))

        if windows_subversion >= 22000: apply_windows_transparency_effect(root) # Windows 11
        apply_windows_dark_bar(root)

        place_background(root, window_width, window_height) # Background
        place_app_title()                                   # App title
        place_advanced_option_button()
        place_AI_combobox()                                 # AI models widget
        place_resize_factor_spinbox()                       # Upscale factor widget
        place_VRAM_spinbox()                                # VRAM widget
        place_backend_combobox()                            # Backend widget
        place_file_extension_combobox()
        place_cpu_number_spinbox()
        place_message_box()                                 # Message box
        place_upscale_button()                              # Upscale button
        place_drag_drop_widget()                            # Drag&Drop widget
        
if __name__ == "__main__":
    multiprocessing.freeze_support()

    root        = tkinterDnD.Tk()
    info_string = tk.StringVar()
    selected_AI = tk.StringVar()
    selected_resize_factor  = tk.StringVar()
    selected_VRAM_limiter   = tk.StringVar()
    selected_backend        = tk.StringVar()
    selected_file_extension = tk.StringVar()
    selected_cpu_number     = tk.StringVar()

    bold10 = tkFont.Font(family = default_font, size   = round(10 * font_scale), weight = 'bold')
    bold11 = tkFont.Font(family = default_font, size   = round(11 * font_scale), weight = 'bold')
    bold12 = tkFont.Font(family = default_font, size   = round(12 * font_scale), weight = 'bold')
    bold13 = tkFont.Font(family = default_font, size   = round(13 * font_scale), weight = 'bold')
    bold14 = tkFont.Font(family = default_font, size   = round(14 * font_scale), weight = 'bold')
    bold15 = tkFont.Font(family = default_font, size   = round(15 * font_scale), weight = 'bold')
    bold20 = tkFont.Font(family = default_font, size   = round(20 * font_scale), weight = 'bold')
    bold21 = tkFont.Font(family = default_font, size   = round(21 * font_scale), weight = 'bold')



    app = App(root)
    root.update()
    root.mainloop()