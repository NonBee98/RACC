import os

import cv2
import numpy as np
import rawpy
import torch

from .common_utils import *
from params import *
from PIL import Image, ImageDraw, ImageFont

MAX_16BIT = 65535
MAX_8BIT = 255


def simple_demosaic(img: np.ndarray, exif):
    raw_colors = np.asarray(exif['CFAPattern2']).reshape((2, 2))
    demosaiced_image = np.zeros((img.shape[0] // 2, img.shape[1] // 2, 3))
    for i in range(2):
        for j in range(2):
            ch = raw_colors[i, j]
            if ch == 1:
                demosaiced_image[:, :, ch] += img[i::2, j::2] / 2
            else:
                demosaiced_image[:, :, ch] = img[i::2, j::2]
    return demosaiced_image


def fix_orientation(image, orientation):
    # 1 = Horizontal(normal)
    # 2 = Mirror horizontal
    # 3 = Rotate 180
    # 4 = Mirror vertical
    # 5 = Mirror horizontal and rotate 270 CW
    # 6 = Rotate 90 CW
    # 7 = Mirror horizontal and rotate 90 CW
    # 8 = Rotate 270 CW
    orientation_dict = [
        "Horizontal (normal)", "Mirror horizontal", "Rotate 180",
        "Mirror vertical", "Mirror horizontal and rotate 270 CW",
        "Rotate 90 CW", "Mirror horizontal and rotate 90 CW", "Rotate 270 CW"
    ]
    orientation_dict = {v: k for k, v in enumerate(orientation_dict)}
    if isinstance(orientation, str):
        orientation = orientation_dict[orientation] + 1

    if orientation == 1:
        pass
    elif orientation == 2:
        image = cv2.flip(image, 0)
    elif orientation == 3:
        image = cv2.rotate(image, cv2.ROTATE_180)
    elif orientation == 4:
        image = cv2.flip(image, 1)
    elif orientation == 5:
        image = cv2.flip(image, 0)
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif orientation == 6:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif orientation == 7:
        image = cv2.flip(image, 0)
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif orientation == 8:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return image


def preprocess_raw(raw_file: str, exif, resoulution=None) -> np.ndarray:
    raw = rawpy.imread(raw_file)
    raw_data = raw.raw_image_visible.astype(np.float32)
    black_level = raw.black_level_per_channel[0]
    white_level = raw.white_level
    normalised_img = (raw_data - black_level) / (white_level - black_level)
    normalised_img = np.clip(normalised_img, 0, 1)
    demosaiced_image = simple_demosaic(normalised_img, exif)
    if resoulution is not None:
        demosaiced_image = cv2.resize(demosaiced_image,
                                      resoulution,
                                      interpolation=cv2.INTER_CUBIC)
    demosaiced_image = fix_orientation(demosaiced_image, exif['Orientation'])
    demosaiced_image = np.clip(demosaiced_image, 0, 1)
    return demosaiced_image.astype(np.float32)


def valid_pixels(img: np.ndarray, th=0.005) -> np.ndarray:
    if np.max(img) > 1 or np.min(img) < 0:
        raise ValueError(
            "Input image must be normalized into (0, 1), but found {:.3f}".
            format(np.max(img)))
    if th > 1 or th < 0:
        raise ValueError(
            "Threshold must be between 0 and 1. Your input is {}".format(th))

    min_th = th
    max_th = 1 - th
    mask = (img > min_th) & (img < max_th)
    mask = np.all(mask, axis=-1)

    return mask


def compute_local_std(img: np.ndarray, kernel_size=17) -> np.ndarray:
    mean = cv2.blur(img, (kernel_size, kernel_size),
                    borderType=cv2.BORDER_REPLICATE)
    sq_mean = cv2.blur(img**2, (kernel_size, kernel_size),
                       borderType=cv2.BORDER_REPLICATE)
    tmp = sq_mean - mean**2
    tmp[tmp < 0] = 0
    std_dev = np.sqrt(tmp)
    return std_dev


def write_image(img_path, img, down_scale_factor=1):
    """
    This function saves the input image as 8bit png according to the input path.
    Note that the color channels muat be sorted as RGB order.

    Args:
        img_path: pathlib.Path
            File path where you want to save the image.
        img: numpy.ndarray
            Image you want to save. Must be sorted as RGB order.
    -------
    Raises:
        TypeError: When your input path is not the pathlib.Path object.
        ValueError: When your input array includes values which is smaller than 0 or larger than 255.
    """

    if np.min(img) < 0 or np.max(img) > MAX_8BIT:
        raise ValueError("Your input array's range doesn't match to 8bit.")
    else:
        img = img.astype(np.uint8)
        h, w = img.shape[:2]
        h, w = h // down_scale_factor, w // down_scale_factor
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(img_path, img)


def rgb_to_srgb(img):
    """
    Convert linear RGB images to sRGB images.
    Args:
        img: ndarray
            The linear RGB image whose sRGB you want.
    Returns: ndarray
    """
    if np.max(img) > 1:
        raise ValueError("Input image must be normalized into (0, 1).")

    a = 0.055

    high_mask = (img > 0.0031308)

    low_c = 12.92 * img
    high_c = (1 + a) * np.power(img, 1.0 / 2.4) - a

    low_c[high_mask] = high_c[high_mask]

    return high_c


def _add_text_to_image(image: np.ndarray, text: str):
    ret = image.copy()
    h, w, c = image.shape
    ret = Image.fromarray(ret)
    font_size = int(h * 0.2)
    stroke_width = font_size // 5
    font = ImageFont.truetype("segoeui.ttf", font_size)
    draw = ImageDraw.Draw(ret)
    position = (int(w * 0.02), h - (font_size + stroke_width) * 1.25)

    bbox = draw.textbbox(position,
                         text,
                         font=font,
                         stroke_width=stroke_width,
                         align="center")
    draw.rectangle(bbox, fill="white")
    draw.text(position, text, font=font, fill="black")
    ret = np.array(ret)
    return ret


def write_white_balanced_image(img: np.ndarray,
                               illum: np.ndarray,
                               output_dir: str,
                               output_name: str,
                               gamma=False,
                               text: str = None,
                               **kwargs):
    os.makedirs(output_dir, exist_ok=True)
    illum /= (illum[1] + 1e-9)
    out_img = img.copy()
    out_img[..., 0] /= illum[0]
    out_img[..., 2] /= illum[2]
    out_img = np.clip(out_img, 0, 1)
    if gamma:
        out_img = out_img**(1. / 2.2)
    out_img = np.clip(out_img, 0, 1)

    save_name = os.path.join(output_dir, output_name)
    out_img *= MAX_8BIT
    out_img = out_img.astype(np.uint8)
    if text is not None:
        out_img = _add_text_to_image(out_img, text)
    write_image(save_name, out_img)


def get_uv_coord(hist_size, device='cpu', dtype=torch.float32, range=1.0):
    """ Gets uv-coordinate extra channels to augment each histogram as
    mentioned in the paper.

  Args:
    hist_size: histogram dimension (scalar).
    tensor: boolean flag for input torch tensor; default is true.
    normalize: boolean flag to normalize each coordinate channel; default
      is false.
    device: output tensor allocation ('cuda' or 'cpu'); default is 'cuda'.

  Returns:
    u_coord: extra channel of the u coordinate values; if tensor arg is True,
      the returned tensor will be in (1 x height x width) format; otherwise,
      it will be in (height x width) format.
    v_coord: extra channel of the v coordinate values. The format is the same
      as for u_coord.
  """

    u_coord, v_coord = torch.meshgrid(torch.arange(-(hist_size - 1) / 2,
                                                   ((hist_size - 1) / 2) + 1),
                                      torch.arange(-(hist_size - 1) / 2,
                                                   ((hist_size - 1) / 2) + 1),
                                      indexing='ij')  # uv could be negative
    scale = range / (hist_size - 1)
    u_coord.requires_grad = False
    v_coord.requires_grad = False
    u_coord = u_coord * scale
    v_coord = v_coord * scale
    u_coord = u_coord.to(device=device, dtype=dtype)
    v_coord = v_coord.to(device=device, dtype=dtype)
    return u_coord, v_coord


def get_chroma_coord(hist_size, device='cpu', dtype=torch.float32, range=1.):
    """ Gets rb-coordinate extra channels to augment each histogram as
    mentioned in the paper.

  Args:
    hist_size: histogram dimension (scalar).
    tensor: boolean flag for input torch tensor; default is true.
    normalize: boolean flag to normalize each coordinate channel; default
      is false.
    device: output tensor allocation ('cuda' or 'cpu'); default is 'cuda'.

  Returns:
    r_coord: extra channel of the u coordinate values; if tensor arg is True,
      the returned tensor will be in (1 x height x width) format; otherwise,
      it will be in (height x width) format.
    b_coord: extra channel of the v coordinate values. The format is the same
      as for r_coord.
  """

    r_coord, b_coord = torch.meshgrid(torch.arange(0, hist_size),
                                      torch.arange(0, hist_size),
                                      indexing='ij')
    r_coord.requires_grad = False
    b_coord.requires_grad = False
    scale = range / (hist_size - 1)
    r_coord = r_coord * scale
    b_coord = b_coord * scale
    r_coord = r_coord.to(device=device, dtype=dtype)
    b_coord = b_coord.to(device=device, dtype=dtype)
    return r_coord, b_coord


def log_uv_to_rgb_torch(uv: torch.Tensor, channel_first=False):
    """ Converts log-chroma space to RGB.

    Args:
        uv: input color(s) in chroma log-chroma space.
        channel_first: boolean flag for input tensor format; default is false.

    Returns:
        color(s) in rgb space.
    """

    rb = torch.exp(-uv)
    if channel_first:
        r = rb[0]
        b = rb[1]
        rgb = torch.stack(
            [r, torch.ones_like(r, dtype=uv.dtype, device=uv.device), b],
            dim=0)
    else:
        r = rb[..., 0]
        b = rb[..., 1]
        rgb = torch.stack(
            [r, torch.ones_like(r, dtype=uv.dtype, device=uv.device), b],
            dim=-1)
    return rgb


def rgb_to_log_uv_torch(rgb: torch.Tensor, channel_first=False):
    """ Converts RGB to log-chroma space.

        Args:
            rgb: input color(s) in rgb space.
            channel_first: boolean flag for input tensor format; default is false.

        Returns:
            color(s) in chroma log-chroma space.
        """

    log_rgb = torch.log(rgb + 1e-9)
    if channel_first:
        u = log_rgb[1] - log_rgb[0]
        v = log_rgb[1] - log_rgb[2]
        uv = torch.stack([u, v], dim=0)
    else:
        u = log_rgb[..., 1] - log_rgb[..., 0]
        v = log_rgb[..., 1] - log_rgb[..., 2]
        uv = torch.stack([u, v], dim=-1)
    return uv


def rgb_to_chroma_torch(rgb: torch.Tensor, channel_first=False):
    """ Converts RGB to chroma space.

    Args:
        rgb: input color(s) in rgb space.
        channel_first: boolean flag for input tensor format; default is false.

    Returns:
        color(s) in chroma space.
    """

    if channel_first:
        rgb = rgb / (rgb.sum(dim=0, keepdim=True) + 1e-9)
        rb = rgb[[0, -1]]
    else:
        rgb = rgb / (rgb.sum(dim=-1, keepdim=True) + 1e-9)
        rb = rgb[..., [0, -1]]
    return rb


def chroma_to_rgb_torch(chroma: torch.Tensor, channel_first=False):
    """ Converts chroma to RGB.

    Args:
        chroma: input color(s) in chroma space.
        channel_first: boolean flag for input tensor format; default is false.

    Returns:
        color(s) in rgb space.
    """

    if channel_first:
        r = chroma[0]
        b = chroma[1]
        g = (1 - r - b)
        r = r / (g + 1e-9)
        b = b / (g + 1e-9)
        rgb = torch.stack([
            r,
            torch.ones_like(r, dtype=chroma.dtype, device=chroma.device), b
        ],
                          dim=0)
    else:
        r = chroma[..., 0]
        b = chroma[..., 1]
        g = (1 - r - b)
        r = r / (g + 1e-9)
        b = b / (g + 1e-9)
        rgb = torch.stack([
            r,
            torch.ones_like(r, dtype=chroma.dtype, device=chroma.device), b
        ],
                          dim=-1)
    return rgb


def compute_uv_histogram_torch(img: torch.Tensor,
                               bin_num=256,
                               boundary_value=2.0,
                               channel_first=False,
                               srgb=False):
    """
    Computes the uv histogram of the input image.

    Args:
        img: input image(s) in rgb space. Eithers in (height x width x 3) or
          (3 x height x width) format.
        bin_num: number of bins for the histogram.
        boundary_value: boundary value for the uv space.
        channel_first: boolean flag for input tensor format; default is false.

    Returns:
        uv histogram.
    """
    if channel_first:
        valid_mask = torch.all(img > 1e-9, dim=0)
        valid_img = img[:, valid_mask]
    else:
        valid_mask = torch.all(img > 1e-9, dim=-1)
        valid_img = img[valid_mask]
    valid_uv = rgb_to_log_uv_torch(valid_img, channel_first=channel_first)

    if channel_first:
        valid_mask = torch.all(valid_uv >= -boundary_value, dim=0) & torch.all(
            valid_uv <= boundary_value, dim=0)
        valid_uv = valid_uv[:, valid_mask]
        valid_img = valid_img[:, valid_mask]
        if srgb:
            illuminance = 0.2126 * valid_img[0] + 0.7152 * valid_img[
                1] + 0.0722 * valid_img[2]
        else:
            illuminance = torch.norm(valid_img, dim=0, p=2)
        valid_uv = valid_uv.permute(1, 0)
    else:
        valid_mask = torch.all(valid_uv >= -boundary_value,
                               dim=-1) & torch.all(valid_uv <= boundary_value,
                                                   dim=-1)
        valid_uv = valid_uv[valid_mask]
        valid_img = valid_img[valid_mask]
        if srgb:
            illuminance = 0.2126 * valid_img[..., 0] + 0.7152 * valid_img[
                ..., 1] + 0.0722 * valid_img[..., 2]
        else:
            illuminance = torch.norm(valid_img, dim=-1, p=2)

    hist, bins = torch.histogramdd(valid_uv,
                                   bins=[bin_num, bin_num],
                                   range=[-boundary_value, boundary_value] * 2,
                                   weight=illuminance)
    hist /= (hist.sum() + 1e-9)
    hist = torch.sqrt(hist)
    return hist


def compute_chroma_histogram_torch(img: torch.Tensor,
                                   bin_num=256,
                                   channel_first=False):
    """

    """
    if channel_first:
        valid_mask = torch.all(img > 1e-9, dim=0)
        valid_img = img[:, valid_mask]
    else:
        valid_mask = torch.all(img > 1e-9, dim=-1)
        valid_img = img[valid_mask]
    valid_chroma = rgb_to_chroma_torch(valid_img, channel_first=channel_first)

    if channel_first:
        illuminance = torch.norm(valid_img, dim=0, p=2)
        valid_chroma = valid_chroma.permute(1, 0)
    else:
        illuminance = torch.norm(valid_img, dim=-1, p=2)

    hist, bins = torch.histogramdd(valid_chroma,
                                   bins=[bin_num, bin_num],
                                   range=[0, 1] * 2,
                                   weight=illuminance)
    hist /= (hist.sum() + 1e-9)
    hist = torch.sqrt(hist)
    return hist


def extract_statistical_features(img: torch.Tensor,
                                 channel_first=False,
                                 thresh_dark=0.02,
                                 thresh_saturation=0.98,
                                 space="rb",
                                 flatten=True):
    if not channel_first:
        img = img.permute(2, 0, 1)
    c, h, w = img.shape
    img = img.reshape(c, -1)
    mask = torch.all(img > thresh_dark, dim=0) & torch.all(
        img < thresh_saturation, dim=0)
    if mask.sum() != 0:
        img = img[:, mask]
    # 2. Average pixel
    mean_v = img.mean(dim=-1)
    if mask.sum() == 0:
        feature_data: torch.Tensor = torch.vstack(
            [mean_v, mean_v, mean_v, mean_v])
    else:
        # 0. Brightest pixel
        brightest_index = torch.argmax(img.sum(dim=0))
        # brightest_index = torch.argmax(img[1])
        bright_v = img[..., brightest_index]
        # 1. Maximum pixel
        max_wp, _ = img.max(dim=-1)
        # 3. Darkest pixel
        darkest_index = torch.argmin(img.sum(dim=0))
        # darkest_index = torch.argmin(img[1])
        dark_v = img[..., darkest_index]
        feature_data: torch.Tensor = torch.vstack(
            [bright_v, max_wp, mean_v, dark_v])
    if space == 'rb':
        feature_data /= (feature_data.sum(axis=-1, keepdim=True) + 1e-9)
        feature_data = feature_data[:, [0, -1]]
    elif space == 'uv':
        feature_data = rgb_to_log_uv_torch(feature_data, channel_first=False)
    if flatten:
        feature_data = feature_data.flatten()
    return feature_data
