import re
import cv2
import numpy as np
import torch
import torch.nn.functional as F

__all__ = [
    "extract_swatches_from_color_checker", "get_white_swatch_loc_index",
    "rearrange_poly_by_white_block_index", "warp_color_checker",
    "apply_2d_transform_matrix", "get_default_color_checker_panel",
    "get_default_swatch_centers"
]


def warp_color_checker(origin_img: np.ndarray,
                       bbox: list,
                       long_side_pixel=1024) -> tuple:
    tl = bbox[0]
    tr = bbox[1]
    br = bbox[2]
    bl = bbox[3]
    corners = (tl, tr, br, bl)

    # Finding the maximum width.
    widthA = np.sqrt(((br[0] - bl[0])**2) + ((br[1] - bl[1])**2))
    widthB = np.sqrt(((tr[0] - tl[0])**2) + ((tr[1] - tl[1])**2))
    maxWidth = max(int(widthA), int(widthB))
    # Finding the maximum height.
    heightA = np.sqrt(((tr[0] - br[0])**2) + ((tr[1] - br[1])**2))
    heightB = np.sqrt(((tl[0] - bl[0])**2) + ((tl[1] - bl[1])**2))
    maxHeight = max(int(heightA), int(heightB))

    if maxWidth > maxHeight:
        standard_width = long_side_pixel
        satandard_height = int(standard_width / 1.5)
    else:
        satandard_height = long_side_pixel
        standard_width = int(satandard_height * 1.5)

    destination_corners = [[0, 0], [standard_width, 0],
                           [standard_width, satandard_height],
                           [0, satandard_height]]

    M = cv2.getPerspectiveTransform(np.float32(corners),
                                    np.float32(destination_corners))
    ret = cv2.warpPerspective(origin_img,
                              M, (standard_width, satandard_height),
                              flags=cv2.INTER_LINEAR)
    return ret, M


def _pad_img(img, pad=1, pad_value=(0, 0, 0)):
    if pad == 0:
        return img
    ret = cv2.copyMakeBorder(img,
                             pad,
                             pad,
                             pad,
                             pad,
                             cv2.BORDER_CONSTANT,
                             value=pad_value)
    return ret


def _compute_point_dist(point1, point2):
    if point1 is None or point2 is None:
        return float('inf')
    dist = np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    return dist


def _get_contour_center(a_contour: np.ndarray):
    a_contour = a_contour.reshape(4, 2)
    return np.mean(a_contour, axis=0).astype(int)


def _get_four_corners_of_a_contour(src_contour, max_approx_time=50):
    for max_error in range(2, max_approx_time + 1):
        dst_contour = cv2.approxPolyDP(src_contour, max_error, closed=True)
        if len(dst_contour) == 4:
            break

        dst_contour = cv2.approxPolyDP(dst_contour, max_error, closed=True)
        if len(dst_contour) == 4:
            break
    return dst_contour if len(dst_contour) == 4 else None


def _get_point_loc_index(center, img_shape):
    x, y = img_shape[1], img_shape[0]
    tl = [0, 0]
    tr = [x, 0]
    br = [x, y]
    bl = [0, y]

    mini_index = 0
    mini_dist = float('inf')
    for i, corner_point in enumerate((tl, tr, br, bl)):
        dist = _compute_point_dist(center, corner_point)
        if dist < mini_dist:
            mini_dist = dist
            mini_index = i
    return mini_index


def _convert_img_by_ae(img: np.ndarray):
    """
    Convert a RGB image to one channel image by Angular error
    """
    img = cv2.GaussianBlur(img, (31, 31), 0)
    img = img.astype(np.float32)
    img /= img.max()
    img = torch.from_numpy(img)
    ref = img.reshape(-1, 3).mean(dim=0)
    ref = torch.ones_like(img, dtype=img.dtype) * ref
    cos = F.cosine_similarity(img, ref, 2, 1e-6)
    cos = torch.clamp(cos, -0.9999, 0.9999)
    ret = torch.acos(cos) * 180 / torch.pi
    ret = (ret - ret.min()) / (ret.max() - ret.min()) * 255
    return ret.numpy().astype(np.uint8)


def _compute_angular_distance(vec1: np.ndarray, vec2: np.ndarray):
    vec1 = torch.from_numpy(vec1).float()
    vec2 = torch.from_numpy(vec2).float()
    cos = F.cosine_similarity(vec1, vec2, dim=0)
    cos = torch.clamp(cos, -0.9999, 0.9999)
    return torch.acos(cos) * 180 / torch.pi


def _get_binary_threhold_img(img, block_size):
    macbeth_split = cv2.split(img)
    macbeth_split_thresh = []
    for channel in macbeth_split:
        res = cv2.adaptiveThreshold(channel,
                                    255,
                                    cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY_INV,
                                    block_size,
                                    C=21)
        macbeth_split_thresh.append(res)
    return np.bitwise_or(*macbeth_split_thresh)


def _close_image_edges(img, block_size, base=3):
    element_size = int(base + block_size / 10)
    shape, ksize = cv2.MORPH_RECT, (element_size, element_size)
    element = cv2.getStructuringElement(shape, ksize)
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, element)


def _is_seq_hole(c):
    return cv2.contourArea(c, oriented=True) > 0


def _is_standard_shape(contour, min_size, max_size):
    _, (w, h), _ = cv2.minAreaRect(contour)
    if min(w, h) == 0:
        return False
    contour_area = w * h
    ratio = max(w, h) / min(w, h)
    return min_size <= contour_area <= max_size and ratio < 3


def _is_quad(c):
    c = c.reshape(-1, 2)
    edge_1 = _compute_point_dist(c[0], c[1])
    edge_2 = _compute_point_dist(c[1], c[2])
    edge_3 = _compute_point_dist(c[2], c[3])
    edge_4 = _compute_point_dist(c[3], c[0])
    tmp = [edge_1, edge_2, edge_3, edge_4]
    for i in range(3):
        for j in range(i + 1, 4):
            ratio = min(tmp[i], tmp[j]) / max(tmp[i], tmp[j])
            if ratio < 0.5:
                return False
    return cv2.isContourConvex(c)


def _get_swatch_contours_by_rgb(img, pad=1):
    '''
    Compute 4 corners of each swatch in the given image
    '''
    block_size = int(min(img.shape[:2]) * 0.05) | 1

    adaptive = _get_binary_threhold_img(img, block_size)
    adaptive = _close_image_edges(adaptive, block_size)
    adaptive = _pad_img(adaptive, pad=pad, pad_value=255)

    tmp = cv2.findContours(image=adaptive,
                           mode=cv2.RETR_LIST,
                           method=cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = tmp

    img_area = np.prod(img.shape[:2])
    min_size = img_area * 0.01
    max_size = img_area * 0.05

    contours = [
        c for c in contours
        if _is_standard_shape(c, min_size, max_size) and _is_seq_hole(c)
    ]
    contours = [_get_four_corners_of_a_contour(x) for x in contours]
    contours = [x for x in contours if x is not None and _is_quad(x)]
    return contours


def _get_swatch_contours_by_ae(img, pad=1):
    '''
    Compute 4 corners of each swatch in the given image
    '''
    img_new = _convert_img_by_ae(img)
    block_size = int(min(img.shape[:2]) * 0.2) | 1

    adaptive = cv2.adaptiveThreshold(img_new,
                                     255,
                                     cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY_INV,
                                     block_size,
                                     C=21)

    element_size = int(25 + block_size / 10)
    shape, ksize = cv2.MORPH_RECT, (element_size, element_size)
    element = cv2.getStructuringElement(shape, ksize)
    adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_DILATE, element)

    element_size = int(17 + block_size / 10)
    shape, ksize = cv2.MORPH_RECT, (element_size, element_size)
    element = cv2.getStructuringElement(shape, ksize)
    adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_ERODE, element)

    adaptive = _pad_img(adaptive, pad, pad_value=255)
    contours, _ = cv2.findContours(image=adaptive,
                                   mode=cv2.RETR_LIST,
                                   method=cv2.CHAIN_APPROX_SIMPLE)

    img_area = np.product(img.shape[:2])
    min_size = img_area * 0.015
    max_size = img_area * 0.05

    contours = [
        c for c in contours
        if _is_standard_shape(c, min_size, max_size) and _is_seq_hole(c)
    ]
    contours = [_get_four_corners_of_a_contour(x) for x in contours]
    contours = [x for x in contours if x is not None and _is_quad(x)]
    return contours


def _find_white_block_index(centers, img: np.ndarray):
    ret = 0
    max_gray = float('-inf')
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ref = img.reshape(-1, 3).mean(axis=0)

    candidates = sorted(
        zip(centers, range(len(centers))),
        key=lambda x: _compute_angular_distance(ref, img[x[0][1]][x[0][0]]))
    candidates = candidates[:5]

    for center, i in candidates:
        value = gray_img[center[1], center[0]]
        if value > max_gray:
            max_gray = value
            ret = i
    return ret


def _find_white_block_center(img, pad=1):
    detected_contours = _get_swatch_contours_by_rgb(img, pad=pad)
    if detected_contours:
        centers = [_get_contour_center(x) - pad for x in detected_contours]
        index = _find_white_block_index(centers, img)
        if index is not None:
            return centers[index]
    return None


def get_default_color_checker_panel(swatch_size=100,
                                    horizontal_count=6,
                                    vertical_count=4):
    '''
    Create a defaul color checker mask and its corners"
    '''
    ret = [[None] * horizontal_count for _ in range(vertical_count)]
    offset = swatch_size / 2
    margin = swatch_size / 7 * 2
    for i in range(vertical_count):
        for j in range(horizontal_count):
            x = offset + (swatch_size + margin) * j
            y = offset + (swatch_size + margin) * i
            ret[i][j] = [x, y]
    gap = offset / 2
    four_corners = [[gap, gap], [ret[0][-1][0] + gap, gap],
                    [ret[-1][-1][0] + gap, ret[-1][-1][1] + gap],
                    [gap, ret[-1][0][1] + gap]]
    four_corners = np.array(four_corners, np.float32)

    return np.array(ret, np.float32), four_corners


def get_default_swatch_centers(
    color_checker_height=682,
    color_checker_width=1024,
    horizontal_count=6,
    vertical_count=4,
):
    '''
    Assume the color checker is well wapred to a standard shape
    '''
    ret = [[None] * horizontal_count for _ in range(vertical_count)]
    swatch_height = color_checker_height / 34 * 7
    swatch_width = color_checker_width / 52 * 7

    offset_x = swatch_width / 2
    offset_y = swatch_height / 2
    margin_x = swatch_width / 7 * 2
    margin_y = swatch_height / 7 * 2

    for i in range(vertical_count):
        for j in range(horizontal_count):
            x = offset_x + (swatch_width + margin_x) * j
            y = offset_y + (swatch_height + margin_y) * i
            ret[i][j] = [x, y]
    four_corners = [[0, 0], [ret[0][-1][0] + offset_x, 0],
                    [ret[-1][-1][0] + offset_x, ret[-1][-1][1] + offset_y],
                    [0, ret[-1][-1][1] + offset_y]]
    four_corners = np.array(four_corners, np.float32).reshape(-1, 2)
    ret = np.array(ret, np.float32).reshape(-1, 2)
    return ret, four_corners


def _apply_2d_transform_matrix(points: np.ndarray,
                               transform_matrix: np.ndarray):
    points = points.reshape(-1, 2)
    addition = np.ones((points.shape[0], 1), np.float32)
    points = np.c_[points, addition]
    ret = np.dot(transform_matrix, points.T)
    ret = ret.T
    return ret[:, :2] / ret[:, -1:]


def _find_match_chart(centers, warped_color_checker):
    h, w = warped_color_checker.shape[:2]
    c = np.array([w // 2, h // 2])
    max_radius = max(h, w) / 4
    rad = []
    for i in range(len(centers)):
        dis_c = centers - centers[i]
        dis = np.sqrt(dis_c[:, 0]**2 + dis_c[:, 1]**2)
        dis.sort()
        min_dist = dis[1] if len(dis) > 1 else float('inf')
        if min_dist < max_radius:
            rad.append(min_dist)
    radius = np.mean(rad)
    index_xy = np.zeros((len(centers), 2))
    for i in range(len(centers)):
        temp = (centers[i] - c) / radius + 0.5
        index_xy[i, 0] = temp[0]
        index_xy[i, 1] = temp[1] - 1
    index = np.round(index_xy) + 2
    index = index[:, 0] + index[:, 1] * 6
    return index


def get_swatches_centers(img, min_points_threshold=6, pad=1):
    contours = _get_swatch_contours_by_ae(img, pad=pad)
    centers = [_get_contour_center(x) - pad for x in contours]
    ret = [[None] * 6 for _ in range(4)]
    found_centers = 0
    if len(centers) < min_points_threshold:
        print("Only find {} points, less than the threshold {}.".format(
            len(centers), min_points_threshold))
        return None
    try:
        index = _find_match_chart(centers, img)
        for i in range(len(centers)):
            y = int(index[i] // 6)
            x = int(index[i] % 6)
            if y >= 0 and y < 4:
                ret[y][x] = centers[i]
                found_centers += 1
    except:
        pass
    if found_centers < min_points_threshold:
        return None
    return ret


def get_white_swatch_loc_index(img, pad=1):
    white_swatch_center = _find_white_block_center(img, pad=pad)
    if white_swatch_center is not None:
        return _get_point_loc_index(white_swatch_center, img.shape)
    return None


def rearrange_poly_by_white_block_index(points, white_block_index):
    '''
    Rearrange poly points so that the white block center locates at bottom left
    '''
    rotate_times = 3 - white_block_index
    tl = (points[0], points[1])
    tr = (points[2], points[3])
    br = (points[4], points[5])
    bl = (points[6], points[7])
    corners = [tl, tr, br, bl]
    corners = corners[-rotate_times:] + corners[:-rotate_times]
    polys = [item for sublist in corners for item in sublist]
    return polys


def apply_2d_transform_matrix(points: np.ndarray,
                              transform_matrix: np.ndarray):
    points = points.reshape(-1, 2)
    addition = np.ones((points.shape[0], 1), np.float32)
    points = np.c_[points, addition]
    ret = np.dot(transform_matrix, points.T)
    ret = ret.T
    return ret[:, :2] / ret[:, -1:]


def extract_swatches_from_color_checker(img,
                                        horizontal_count=6,
                                        vertical_count=4):
    '''
    Get swatch centers from an image with color checker
    White swatch should be rotated to the bottom left before use this function
    Return: n * 2  vector, the last 4 representing the four corners
    '''

    matched_centers = get_swatches_centers(img, pad=1)
    if matched_centers is not None:
        src_centers = []
        dst_centers = []
        default_panel, four_corners = get_default_color_checker_panel(
            100, horizontal_count, vertical_count)
        for i in range(vertical_count):
            for j in range(horizontal_count):
                if matched_centers[i][j] is not None:
                    src_centers.append(default_panel[i][j])
                    dst_centers.append(matched_centers[i][j])

        src_centers = np.array(src_centers, np.float32)
        dst_centers = np.array(dst_centers, np.float32)
        transform_matrix, _ = cv2.findHomography(src_centers,
                                                 dst_centers,
                                                 method=cv2.RANSAC,
                                                 ransacReprojThreshold=4)

        swatches_centers = _apply_2d_transform_matrix(
            default_panel, transform_matrix).astype(int)
        four_corner_points = _apply_2d_transform_matrix(
            four_corners, transform_matrix).astype(int)

        ret = np.vstack([swatches_centers, four_corner_points])
        return ret
