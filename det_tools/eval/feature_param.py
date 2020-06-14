import numpy as np
import cv2
from skimage import morphology


def extract_length_width_bydt(mask):
    """extract_length and width by distance transform"""
    height, width = mask.shape[0], mask.shape[1]
    # 使用骨架算法
    skeleton = morphology.skeletonize(mask)
    length = sum(skeleton.flatten())
    if length < min(height, width):
        length = min(height, width)  # 圆形缺陷的skeleton会被提取为一个点
    # distance transform
    dist_img = cv2.distanceTransform(mask.astype('uint8'), cv2.DIST_L2, cv2.DIST_MASK_3)
    width = np.median(dist_img[skeleton]) * 2
    return length, width


def extract_pixel_area(mask):
    """
    extract_pixel_area
    :param mask:
    :return:
    """
    return sum(mask.flatten())


def extract_brightness(mask, image):
    """
    extract_brightness
    :param mask:
    :param image:
    :return:
    """
    try:
        segm_pixels = image[mask == 1].flatten().tolist()
    except Exception as e:
        logging.debug('Mask shape: {}, image_shape: {}'.format(mask.shape, image.shape))
        return 0, 0, 0
    if len(segm_pixels) == 0:
        return 0, 0, 0
    top_k = max(1, int(len(segm_pixels) * 0.2))
    top_k_idx = sorted(segm_pixels, reverse=True)[0:top_k]
    low_k_idx = sorted(segm_pixels)[0:top_k]
    return sum(segm_pixels) / len(segm_pixels), sum(top_k_idx) / len(top_k_idx), sum(
        low_k_idx) / len(low_k_idx)


def extract_gradients(mask, image):
    """
    extract_gradients
    :param mask:
    :param image:
    :return:
    """
    gray_x = cv2.Sobel(image, cv2.CV_32F, 1, 0)  # x方向一阶导数
    gray_y = cv2.Sobel(image, cv2.CV_32F, 0, 1)  # y方向一阶导数
    gradx = cv2.convertScaleAbs(gray_x)  # 转回原来的uint8形式
    grady = cv2.convertScaleAbs(gray_y)
    grad = cv2.addWeighted(gradx, 0.5, grady, 0.5, 0)  # 图像融合
    # 提取mask边缘点的梯度值
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    # 提取边缘点
    edge_points = []
    for contour in contours:
        for i in range(contour.shape[0]):
            edge_point = contour[i, 0, :]
            edge_points.append(edge_point)
    # 计算边缘点梯度均值
    grad_sum = 0
    for ep in edge_points:
        x, y = ep[0], ep[1]
        grad_sum += grad[y, x]
    return grad_sum if len(edge_points) == 0 else grad_sum / len(edge_points)


def extract_contrast(mask, image, up_scale=100):
    """
    extract_contrast
    :param mask:
    :param image:
    :param up_scale:
    :return:
    """
    image_norm = image / 255
    fgs = image[mask != 0].flatten()
    bgs = image[mask == 0].flatten()
    if len(fgs) == 0:
        fg_mean = 0
    else:
        fg_mean = sum(fgs) / len(fgs)
    if len(bgs) == 0:
        bg_mean = 0
    else:
        bg_mean = sum(bgs) / len(bgs)
    contrast = abs(fg_mean - bg_mean)
    return contrast * up_scale
