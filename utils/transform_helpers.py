import torch
import math
from enum import Enum
import torch.nn.functional as pt_f
import numpy as np


def get_shift_matrix(shift_vals: tuple) -> np.ndarray:
    """Helper function for AffineNP"""
    matrix = np.identity(3)
    matrix[0:2, 2] = shift_vals[1], shift_vals[0]
    return matrix


def get_rot_matrix(rot_vals: tuple) -> np.ndarray:
    """Helper function for AffineNP"""
    matrix = np.identity(3)
    rot = np.radians(rot_vals[2])
    translation_matrix_1 = get_shift_matrix((-rot_vals[0], -rot_vals[1]))
    translation_matrix_2 = get_shift_matrix(rot_vals[:2])
    matrix[0:2, 0:2] = [[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]]
    matrix = translation_matrix_2 @ matrix @ translation_matrix_1
    return matrix


def get_shear_matrix(shear_vals: tuple) -> np.ndarray:
    """Helper function for AffineNP"""
    translation_matrix_1 = get_shift_matrix((-shear_vals[0], -shear_vals[1]))
    translation_matrix_2 = get_shift_matrix(shear_vals[:2])
    matrix = np.identity(3)
    matrix[1, 0] = shear_vals[2]
    matrix[0, 1] = shear_vals[3]
    matrix = translation_matrix_2 @ matrix @ translation_matrix_1
    return matrix


def rect_from_mask(mask: np.ndarray, dims: list, scale: int):
    """Helper function for AffineNP.
    From mask (0 invalid, 1 valid) and given dims find largest rectangle ratio ver_dim:hor_dim inscribed in mask"""
    # https://gis.stackexchange.com/questions/59215/how-to-find-the-maximum-area-rectangle-inside-a-convex-polygon
    # Above says: if rectangle not square, three points will be on boundary. This means that if we start at one point,
    #   find the next two corner points, follow to the natural 4th and the line hits the boundary so that the resulting
    #   actual rectangle would be smaller, this means that adjusting to get to a better rectangle would mean two
    #   vertices are inside the boundary --> not maximal anyway.
    #   There can be exceptions for this if the line to the corner points from the first points is on a boundary, but in
    #   that case we would still detect the actual maximal rectangle later anyway, when we systematically go through all
    #   other boundary points available as starting points.
    mask = np.round(mask).astype('i')  # Mask cleanup
    # noinspection PyBroadException
    try:
        v1_r, v2_r, h1_r, h2_r, f_r = _fit_rect_single_side(mask, dims, scale)  # Starting from left
        h1_c, h2_c, v1_c, v2_c, f_c = _fit_rect_single_side(np.transpose(mask), dims[::-1], scale)  # Starting from top
        if f_r > f_c:
            h1, h2, v1, v2 = h1_r, h2_r, v1_r, v2_r
        else:
            h1, h2, v1, v2 = h1_c, h2_c, v1_c, v2_c
        return h1, h2, v1, v2
    except Exception:
        # Fallback in case the mask hasn't loaded properly,
        # or an error occurs in _fit_rect_single_side (see screenshot 2020_07_02 in OneDrive docs)
        print("                     Warning: rect_from_mask fallback used")
        return 0, mask.shape[1] - 1, 0, mask.shape[0] - 1


def _fit_rect_single_side(mask: np.ndarray, dims: list, scale: int):
    """Helper function for rect_from_mask.
    Drawings see 29/05/20, notebook 2"""

    assert scale % 2 == 0
    mask = mask[::scale, ::scale]
    dims = [dims[0]//scale, dims[1]//scale]

    r_sum = np.sum(mask, axis=1)

    # Step 1: get pt1 / pt2 (left-most / right-most leading points), and tops / bots (upper / lower boundary pixels)
    pt1 = np.stack((np.arange(mask.shape[0]), (mask != 0).argmax(axis=1)), axis=-1)[r_sum > 0]
    pt2 = np.stack((np.arange(mask.shape[0]), (mask[:, ::-1] != 0).argmax(axis=1)), axis=-1)[r_sum > 0]
    pt2[:, 1] = mask.shape[1] - pt2[:, 1] - 1  # correct for flipping
    tops = np.stack(((mask != 0).argmax(axis=0), np.arange(mask.shape[1])), axis=-1)
    bots = np.stack(((mask[::-1, :] != 0).argmax(axis=0), np.arange(mask.shape[1])), axis=-1)
    bots[:, 0] = mask.shape[0] - bots[:, 0] - 1  # correct for flipping

    # Step 2: construct array [num_pt * num_pt * 2=(ver, hor)], for each point comb the vertical / horizontal distance
    dists = (np.abs(np.expand_dims(pt1, axis=1) - np.expand_dims(pt2, axis=0))).astype('f')

    # Step 3: construct 2 arrays [num_pt * 2(ver up, ver down)], for each point how far up / down there is 'space'
    space1 = np.stack((pt1[:, 0] - tops[pt1[:, 1], 0], bots[pt1[:, 1], 0] - pt1[:, 0]), axis=-1)
    space2 = np.stack((pt2[:, 0] - tops[pt2[:, 1], 0], bots[pt2[:, 1], 0] - pt2[:, 0]), axis=-1)

    # Step 4: if dist vertically 0, set all vertical dists to max of the min of space either side of the two points
    max_min_space = np.max(np.minimum(space1, space2), axis=-1)
    dists[np.identity(len(max_min_space)) > 0, 0] = max_min_space

    # Step 5: set all dists to 0 where the pt1/pt2 combination does not work
    pt1_lims = np.stack((tops[pt1[:, 1], 0], bots[pt1[:, 1], 0]), axis=-1)
    pt2_lims = np.stack((tops[pt2[:, 1], 0], bots[pt2[:, 1], 0]), axis=-1)
    pt1_within = np.greater_equal(pt1[:, 0, np.newaxis], pt2_lims[np.newaxis, :, 0]) &\
        np.less_equal(pt1[:, 0, np.newaxis], pt2_lims[np.newaxis, :, 1])
    pt2_within = np.greater_equal(pt2[:, 0, np.newaxis], pt1_lims[np.newaxis, :, 0]) & \
        np.less_equal(pt2[:, 0, np.newaxis], pt1_lims[np.newaxis, :, 1])
    within = pt1_within & np.transpose(pt2_within)
    dists[~within] = 0

    # Step 6: convert dists into fractions of the input dims, find dir-pair-wise minimum, and overall maximum
    fractions = np.copy(dists) + 1
    fractions[..., 0] /= dims[0]
    fractions[..., 1] /= dims[1]
    min_dir_fraction = np.min(fractions, axis=-1)
    max_fraction = np.max(min_dir_fraction)
    idx = np.unravel_index(min_dir_fraction.argmax(), min_dir_fraction.shape[:2])
    dom_dir = np.argmin(dists[idx])
    p1, p2 = pt1[idx[0]], pt2[idx[1]]
    # NOTE: following line should work, but... when it doesn't, it just throws an error and what's the point.
    # assert mask[p1[0], p2[1]] > 0 and mask[p2[0], p1[1]] > 0

    # Step 7: determine v1, h1, v2, h2
    v1, h1 = p1
    h2 = p2[1]
    v2 = None
    if p1[0] == p2[0]:  # two points in one line
        v2_sel = [v1 - dists[idx[0], idx[1], 0], v1 + dists[idx[0], idx[1], 0]]
        for v in v2_sel:
            v = int(v)
            if 0 <= v < mask.shape[0]:
                if mask[v, h1] > 0 and mask[v, h2] > 0:
                    v2 = v
    else:
        v2 = p2[0]
    [v1, v2] = np.sort([v1, v2])

    # Step 8: determine actual borders with correct ratio
    if dom_dir == 0:  # vertical direction correct: v1, v2, h1 unchanged, h2 adjusted
        v_dist = v2 - v1 + 1
        h_dist = np.floor(v_dist * dims[1] / dims[0]).astype('i')
        h2 = h1 + h_dist - 1
    else:  # horizontal direction correct: h1, h2, v1 unchanged, v2 adjusted
        h_dist = h2 - h1 + 1
        v_dist = np.floor(h_dist * dims[0] / dims[1]).astype('i')
        v2 = v1 + v_dist - 1

    v1, v2, h1, h2 = scale*v1, scale*v2, scale*h1, scale*h2

    return v1, v2, h1, h2, max_fraction


class Lambda:
    """Apply a user-defined lambda as a transform. This transform does not support torchscript.

    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        if not callable(lambd):
            raise TypeError("Argument lambd should be callable, got {}".format(repr(type(lambd).__name__)))
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class InterpolationMode(Enum):
    """Interpolation modes
    """
    NEAREST = "nearest"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    # For PIL compatibility
    BOX = "box"
    HAMMING = "hamming"
    LANCZOS = "lanczos"


def rotate(img: torch.Tensor,
           matrix: list,
           interpolation: str = "nearest",
           expand: bool = False,
           fill: list = None) -> torch.Tensor:
    _assert_grid_transform_inputs(img, matrix, interpolation, fill, ["nearest", "bilinear"])
    w, h = img.shape[-1], img.shape[-2]
    ow, oh = _compute_output_size(matrix, w, h) if expand else (w, h)
    dtype = img.dtype if torch.is_floating_point(img) else torch.float32
    theta = torch.tensor(matrix, dtype=dtype, device=img.device).reshape(1, 2, 3)
    # grid will be generated on the same device as theta and img
    grid = _gen_affine_grid(theta, w=w, h=h, ow=ow, oh=oh)

    return _apply_grid_transform(img, grid, interpolation, fill=fill)


def get_inverse_affine_matrix(center: list, angle: float, translate: list, scale: float, shear: list) -> list:
    # Helper method to compute inverse matrix for affine transformation

    # As it is explained in PIL.Image.rotate
    # We need compute INVERSE of affine transformation matrix: M = T * C * RSS * C^-1
    # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
    #       C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
    #       RSS is rotation with scale and shear matrix
    #       RSS(a, s, (sx, sy)) =
    #       = R(a) * S(s) * SHy(sy) * SHx(sx)
    #       = [ s*cos(a - sy)/cos(sy), s*(-cos(a - sy)*tan(x)/cos(y) - sin(a)), 0 ]
    #         [ s*sin(a + sy)/cos(sy), s*(-sin(a - sy)*tan(x)/cos(y) + cos(a)), 0 ]
    #         [ 0                    , 0                                      , 1 ]
    #
    # where R is a rotation matrix, S is a scaling matrix, and SHx and SHy are the shears:
    # SHx(s) = [1, -tan(s)] and SHy(s) = [1      , 0]
    #          [0, 1      ]              [-tan(s), 1]
    #
    # Thus, the inverse is M^-1 = C * RSS^-1 * C^-1 * T^-1

    rot = math.radians(angle)
    sx, sy = [math.radians(s) for s in shear]

    cx, cy = center
    tx, ty = translate

    # RSS without scaling
    a = math.cos(rot - sy) / math.cos(sy)
    b = -math.cos(rot - sy) * math.tan(sx) / math.cos(sy) - math.sin(rot)
    c = math.sin(rot - sy) / math.cos(sy)
    d = -math.sin(rot - sy) * math.tan(sx) / math.cos(sy) + math.cos(rot)

    # Inverted rotation matrix with scale and shear
    # det([[a, b], [c, d]]) == 1, since det(rotation) = 1 and det(shear) = 1
    matrix = [d, -b, 0.0, -c, a, 0.0]
    matrix = [x / scale for x in matrix]

    # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
    matrix[2] += matrix[0] * (-cx - tx) + matrix[1] * (-cy - ty)
    matrix[5] += matrix[3] * (-cx - tx) + matrix[4] * (-cy - ty)

    # Apply center translation: C * RSS^-1 * C^-1 * T^-1
    matrix[2] += cx
    matrix[5] += cy

    return matrix


def _assert_grid_transform_inputs(img: torch.Tensor,
                                  matrix: list,
                                  interpolation: str,
                                  fill: list,
                                  supported_interpolation_modes: list,
                                  coeffs: list = None):

    if not (isinstance(img, torch.Tensor)):
        raise TypeError("Input img should be Tensor")

    if matrix is not None and not isinstance(matrix, list):
        raise TypeError("Argument matrix should be a list")

    if matrix is not None and len(matrix) != 6:
        raise ValueError("Argument matrix should have 6 float values")

    if coeffs is not None and len(coeffs) != 8:
        raise ValueError("Argument coeffs should have 8 float values")

    if fill is not None and not isinstance(fill, (int, float, tuple, list)):
        raise ValueError("Argument fill should be either int, float, tuple or list")

    # Check fill
    num_channels = _get_image_num_channels(img)
    if isinstance(fill, (tuple, list)) and (len(fill) > 1 and len(fill) != num_channels):
        msg = ("The number of elements in 'fill' cannot broadcast to match the number of "
               "channels of the image ({} != {})")
        raise ValueError(msg.format(len(fill), num_channels))

    if interpolation not in supported_interpolation_modes:
        raise ValueError("Interpolation mode '{}' is unsupported with Tensor input".format(interpolation))


def _compute_output_size(matrix: list, w: int, h: int) -> tuple:
    # Inspired of PIL implementation:
    # https://github.com/python-pillow/Pillow/blob/11de3318867e4398057373ee9f12dcb33db7335c/src/PIL/Image.py#L2054

    # pts are Top-Left, Top-Right, Bottom-Left, Bottom-Right points.
    pts = torch.tensor([
        [-0.5 * w, -0.5 * h, 1.0],
        [-0.5 * w, 0.5 * h, 1.0],
        [0.5 * w, 0.5 * h, 1.0],
        [0.5 * w, -0.5 * h, 1.0],
    ])
    theta = torch.tensor(matrix, dtype=torch.float).reshape(1, 2, 3)
    new_pts = pts.view(1, 4, 3).bmm(theta.transpose(1, 2)).view(4, 2)
    min_vals, _ = new_pts.min(dim=0)
    max_vals, _ = new_pts.max(dim=0)

    # Truncate precision to 1e-4 to avoid ceil of Xe-15 to 1.0
    tol = 1e-4
    cmax = torch.ceil((max_vals / tol).trunc_() * tol)
    cmin = torch.floor((min_vals / tol).trunc_() * tol)
    size = cmax - cmin
    return int(size[0]), int(size[1])


def _gen_affine_grid(theta: torch.Tensor, w: int, h: int, ow: int, oh: int) -> torch.Tensor:
    # https://github.com/pytorch/pytorch/blob/74b65c32be68b15dc7c9e8bb62459efbfbde33d8/aten/src/ATen/native/
    # AffineGridGenerator.cpp#L18
    # Difference with AffineGridGenerator is that:
    # 1) we normalize grid values after applying theta
    # 2) we can normalize by other image size, such that it covers "extend" option like in PIL.Image.rotate

    d = 0.5
    base_grid = torch.empty(1, oh, ow, 3, dtype=theta.dtype, device=theta.device)
    x_grid = torch.linspace(-ow * 0.5 + d, ow * 0.5 + d - 1, steps=ow, device=theta.device)
    base_grid[..., 0].copy_(x_grid)
    y_grid = torch.linspace(-oh * 0.5 + d, oh * 0.5 + d - 1, steps=oh, device=theta.device).unsqueeze_(-1)
    base_grid[..., 1].copy_(y_grid)
    base_grid[..., 2].fill_(1)

    rescaled_theta = theta.transpose(1, 2) / torch.tensor([0.5 * w, 0.5 * h], dtype=theta.dtype, device=theta.device)
    output_grid = base_grid.view(1, oh * ow, 3).bmm(rescaled_theta)
    return output_grid.view(1, oh, ow, 2)


def _apply_grid_transform(img: torch.Tensor, grid: torch.Tensor, mode: str, fill: list) -> torch.Tensor:

    img, need_cast, need_squeeze, out_dtype = _cast_squeeze_in(img, [grid.dtype, ])

    if img.shape[0] > 1:
        # Apply same grid to a batch of images
        grid = grid.expand(img.shape[0], grid.shape[1], grid.shape[2], grid.shape[3])

    # Append a dummy mask for customized fill colors, should be faster than grid_sample() twice
    if fill is not None:
        dummy = torch.ones((img.shape[0], 1, img.shape[2], img.shape[3]), dtype=img.dtype, device=img.device)
        img = torch.cat((img, dummy), dim=1)

    img = pt_f.grid_sample(img, grid, mode=mode, padding_mode="zeros")

    # Fill with required color
    if fill is not None:
        mask = img[:, -1:, :, :]  # N * 1 * H * W
        img = img[:, :-1, :, :]  # N * C * H * W
        mask = mask.expand_as(img)
        len_fill = len(fill) if isinstance(fill, (tuple, list)) else 1
        fill_img = torch.tensor(fill, dtype=img.dtype, device=img.device).view(1, len_fill, 1, 1).expand_as(img)
        if mode == 'nearest':
            mask = mask < 0.5
            img[mask] = fill_img[mask]
        else:  # 'bilinear'
            img = img * mask + (1.0 - mask) * fill_img

    img = _cast_squeeze_out(img, need_cast, need_squeeze, out_dtype)
    return img


def gaussian_blur(img: torch.Tensor, kernel_size: list, sigma: list = None) -> torch.Tensor:
    """NOTE: this and all dependent methods taken from Pytorch 1.7 source"""
    if sigma is None:
        sigma = [ksize * 0.15 + 0.35 for ksize in kernel_size]
    dtype = img.dtype if torch.is_floating_point(img) else torch.float32
    kernel = _get_gaussian_kernel2d(kernel_size, sigma, dtype=dtype, device=img.device)
    kernel = kernel.expand(img.shape[-3], 1, kernel.shape[0], kernel.shape[1])

    img, need_cast, need_squeeze, out_dtype = _cast_squeeze_in(img, [kernel.dtype, ])

    # padding = (left, right, top, bottom)
    padding = [kernel_size[0] // 2, kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[1] // 2]
    img = pt_f.pad(img, padding, mode="reflect")
    img = pt_f.conv2d(img, kernel, groups=img.shape[-3])

    img = _cast_squeeze_out(img, need_cast, need_squeeze, out_dtype)
    return img


def _get_gaussian_kernel2d(kernel_size: list, sigma: list, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    kernel1d_x = _get_gaussian_kernel1d(kernel_size[0], sigma[0]).to(device, dtype=dtype)
    kernel1d_y = _get_gaussian_kernel1d(kernel_size[1], sigma[1]).to(device, dtype=dtype)
    kernel2d = torch.mm(kernel1d_y[:, None], kernel1d_x[None, :])
    return kernel2d


def _get_gaussian_kernel1d(kernel_size: int, sigma: float) -> torch.Tensor:
    ksize_half = (kernel_size - 1) * 0.5

    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    kernel1d = pdf / pdf.sum()

    return kernel1d


def _cast_squeeze_in(img: torch.Tensor, req_dtypes: list) -> tuple:
    need_squeeze = False
    # make image NCHW
    if len(img.shape) < 4:
        img = img.unsqueeze(dim=0)
        need_squeeze = True

    out_dtype = img.dtype
    need_cast = False
    if out_dtype not in req_dtypes:
        need_cast = True
        req_dtype = req_dtypes[0]
        img = img.to(req_dtype)
    return img, need_cast, need_squeeze, out_dtype


def _cast_squeeze_out(img: torch.Tensor, need_cast: bool, need_squeeze: bool, out_dtype: torch.dtype):
    if need_squeeze:
        img = img.squeeze(dim=0)

    if need_cast:
        if out_dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            # it is better to round before cast
            img = torch.round(img)
        img = img.to(out_dtype)

    return img


def adjust_brightness(img: torch.Tensor, brightness_factor: float) -> torch.Tensor:
    if brightness_factor < 0:
        raise ValueError('brightness_factor ({}) is not non-negative.'.format(brightness_factor))

    _assert_channels(img, [1, 3])

    return _blend(img, torch.zeros_like(img), brightness_factor)


def adjust_contrast(img: torch.Tensor, contrast_factor: float) -> torch.Tensor:
    if contrast_factor < 0:
        raise ValueError('contrast_factor ({}) is not non-negative.'.format(contrast_factor))

    _assert_channels(img, [3])

    dtype = img.dtype if torch.is_floating_point(img) else torch.float32
    mean = torch.mean(rgb_to_grayscale(img).to(dtype), dim=(-3, -2, -1), keepdim=True)

    return _blend(img, mean, contrast_factor)


def adjust_hue(img: torch.Tensor, hue_factor: float) -> torch.Tensor:
    if not (-0.5 <= hue_factor <= 0.5):
        raise ValueError('hue_factor ({}) is not in [-0.5, 0.5].'.format(hue_factor))

    if not (isinstance(img, torch.Tensor)):
        raise TypeError('Input img should be Tensor image')

    _assert_channels(img, [1, 3])
    if _get_image_num_channels(img) == 1:  # Match PIL behaviour
        return img

    orig_dtype = img.dtype
    if img.dtype == torch.uint8:
        img = img.to(dtype=torch.float32) / 255.0

    img = _rgb2hsv(img)
    h, s, v = img.unbind(dim=-3)
    h = (h + hue_factor) % 1.0
    img = torch.stack((h, s, v), dim=-3)
    img_hue_adj = _hsv2rgb(img)

    if orig_dtype == torch.uint8:
        img_hue_adj = (img_hue_adj * 255.0).to(dtype=orig_dtype)

    return img_hue_adj


def adjust_saturation(img: torch.Tensor, saturation_factor: float) -> torch.Tensor:
    if saturation_factor < 0:
        raise ValueError('saturation_factor ({}) is not non-negative.'.format(saturation_factor))

    _assert_channels(img, [3])

    return _blend(img, rgb_to_grayscale(img), saturation_factor)


def adjust_gamma(img: torch.Tensor, gamma: float, gain: float = 1) -> torch.Tensor:
    if not isinstance(img, torch.Tensor):
        raise TypeError('Input img should be a Tensor.')

    _assert_channels(img, [1, 3])

    if gamma < 0:
        raise ValueError('Gamma should be a non-negative real number')

    result = img
    dtype = img.dtype
    if not torch.is_floating_point(img):
        result = convert_image_dtype(result, torch.float32)

    result = (gain * result ** gamma).clamp(0, 1)

    result = convert_image_dtype(result, dtype)
    result = result.to(dtype)
    return result


def rgb_to_grayscale(img: torch.Tensor, num_output_channels: int = 1) -> torch.Tensor:
    if len(img.shape) < 3:
        raise TypeError("Input image tensor should have at least 3 dimensions, but found {}".format(len(img.shape)))
    _assert_channels(img, [3])

    if num_output_channels not in (1, 3):
        raise ValueError('num_output_channels should be either 1 or 3')

    r, g, b = img.unbind(dim=-3)
    # This implementation closely follows the TF one:
    # https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/ops/image_ops_impl.py#L2105-L2138
    l_img = (0.2989 * r + 0.587 * g + 0.114 * b).to(img.dtype)
    l_img = l_img.unsqueeze(dim=-3)

    if num_output_channels == 3:
        return l_img.expand(img.shape)

    return l_img


def _assert_channels(img: torch.Tensor, permitted: list) -> None:
    c = _get_image_num_channels(img)
    if c not in permitted:
        raise TypeError("Input image tensor permitted channel values are {}, but found {}".format(permitted, c))


def _get_image_num_channels(img: torch.Tensor) -> int:
    if len(img.shape) == 2:
        return 1
    elif len(img.shape) > 2:
        return img.shape[-3]

    raise TypeError("Input ndim should be 2 or more. Got {}".format(len(img.shape)))


def convert_image_dtype(image: torch.Tensor, dtype: torch.dtype = torch.float) -> torch.Tensor:
    if image.dtype == dtype:
        return image

    if image.is_floating_point():
        if torch.tensor(0, dtype=dtype).is_floating_point():
            return image.to(dtype)

        # float to int
        if (image.dtype == torch.float32 and dtype in (torch.int32, torch.int64)) or (
            image.dtype == torch.float64 and dtype == torch.int64
        ):
            msg = f"The cast from {image.dtype} to {dtype} cannot be performed safely."
            raise RuntimeError(msg)

        # https://github.com/pytorch/vision/pull/2078#issuecomment-612045321
        # For data in the range 0-1, (float * 255).to(uint) is only 255
        # when float is exactly 1.0.
        # `max + 1 - epsilon` provides more evenly distributed mapping of
        # ranges of floats to ints.
        eps = 1e-3
        max_val = _max_value(dtype)
        result = image.mul(max_val + 1.0 - eps)
        return result.to(dtype)
    else:
        input_max = _max_value(image.dtype)

        # int to float
        if torch.tensor(0, dtype=dtype).is_floating_point():
            image = image.to(dtype)
            return image / input_max

        output_max = _max_value(dtype)

        # int to int
        if input_max > output_max:
            # factor should be forced to int for torch jit script
            # otherwise factor is a float and image // factor can produce different results
            factor = int((input_max + 1) // (output_max + 1))
            image = image // factor
            return image.to(dtype)
        else:
            # factor should be forced to int for torch jit script
            # otherwise factor is a float and image * factor can produce different results
            factor = int((output_max + 1) // (input_max + 1))
            image = image.to(dtype)
            return image * factor


def _max_value(dtype: torch.dtype) -> float:
    # https://github.com/pytorch/pytorch/issues/41492

    a = torch.tensor(2, dtype=dtype)
    signed = 1 if torch.tensor(0, dtype=dtype).is_signed() else 0
    bits = 1
    max_value = torch.tensor(-signed, dtype=torch.long)
    while True:
        next_value = a.pow(bits - signed).sub(1)
        if next_value > max_value:
            max_value = next_value
            bits *= 2
        else:
            break
    return max_value.item()


def _blend(img1: torch.Tensor, img2: torch.Tensor, ratio: float) -> torch.Tensor:
    ratio = float(ratio)
    bound = 1.0 if img1.is_floating_point() else 255.0
    return (ratio * img1 + (1.0 - ratio) * img2).clamp(0, bound).to(img1.dtype)


# noinspection PyUnresolvedReferences
def _rgb2hsv(img):
    r, g, b = img.unbind(dim=-3)

    # Implementation is based on https://github.com/python-pillow/Pillow/blob/4174d4267616897df3746d315d5a2d0f82c656ee/
    # src/libImaging/Convert.c#L330
    maxc = torch.max(img, dim=-3).values
    minc = torch.min(img, dim=-3).values

    # The algorithm erases S and H channel where `maxc = minc`. This avoids NaN
    # from happening in the results, because
    #   + S channel has division by `maxc`, which is zero only if `maxc = minc`
    #   + H channel has division by `(maxc - minc)`.
    #
    # Instead of overwriting NaN afterwards, we just prevent it from occuring so
    # we don't need to deal with it in case we save the NaN in a buffer in
    # backprop, if it is ever supported, but it doesn't hurt to do so.
    eqc = maxc == minc

    cr = maxc - minc
    # Since `eqc => cr = 0`, replacing denominator with 1 when `eqc` is fine.
    ones = torch.ones_like(maxc)
    s = cr / torch.where(eqc, ones, maxc)
    # Note that `eqc => maxc = minc = r = g = b`. So the following calculation
    # of `h` would reduce to `bc - gc + 2 + rc - bc + 4 + rc - bc = 6` so it
    # would not matter what values `rc`, `gc`, and `bc` have here, and thus
    # replacing denominator with 1 when `eqc` is fine.
    cr_divisor = torch.where(eqc, ones, cr)
    rc = (maxc - r) / cr_divisor
    gc = (maxc - g) / cr_divisor
    bc = (maxc - b) / cr_divisor

    hr = (maxc == r).to(torch.float) * (bc - gc)
    hg = ((maxc == g) & (maxc != r)).to(torch.float) * (2.0 + rc - bc)
    hb = ((maxc != g) & (maxc != r)).to(torch.float) * (4.0 + gc - rc)
    h = (hr + hg + hb)
    h = torch.fmod((h / 6.0 + 1.0), 1.0)
    return torch.stack((h, s, maxc), dim=-3)


# noinspection PyUnresolvedReferences
def _hsv2rgb(img):
    h, s, v = img.unbind(dim=-3)
    i = torch.floor(h * 6.0)
    f = (h * 6.0) - i
    i = i.to(dtype=torch.int32)

    p = torch.clamp((v * (1.0 - s)), 0.0, 1.0)
    q = torch.clamp((v * (1.0 - s * f)), 0.0, 1.0)
    t = torch.clamp((v * (1.0 - s * (1.0 - f))), 0.0, 1.0)
    i = i % 6

    mask = i.unsqueeze(dim=-3) == torch.arange(6, device=i.device).view(-1, 1, 1).to(torch.int)

    a1 = torch.stack((v, q, p, p, t, v), dim=-3)
    a2 = torch.stack((t, v, v, q, p, p), dim=-3)
    a3 = torch.stack((p, p, t, v, v, q), dim=-3)
    a4 = torch.stack((a1, a2, a3), dim=-4)

    return torch.einsum("...ijk, ...xijk -> ...xjk", mask.to(dtype=img.dtype), a4)
