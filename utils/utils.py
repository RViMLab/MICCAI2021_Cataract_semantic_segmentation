import cv2
import os
import pathlib
import json
import torch
from torchvision.transforms import ToPILImage, ColorJitter, ToTensor, Normalize, RandomApply
from utils import PadNP, FlipNP, AffineNP, BlurPIL, CropNP, CLASS_INFO, DEFAULT_CONFIG_DICT, DEFAULT_CONFIG_NESTED_DICT
import pandas as pd
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size


def remap_experiment(mask, experiment):
    """Remap mask for Experiment 'experiment' (needs to be int)"""
    colormap = get_remapped_colormap(CLASS_INFO[experiment][0])
    remapped_mask = remap_mask(mask, class_remapping=CLASS_INFO[experiment][0])
    return remapped_mask, CLASS_INFO[experiment][1], colormap


def remap_mask(mask, class_remapping, ignore_label=255, to_network=None):
    """
    Remaps mask class ids
    :param mask: 2D/3D ndarray of input segmentation mask
    :param class_remapping: dictionary that indicates class remapping
    :param ignore_label: class ids to be ignored
    :param to_network: default False. If true, the ignore value (255) is remapped to the correct number for exp 2 or 3
    :return: 2D/3D ndarray of remapped segmentation mask
    """
    to_network = False if to_network is None else to_network
    classes = []
    for key, val in class_remapping.items():
        for cls in val:
            classes.append(cls)
    assert len(classes) == len(set(classes))

    n = max(len(classes), mask.max() + 1)
    remap_array = np.full(n, ignore_label, dtype=np.uint8)
    for key, val in class_remapping.items():
        for v in val:
            remap_array[v] = key
    mask_remapped = remap_array[mask]
    if to_network:
        mask_remapped[mask_remapped == 255] = len(class_remapping) - 1
    return mask_remapped


def get_remapped_colormap(class_remapping):
    """
    Generated colormap of remapped classes
    Classes that are not remapped are indicated by the same color across all experiments
    :param class_remapping: dictionary that indicates class remapping
    :return: 2D ndarray of rgb colors for remapped colormap
    """
    colormap = get_cadis_colormap()
    remapped_colormap = {}
    for key, val in class_remapping.items():
        if key == 255:
            remapped_colormap.update({key: [0, 0, 0]})
        else:
            remapped_colormap.update({key: colormap[val[0]]})
    return remapped_colormap


def get_cadis_colormap():
    """
    Returns cadis colormap as in paper
    :return: ndarray of rgb colors
    """
    return np.asarray(
        [
            [0, 137, 255],
            [255, 165, 0],
            [255, 156, 201],
            [99, 0, 255],
            [255, 0, 0],
            [255, 0, 165],
            [255, 255, 255],
            [141, 141, 141],
            [255, 218, 0],
            [173, 156, 255],
            [73, 73, 73],
            [250, 213, 255],
            [255, 156, 156],
            [99, 255, 0],
            [157, 225, 255],
            [255, 89, 124],
            [173, 255, 156],
            [255, 60, 0],
            [40, 0, 255],
            [170, 124, 0],
            [188, 255, 0],
            [0, 207, 255],
            [0, 255, 207],
            [188, 0, 255],
            [243, 0, 255],
            [0, 203, 108],
            [252, 255, 0],
            [93, 182, 177],
            [0, 81, 203],
            [211, 183, 120],
            [231, 203, 0],
            [0, 124, 255],
            [10, 91, 44],
            [2, 0, 60],
            [0, 144, 2],
            [133, 59, 59],
        ]
    )


def mask_from_network(mask, experiment):
    """
    Converts the segmentation masks as used in the network to using the IDs as used by the CaDISv2 paper
    :param mask: Input mask with classes numbered strictly from 0 to num_classes-1
    :param experiment: Experiment number
    :return: Mask with classes numbered as required by CaDISv2 for the specific experiment (includes '255')
    """
    if experiment == 2 or experiment == 3:
        mask[mask == len(CLASS_INFO[experiment][1]) - 1] = 255
    return mask


def mask_to_colormap(mask, colormap, from_network=None, experiment=None):
    """
    Genarates RGB mask colormap from mask with class ids
    :param mask: 2D/3D ndarray of input segmentation mask
    :param colormap: dictionary that indicates color corresponding to each class
    :param from_network: Default False. If True, class IDs as used in the network are first corrected to CaDISv2 usage
    :param experiment: Needed if from_network = True to determine which IDs need to be corrected
    :return: 3D ndarray Generated RGB mask
    """
    from_network = False if from_network is None else from_network
    if from_network:
        mask = mask_from_network(mask, experiment)
    rgb = np.zeros(mask.shape[:2] + (3,), dtype=np.uint8)
    # TODO: I feel this can be vectorised for speed
    for label, color in colormap.items():
        rgb[mask == label] = color
    return rgb


def plot_images(img, remapped_mask, remapped_colormap, classes_exp):
    """
    Generates plot of Image and RGB mask with class colorbar
    :param img: 3D ndarray of input image
    :param remapped_mask: 2D/3D ndarray of input segmentation mask with class ids
    :param remapped_colormap: dictionary that indicates color corresponding to each class
    :param classes_exp: dictionary of classes names and corresponding class ids
    :return: plot of image and rgb mask with class colorbar
    """
    mask_rgb = mask_to_colormap(remapped_mask, colormap=remapped_colormap)

    fig, axs = plt.subplots(1, 2, figsize=(26, 7))
    plt.subplots_adjust(left=1 / 16.0, right=1 - 1 / 16.0, bottom=1 / 8.0, top=1 - 1 / 8.0)
    axs[0].imshow(img)
    axs[0].axis("off")

    img_u_labels = np.unique(remapped_mask)
    c_map = []
    cl = []
    for i_label in img_u_labels:
        for i_key, i_color in remapped_colormap.items():
            if i_label == i_key:
                c_map.append(i_color)
        for i_key, i_class in classes_exp.items():
            if i_label == i_key:
                cl.append(i_class)
    cl = np.asarray(cl)
    cmp = np.asarray(c_map) / 255
    cmap_mask = LinearSegmentedColormap.from_list("seg_mask_colormap", cmp, N=len(cmp))
    im = axs[1].imshow(mask_rgb, cmap=cmap_mask)
    intervals = np.linspace(0, 255, num=len(cl) + 1)
    ticks = intervals + int((intervals[1] - intervals[0]) / 2)
    divider = make_axes_locatable(axs[1])
    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    cbar1 = fig.colorbar(mappable=im, cax=cax1, ticks=ticks, orientation="vertical")
    cbar1.ax.set_yticklabels(cl)
    axs[1].axis("off")
    fig.tight_layout()

    return fig


def plot_experiment(img_path, mask_path, experiment=1):
    """
    Generates plot of image and rgb mask with colorbar for specified experiment
    :param img_path: Path to input image
    :param mask_path: Path to input segmentation mask
    :param experiment: int Experimental setup (1,2 or 3)
    :return: plot of image and rgb mask with class colorbar
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.COLOR_BGR2GRAY)
    remapped_mask, classes_exp, colormap = remap_experiment(mask, experiment)
    return plot_images(img, remapped_mask, colormap, classes_exp)


def to_comb_image(img, lbl, lbl_pred, experiment):
    with torch.no_grad():
        img, lbl, lbl_pred = to_numpy(img), to_numpy(lbl), to_numpy(lbl_pred)
        img = np.round(np.moveaxis(img, 0, -1) * 255).astype('uint8')
        lbl = mask_to_colormap(lbl, get_remapped_colormap(CLASS_INFO[experiment][0]),
                               from_network=True, experiment=experiment)
        lbl_pred = mask_to_colormap(lbl_pred, get_remapped_colormap(CLASS_INFO[experiment][0]),
                                    from_network=True, experiment=experiment)
        comb_img = np.concatenate((img, lbl, lbl_pred), axis=1)
    return comb_img


def get_matrix_fig(matrix, exp_num):
    labels = [item[1] for item in CLASS_INFO[exp_num][1].items()]
    n = len(labels)
    fig, ax = plt.subplots(figsize=(.7*n, .7*n))
    im, cbar = heatmap(matrix, labels, labels, ax=ax, cbar_kw={}, cmap="YlGn", cbarlabel="Percentage probability")
    annotate_heatmap(im, valfmt='{x:.2f}', threshold=.6)
    fig.tight_layout()
    return fig


def heatmap(data, row_labels, col_labels, ax=None, cbar_kw=None, cbarlabel="", **kwargs):
    """Create a heatmap from a numpy array and two lists of labels. {COPIED FROM MATPLOTLIB DOCS}

    :param data: A 2D numpy array of shape (N, M).
    :param row_labels: A list or array of length N with the labels for the rows.
    :param col_labels: A list or array of length M with the labels for the columns.
    :param ax: A `matplotlib.axes.Axes` instance to which the heatmap is plotted. If not provided, use current axes or
        create a new one. Optional.
    :param cbar_kw: A dictionary with arguments to `matplotlib.Figure.colorbar`. Optional.
    :param cbarlabel: The label for the colorbar. Optional.
    :param kwargs: All other arguments are forwarded to `imshow`.
    :return: im, cbar
    """
    cbar_kw = {} if cbar_kw is None else cbar_kw

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)
    im.set_clim(vmin=0, vmax=1)

    # Create colorbar
    # Code adapted from: https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
    aspect = 20
    pad_fraction = 0.5
    divider = make_axes_locatable(im.axes)
    width = axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_size.Fraction(pad_fraction, width)
    cax = divider.append_axes("right", size=width, pad=pad)
    cbar = ax.figure.colorbar(im, cax=cax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels[:data.shape[0]])
    ax.set_yticklabels(row_labels[:data.shape[0]])

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}", textcolors=None, threshold=None, **textkw):
    """A function to annotate a heatmap. {COPIED FROM MATPLOTLIB DOCS}

    :param im: The AxesImage to be labeled.
    :param data: Data used to annotate.  If None, the image's data is used.  Optional.
    :param valfmt: The format of the annotations inside the heatmap. This should either use the string format method,
        e.g. "$ {x:.2f}", or be a `matplotlib.ticker.Formatter`. Optional.
    :param textcolors: A list or array of two color specifications. The first is used for values below a threshold,
        the second for those above. Optional.
    :param threshold: Value in data units according to which the colors from textcolors are applied. If None (the
        default) uses the middle of the colormap as separation. Optional.
    :param textkw: All other arguments are forwarded to each call to `text` used to create the text labels.
    :return: texts
    """
    textcolors = ["black", "white"] if textcolors is None else textcolors
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(np.max(data)) / 2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        # noinspection PyUnresolvedReferences
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            # noinspection PyCallingNonCallable
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def parse_transform_list(transform_list, transform_values, num_classes):
    """Helper function to parse given dataset transform list. Order of things:
    - first the 'common' transforms are applied. At this point, the input is expected to be a numpy array.
    - then the img and lbl transforms are each applied as necessary. The input is expected to be a numpy array, the
        output will be a tensor, as required by PyTorch"""
    transforms_dict = {
        'train': {
            'common': [],
            'img': [],
            'lbl': [],
        },
        'valid': {
            'common': [],
            'img': [],
            'lbl': [],
        }
    }

    # Step 1: Go through all transforms that need to go into the 'commom' section, i.e. which rely on using the same
    # random parameters on both the image and the label: generally actual augmentation transforms.
    #   Input: np.ndarray; Output: np.ndarray
    if 'flip' in transform_list:
        transforms_dict['train']['common'].append(FlipNP())

    rotation = 0
    rot_centre_offset = (.2, .2)
    shift = 0
    shear = (0, 0)
    shear_centre_offset = (.2, .2)
    set_affine = False
    if 'rot' in transform_list:
        rotation = 15
        set_affine = True
    if 'shift' in transform_list:
        shift = .1
        set_affine = True
    if 'shear' in transform_list:
        shear = (.1, .1)
        set_affine = True
    if 'affine' in transform_list:
        rotation = 10
        shear = (.1, .1)
        rot_centre_offset = (.1, .1)
        set_affine = True
    if set_affine:
        transforms_dict['train']['common'].append(AffineNP(num_classes,
                                                           crop_to_fit=False,
                                                           rotation=rotation,
                                                           rot_centre_offset=rot_centre_offset,
                                                           shift=shift,
                                                           shear=shear,
                                                           shear_centre_offset=shear_centre_offset))

    if 'crop' in transform_list:
        transforms_dict['train']['common'].append(CropNP(size=transform_values['crop_size'],
                                                         crop_mode=transform_values['crop_mode'],
                                                         experiment=transform_values['experiment']))

    # Step 2: Go through all transforms that need to be applied individually afterwards
    #
    # Pad (if necessary) will be the first element of 'img' / 'lbl' transforms.
    #   Input: np.ndarray; Output: np.ndarray
    if 'pad' in transform_list:
        # Needs to be added to img and lbl, train and valid
        if 'crop' not in transform_list:  # Padding only necessary if no cropping has happened
            for obj in ['img', 'lbl']:
                transforms_dict['train'][obj].append(PadNP(ver=(2, 2), hor=(0, 0), padding_mode='reflect'))
        for obj in ['img', 'lbl']:  # Padding for validation always necessary, as never cropped
            transforms_dict['valid'][obj].append(PadNP(ver=(2, 2), hor=(0, 0), padding_mode='reflect'))

    # PIL Image: needed for training images if some of the pytorch transform functions are present
    pil_needed = False
    for t in transform_list:
        if t in ['colorjitter', 'blur']:  # Add other keywords for fcts that need pil.Image input here
            pil_needed = True
    if pil_needed:
        transforms_dict['train']['img'].append(ToPILImage())

    # ColorJitter only applied on training images
    #   Input: pil.Image; Output: pil.Image
    if 'blur' in transform_list:
        transforms_dict['train']['img'].append(BlurPIL(probability=.05, kernel_limits=(3, 7)))

    if 'colorjitter' in transform_list:
        transforms_dict['train']['img'].append(ColorJitter(brightness=(2/3, 1.5), contrast=(2/3, 1.5),
                                                           saturation=(2/3, 1.5), hue=(-.05, .05)))

    s = None
    # for using stronger than default augmentation -- may be removed in the future
    if 'pseudo_colorjitter' in transform_list:
        for e in transform_list:
            if isinstance(e, dict):
                if 'strength' in e:
                    s = e['strength']
                    assert s in [1, 2, 3]
                    break
        if s is None:
            s = 2
        range_extent = (1-s*0.25, 1+s*0.25)
        color_jitter = ColorJitter(brightness=range_extent,
                                   contrast=range_extent,
                                   saturation=range_extent,
                                   hue=(-.02 * s, .02 * s))
        rnd_color_jitter = RandomApply([color_jitter], p=0.7)
        transforms_dict['train']['img'].append(rnd_color_jitter)

    # Tensor: needed by default.
    #   Input: np.array or pil.Image; Output: torch.Tensor
    for stage in ['train', 'valid']:
        for obj in ['img', 'lbl']:
            transforms_dict[stage][obj].append(ToTensor())

    # Normalisation (e.g. for use with the pretrained ResNets from the torchvision model zoo)
    #   Input: torch.Tensor; Output: torch.Tensor
    if 'torchvision_normalise' in transform_list:
        for stage in ['train', 'valid']:
            transforms_dict[stage]['img'].append(Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    return transforms_dict


def un_normalise(arr: torch.Tensor, mean: list = None, std: list = None):
    """Reverts the action of torchvision.transforms.Normalize (on numpy). Assumes NCHW shape"""
    mean = [0.485, 0.456, 0.406] if mean is None else mean
    std = [0.229, 0.224, 0.225] if std is None else std
    mean = torch.as_tensor(mean, device=arr.device).view(-1, 1, 1)
    std = torch.as_tensor(std, device=arr.device).view(-1, 1, 1)
    unnorm_arr = arr * std + mean
    return unnorm_arr


def to_numpy(tensor):
    """Tensor to numpy, calls .cpu() if necessary"""
    with torch.no_grad():
        if tensor.device.type == 'cuda':
            tensor = tensor.cpu()
        return tensor.numpy()


def softmax(x, theta=1.0, axis=None):
    """Compute the softmax of each element along an axis of X.
    From: https://nolanbconaway.github.io/blog/2017/softmax-numpy.html

    :param x: ND-Array. Probably should be floats.
    :param theta: (optional) float parameter, used as a multiplier prior to exponentiation. Default = 1.0
    :param axis: (optional) axis to compute values along. Default is the first non-singleton axis.
    :return: an array the same size as X. The result will sum to 1 along the specified axis.
    """
    # make x at least 2d
    y = np.atleast_2d(x)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if x was 1D
    if len(x.shape) == 1:
        p = p.flatten()

    return p


def parse_config(file_path, user, device):
    # Load config
    try:
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
    except FileNotFoundError:
        print("Configuration file not found at given path '{}'".format(file_path))
        exit(1)
    # Fill in correct paths
    config_path = pathlib.Path(file_path).parent
    with open(config_path / 'path_info.json', 'r') as f:
        path_info = json.load(f)  # Dict: keys are user codes, values are a list of 'data_path', 'log_path' (absolute)
    if user in path_info:
        config_dict.update({
            'data_path': path_info[user][0],
            'log_path': path_info[user][1],
            'ss_pretrained_path': path_info['ss_pretrained_{}'.format(user)][0]
        })
    else:
        ValueError("User '{}' not found in configs/path_info.json".format(user))
    # Fill in GPU device if applicable
    if device >= 0:  # Only update config if user entered a device (default otherwise -1)
        config_dict['gpu_device'] = device

    # Make sure all necessary default values exist
    default_dict = DEFAULT_CONFIG_DICT.copy()
    default_dict.update(config_dict)  # Keeps all default values not overwritten by the passed config
    nested_default_dicts = DEFAULT_CONFIG_NESTED_DICT.copy()
    for k, v in nested_default_dicts.items():  # Go through the nested dicts, set as default first, then update
        default_dict[k] = v  # reset to default values
        default_dict[k].update(config_dict[k])  # Overwrite defaults with the passed config values

    # Extra config bits needed
    default_dict['data']['transform_values']['experiment'] = default_dict['data']['experiment']

    return default_dict


def fig_from_dist(elements: np.ndarray, counts: np.ndarray, desired_num_bins: int,
                  xlabel: str = '', ylabel: str = ''):
    """Returns bar chart figure with frequency count as y axis, bins along x axis"""
    els_per_bin = np.maximum(len(elements) // desired_num_bins, 1)
    num_bins = len(elements) // els_per_bin
    el_ind_lists = np.arange(els_per_bin * num_bins)\
        .reshape((num_bins, els_per_bin)).tolist()
    if len(elements) > els_per_bin * num_bins:
        el_ind_lists.append(np.arange(els_per_bin * num_bins, len(elements)).tolist())
        num_bins += 1  # correction due to extra bin for left-over elements
    chart_count = np.zeros(num_bins, 'i')
    for i, el_ind_list in enumerate(el_ind_lists):
        chart_count[i] = np.sum(counts[el_ind_list])
    fig, ax = plt.subplots(figsize=(.2 * num_bins, 8))
    tick_labels = []
    for i in range(num_bins):
        if len(el_ind_lists[i]) > 1:
            lbl = '{} - {}'.format(el_ind_lists[i][0], el_ind_lists[i][-1])
        else:
            lbl = '{}'.format(el_ind_lists[i][0])
        tick_labels.append(lbl)
    ax.bar(tick_labels, chart_count)
    plt.xticks(rotation=45)
    plt.axis('tight')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig


def get_class_info(training_df: pd.DataFrame, experiment: int, with_name=False):
    # with_name=True will add per frame class information with columns named with their "names" instead of numbrer id
    classes = [c for c in CLASS_INFO[experiment][0].keys() if c != 255]
    if with_name:
        classes = [CLASS_INFO[experiment][1][c] for c in classes]
    for c, c_name in enumerate(classes):  # Leave out the 'ignore' class, if it exists
        col_sum = training_df[[CLASS_INFO[0][1][i] for i in CLASS_INFO[experiment][0][c]]].sum(1)
        if with_name:
            training_df[c_name] = col_sum
        else:
            training_df[c] = col_sum
    return training_df


def reverse_one_to_many_mapping(mapping: dict):
    """ inverts class experiment mappings or id to name dicts """
    reverse_mapping = dict()
    for key in mapping.keys():
        vals = mapping[key]
        if isinstance(vals, list):
            for key_new in vals:
                reverse_mapping[key_new] = key
        elif type(vals) == str:
            reverse_mapping[vals] = key
    return reverse_mapping


def create_new_directory(d):
    """create if it does not exist else do nothing and return -1"""
    _ = os.makedirs(d) if not(os.path.isdir(d)) else -1
    return _


def colourise_data(data: np.ndarray,  # NHW expected
                   low: float = 0, high: float = 1,
                   repeat: list = None,
                   perf_colour: tuple = (255, 0, 0)) -> np.ndarray:
    # perf_colour in RGB
    if high == -1:  # Scale by maximum present
        high = np.max(data)
    data = np.clip((data - low) / (high - low), 0, 1)
    colour_img = np.round(data[..., np.newaxis] *
                          np.array(perf_colour)[np.newaxis, np.newaxis, np.newaxis, :]).astype('uint8')
    if repeat is not None:
        colour_img = np.repeat(np.repeat(colour_img, repeat[0], axis=1), repeat[1], axis=2)
    return colour_img


def worker_init_fn(_):
    np.random.seed(torch.initial_seed() % 2**32)
