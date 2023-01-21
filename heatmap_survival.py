from typing import Tuple
import random
import numpy as np
from openslide import OpenSlide
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from stqdm import stqdm
import matplotlib.pyplot as plt
from PIL import Image
import math


from get_patch_img import get_mask


def get_matrix_shape(slide : OpenSlide, patch_size: Tuple, cohort = "TCGA"):
    
    PATCH_LEVEL = 0
    xmax, ymax = slide.level_dimensions[PATCH_LEVEL]

    # handle slides with 40 magnification at base level
    if cohort == "TCGA":
        resize_factor = float(slide.properties.get('aperio.AppMag', 20)) / 20.0
    else:
        if not slide.properties.get('openslide.mpp-x'): print(f"resolution is not found, using default 0.5um/px")
        resize_factor = 0.5 / float(slide.properties.get('openslide.mpp-x', 0.5))
    patch_size_resized = (int(resize_factor * patch_size[0]), int(resize_factor * patch_size[1]))
    return(patch_size_resized)

def get_matracies(wsi_path, slide_id, cluster_df, patch_size, args):

    if args.cohort == "Stanford":
        slide = OpenSlide(os.path.join(wsi_path, f'{slide_id}.tiff'))
    else:
        slide = OpenSlide(os.path.join(wsi_path, f'{slide_id}.svs'))

    patch_size_resized = get_matrix_shape(slide, patch_size, cohort = args.cohort)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cluster_df['new_x'] = np.ceil(cluster_df['x'].values / patch_size_resized[0])
        cluster_df['new_y'] = np.ceil(cluster_df['y'].values / patch_size_resized[1])

        cluster_df['new_x'] = cluster_df['new_x'].astype(int)
        cluster_df['new_y'] = cluster_df['new_y'].astype(int)

    matrix_trait = pd.DataFrame({args.label: cluster_df[args.label], 'x':  cluster_df['new_x'], 'y': cluster_df['new_y']})

    return(matrix_trait)


def assig_to_heatmap(heatmap, patch, x, y, ratio_patch_x, ratio_patch_y):

    new_x = int(x / ratio_patch_x)
    new_y = int(y / ratio_patch_y)

    try:
        if new_x+patch.shape[0] > heatmap.shape[0] and new_y+patch.shape[1] < heatmap.shape[1]:
            dif =  heatmap.shape[0] - new_x
            heatmap[new_x:heatmap.shape[0], new_y:new_y+patch.shape[1], :] = patch[:dif, :, :]
        elif new_x+patch.shape[0] < heatmap.shape[0] and new_y+patch.shape[1] > heatmap.shape[1]:
            dif = heatmap.shape[1] - new_y
            heatmap[new_x:new_x+patch.shape[0], new_y:, :] = patch[:, :dif, :]
        elif new_x+patch.shape[0] > heatmap.shape[0] and new_y+patch.shape[1] > heatmap.shape[1]:
            return heatmap
        else:
            heatmap[new_x:new_x+patch.shape[0], new_y:new_y+patch.shape[1], :] = patch
        return heatmap
    except:
        return heatmap

def get_indices(slide : OpenSlide, patch_size: Tuple, PATCH_LEVEL = 0, dezoom_factor = 1, use_h5 = False):
    
    xmax, ymax = slide.level_dimensions[PATCH_LEVEL]

    # handle slides with 40 magnification at base level
    if use_h5:
        resize_factor = 0.5 / float(slide.properties.get('openslide.mpp-x', 0.5))
    else:
        resize_factor = float(slide.properties.get('aperio.AppMag', 20)) / 20.0
    
    resize_factor = resize_factor * dezoom_factor
    patch_size_resized = (int(resize_factor * patch_size[0]), int(resize_factor * patch_size[1]))

    indices = [(x, y) for x in range(0, xmax, patch_size_resized[0]) 
            for y in range(0, ymax, patch_size_resized[0])]

    return(indices, xmax, ymax, patch_size_resized, resize_factor)

def get_color_linear(minimum, maximum, value):
    # give the minimun and maxium value, generate a color mapped to blue-red heatmap
    minimum, maximum = float(minimum), float(maximum)
    ratio = 2 * (value-minimum) / (maximum - minimum)
    b = int(max(0, 255*(1 - ratio)))
    r = int(max(0, 255*(ratio - 1)))
    g = 255 - b - r
    return r, g, b

def make_dict(labels):
    keys = labels['coordinates']
    values = labels['risk_score']
    survival_labels = dict()
    for key, value in zip(keys, values):
        for k, v in zip(key, value):
            survival_labels[k] = v
    return survival_labels

def generate_heatpmap_survival(slide, patch_size: Tuple, results: dict, min_val=-2, max_val=2.34, config = None):

    PATCH_LEVEL = 0
    indices, xmax, ymax, patch_size_resized, resize_factor = get_indices(slide, patch_size, PATCH_LEVEL, use_h5 = config['use_h5'])

    compress_factor = config['compress_factor'] * round(resize_factor)

    heatmap = np.zeros((xmax // compress_factor, ymax // compress_factor, 3))

    labels_dict = make_dict(results)

    for x, y in indices:
        try:
            patch = np.transpose(np.array(slide.read_region((x, y), PATCH_LEVEL, patch_size_resized).convert('RGB')), axes=[1, 0, 2])
            patch = Image.fromarray(patch)
            patch = patch.resize((math.ceil(patch_size_resized[0] / compress_factor), math.ceil(patch_size_resized[1] / compress_factor)))
            patch = np.asarray(patch) 

            if (x, y) in labels_dict:
                score = labels_dict[(x,y)]
                color = get_color_linear(min_val, max_val, score)
                visualization = np.empty((math.ceil(patch_size_resized[0] / compress_factor), math.ceil(patch_size_resized[1] / compress_factor), 3), np.uint8)
                visualization[:] = color[0] * 255, color[1] * 255, color[2] * 255
                heatmap = assig_to_heatmap(heatmap, visualization, x, y, compress_factor, compress_factor)              
            else:
                heatmap = assig_to_heatmap(heatmap, patch, x, y, compress_factor, compress_factor)                           
            histo = assig_to_heatmap(histo, patch, x, y, compress_factor, compress_factor)   
        except Exception as e:
            print(e)
    
    # since the x and y coordiante is flipped after converting the patch to RGB, we flipped the image again to match the original image
    heatmap = np.transpose(heatmap, axes=[1, 0, 2]).astype(np.uint8)

    return heatmap
