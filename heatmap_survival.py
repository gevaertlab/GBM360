import os
import argparse
import re
from typing import Tuple
import random
import pandas as pd

import numpy as np
from openslide import OpenSlide
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
from scipy.ndimage.morphology import binary_dilation, binary_erosion
import json
from stqdm import stqdm
from get_patch_img import get_mask


def assig_to_heatmap_old(heatmap, patch, x, y, ratio_patch_x, ratio_patch_y,xmax, ymax):
    new_x = int(x / ratio_patch_x)
    new_y = int(y / ratio_patch_y)
    try:
        if new_x+patch.shape[0] > heatmap.shape[0] and new_y+patch.shape[1] < heatmap.shape[1]:
            #print("case1")
            dif = new_x+patch.shape[0] - xmax
            dif = patch.shape[0] - dif
            heatmap[new_x:, new_y:new_y+patch.shape[1], :] = patch[:dif, :, :]
        elif new_x+patch.shape[0] < heatmap.shape[0] and new_y+patch.shape[1] > heatmap.shape[1]:
            #print("case2")
            dif = new_y+patch.shape[1] - ymax
            dif = patch.shape[1] - dif
            heatmap[new_x:new_x+patch.shape[0], new_y:, :] = patch[:, :dif, :]
        elif new_x+patch.shape[0] > heatmap.shape[0] and new_y+patch.shape[1] > heatmap.shape[1]:
            #print("case3")
            return heatmap
        else:
            #print("case4")
            heatmap[new_x:new_x+patch.shape[0], new_y:new_y+patch.shape[1], :] = patch
        return heatmap
    except:
        return heatmap

def assig_to_heatmap(heatmap, patch, x, y):
    heatmap[x:x+patch.shape[0], y:y+patch.shape[1], :] = patch
    return heatmap

def get_indices(slide : OpenSlide, patch_size: Tuple, dezoom_factor=1.0):
 
    mask, mask_level = get_mask(slide)
    mask = binary_dilation(mask, iterations=3)
    mask = binary_erosion(mask, iterations=3)

    mask_level = len(slide.level_dimensions) - 1
    
    PATCH_LEVEL = 0
    BACKGROUND_THRESHOLD = .2

    ratio_x = slide.level_dimensions[PATCH_LEVEL][0] / slide.level_dimensions[mask_level][0] # 128
    ratio_y = slide.level_dimensions[PATCH_LEVEL][1] / slide.level_dimensions[mask_level][1] # 128

    xmax, ymax = slide.level_dimensions[PATCH_LEVEL]

    # handle slides with 40 magnification at base level
    resize_factor = float(slide.properties.get('aperio.AppMag', 20)) / 20.0
    resize_factor = resize_factor * dezoom_factor
    patch_size_resized = (int(resize_factor * patch_size[0]), int(resize_factor * patch_size[1]))
    i = 0

    indices = [(x, y) for x in range(0, xmax, patch_size_resized[0]) 
            for y in range(0, ymax, patch_size_resized[0])]
    
    return(indices, xmax, ymax, patch_size_resized)

def get_color_class(classes):
    # give a list of labels and assign a unique RGB value to each label
    colors = []
    for i in range(len(classes)):
        hex = '#%06X' % random.randint(0, 0xFFFFFF)
        value = hex.lstrip('#')
        lv = len(value)
        rgb = tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
        bgr = (rgb[0], rgb[1], rgb[2])
        colors.append(rgb)
    areas_class = dict()
    class_predictions = dict()
    for label,color in zip(classes, colors):
        print('{}/{}'.format(label,color))
        areas_class[label] = 0
        class_predictions[label] = 0
    return  areas_class, class_predictions

def get_color_linear(minimum, maximum, value):
    # give the minimun and maxium value, generate a color mapped to blue-red heatmap
    minimum, maximum = float(minimum), float(maximum)
    ratio = 2 * (value-minimum) / (maximum - minimum)
    b = int(max(0, 255*(1 - ratio)))
    r = int(max(0, 255*(ratio - 1)))
    g = 255 - b - r
    return r, g, b

def make_dict(labels, config):
    keys = labels['coordinates']
    values = labels[config['label_column']]
    keys = np.concatenate((keys), axis=0)
    values = np.concatenate((values), axis=0)
    survival_labels = dict()
    for key, value in zip(keys, values):
        survival_labels[tuple(key)] = value
    return survival_labels

def generate_heatpmap_survival(slide, patch_size: Tuple, results: dict, min_val=-2, max_val=2.34, resize_factor=1):
            
    indices, xmax_patch, ymax_patch, patch_size_resized = get_indices(slide, patch_size)
    
    heatmap = np.zeros((xmax_patch, ymax_patch, 3))
    histo = np.zeros((xmax_patch, ymax_patch, 3))

    PATCH_LEVEL = 0

    #pdb.set_trace()
    for x, y in stqdm(indices):
        patch = np.transpose(np.array(slide.read_region((x, y), PATCH_LEVEL, patch_size_resized).convert('RGB')), axes=[1, 0, 2])
        #patch = patch.resize((224,224))
        if (x, y) in list(results['coordinates']):
            score = results[results['coordinates'] == (x,y)]['risk_score'].item()
            color = get_color_linear(min_val, max_val, score)
            visualization = np.empty((patch_size[0],patch_size[1],3), np.uint8)
            visualization[:] = color[0], color[1], color[2]
            heatmap = assig_to_heatmap(heatmap, visualization, x, y)              
        else:
            try:
                heatmap = assig_to_heatmap(heatmap, patch, x, y)                           
            except:
                continue
        histo = assig_to_heatmap(histo, patch, x, y)   
                            
    return heatmap, histo