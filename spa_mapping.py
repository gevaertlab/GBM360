# Script to generate the visulization of cell types in whole slide images

import os
import argparse
import re
from typing import Tuple
import random
import pandas as pd
import numpy as np
from openslide import OpenSlide
import json
from multiprocessing import Pool, Value, Lock
import seaborn as sns
from stqdm import stqdm
import matplotlib.pyplot as plt

def assig_to_heatmap(heatmap, patch, x, y, ratio_patch_x, ratio_patch_y,xmax, ymax):
    new_x = int(x / ratio_patch_x)
    new_y = int(y / ratio_patch_y)
    
    try:
        if new_x+patch.shape[0] > heatmap.shape[0] and new_y+patch.shape[1] < heatmap.shape[1]:
            dif = new_x+patch.shape[0] - xmax
            dif = patch.shape[0] - dif
            heatmap[new_x:, new_y:new_y+patch.shape[1], :] = patch[:dif, :, :]
        elif new_x+patch.shape[0] < heatmap.shape[0] and new_y+patch.shape[1] > heatmap.shape[1]:
            dif = new_y+patch.shape[1] - ymax
            dif = patch.shape[1] - dif
            heatmap[new_x:new_x+patch.shape[0], new_y:, :] = patch[:, :dif, :]
        elif new_x+patch.shape[0] > heatmap.shape[0] and new_y+patch.shape[1] > heatmap.shape[1]:
            return heatmap
        else:
            heatmap[new_x:new_x+patch.shape[0], new_y:new_y+patch.shape[1], :] = patch
    
        return heatmap
    except:
        return heatmap

# def assig_to_heatmap(heatmap, patch, x, y):
#     heatmap[x:x+patch.shape[0], y:y+patch.shape[1], :] = patch
#     return heatmap

def get_indices(slide : OpenSlide, patch_size: Tuple, PATCH_LEVEL = 0, dezoom_factor=1):
    
    xmax, ymax = slide.level_dimensions[PATCH_LEVEL]

    # handle slides with 40 magnification at base level
    resize_factor = float(slide.properties.get('aperio.AppMag', 20)) / 20.0
    resize_factor = resize_factor * dezoom_factor
    patch_size_resized = (int(resize_factor * patch_size[0]), int(resize_factor * patch_size[1]))

    indices = [(x, y) for x in range(0, xmax, patch_size_resized[0]) 
            for y in range(0, ymax, patch_size_resized[0])]

    return(indices, xmax, ymax, patch_size_resized)

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
    labels = labels[config['label_column']]
    keys = np.concatenate((keys), axis=0)
    labels = np.concatenate((labels), axis=0)
    # convert predicted labels to actual cell types
    class2idx = {
        'Normal' : 0,
        'NPC-like' : 1,
        'OPC-like' : 2,
        'Immune' : 3,
        'Hypoxia': 4,
        'MES': 5
    }
    id2class = {v: k for k, v in class2idx.items()}   
    cell_types = [id2class[k] for k in labels]
    # Match cell types to colors
    color_ids = {
        'Normal' : 1,
        'NPC-like' : 0,
        'OPC-like' : 3,
        'Immune' : 2,
        'Hypoxia': 4,
        'MES': 5
    }
    colors = [color_ids[k] for k in cell_types]

    color_labels = dict()
    for key, value in zip(keys, colors):
        color_labels[tuple(key)] = value
    return color_labels

def compute_percentage(labels, config):
    labels = labels[config['label_column']]
    labels = np.concatenate((labels), axis=0)
    # convert predicted labels to actual cell types
    class2idx = {
        'Normal' : 0,
        'NPC-like' : 1,
        'OPC-like' : 2,
        'Immune' : 3,
        'Hypoxia': 4,
        'MES': 5
    }
    id2class = {v: k for k, v in class2idx.items()}   
    cell_types = [id2class[k] for k in labels]
    total = len(cell_types)
    percentages = {
        'Normal' : 0.0,
        'NPC-like' : 0.0,
        'OPC-like' : 0.0,
        'Immune' : 0.0,
        'Hypoxia': 0.0,
        'MES': 0
    }
    for cell_type in percentages.keys():
        count = cell_types.count(cell_type)
        percentages[cell_type] = float(count/total) * 100
    
    return percentages

def generate_heatmap(slide, patch_size: Tuple, labels, config):
    PATCH_LEVEL = 0
    
    indices, xmax, ymax, patch_size_resized = get_indices(slide, patch_size, PATCH_LEVEL)

    compress_factor = config['compress_factor']

    heatmap = np.zeros((xmax // compress_factor, ymax // compress_factor, 3))
    histo = np.zeros((xmax // compress_factor, ymax // compress_factor, 3))

    labels_dict = make_dict(labels, config)
    percentages = compute_percentage(labels, config)
    for x, y in stqdm(indices):
        try:
            patch = np.transpose(np.array(slide.read_region((x, y), PATCH_LEVEL, patch_size_resized).convert('RGB')), axes=[1, 0, 2])
            if (x, y) in labels_dict:
                score = labels_dict[(x,y)]
                color = sns.color_palette()[score]
                visualization = np.empty((patch_size[0], patch_size[1], 3), np.uint8)
                visualization[:] = color[0] * 255, color[1] * 255, color[2] * 255
                heatmap = assig_to_heatmap(heatmap, visualization, x, y, compress_factor, compress_factor, xmax, ymax)              
            else:
                heatmap = assig_to_heatmap(heatmap, patch, x, y, compress_factor, compress_factor, xmax, ymax)                           
        except Exception as e:
            print(e)
    
    # since the x and y coordiante is flipped after converting the patch to RGB, we flipped the image again to match the origianl image
    heatmap = np.transpose(heatmap, axes=[1, 0, 2]).astype(np.uint8)

    return heatmap, percentages
