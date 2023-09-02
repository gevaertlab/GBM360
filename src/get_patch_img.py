"""
Utility functions to isolate foreground from background and to extract patches
"""

import numpy as np
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
from skimage.exposure.exposure import is_low_contrast
from skimage.transform import resize
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from stqdm import stqdm
from torchvision import transforms
from torch.utils.data import DataLoader, SequentialSampler, DataLoader
from dataset import PatchDataset

def read_patches(slide, max_patches_per_slide = np.inf, image_type = 'svs'):
    patches, coordinates = extract_patches(slide, patch_size=(112,112), \
                                           max_patches_per_slide=max_patches_per_slide, \
                                           image_type = image_type)

    data_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = PatchDataset(patches, coordinates, data_transforms)
    image_samplers = SequentialSampler(dataset)
    
    # Create training and validation dataloaders
    dataloader = DataLoader(dataset, batch_size=64, sampler=image_samplers)
    return dataloader

def get_mask_image(img_RGB, RGB_min=50, image_type = 'svs'):

    img_HSV = rgb2hsv(img_RGB)

    background_R = img_RGB[:, :, 0] > threshold_otsu(img_RGB[:, :, 0])
    background_G = img_RGB[:, :, 1] > threshold_otsu(img_RGB[:, :, 1])
    background_B = img_RGB[:, :, 2] > threshold_otsu(img_RGB[:, :, 2])
    tissue_RGB = np.logical_not(background_R & background_G & background_B)
    tissue_S = img_HSV[:, :, 1] > threshold_otsu(img_HSV[:, :, 1])
    min_R = img_RGB[:, :, 0] > RGB_min
    min_G = img_RGB[:, :, 1] > RGB_min
    min_B = img_RGB[:, :, 2] > RGB_min

    if image_type == "svs":
        mask = tissue_S & tissue_RGB & min_R & min_G & min_B
    else:
        mask = min_R & min_G & min_B
    return mask

def get_mask(slide, level='max', RGB_min=50, image_type = 'svs'):
    #read svs image at a certain level  and compute the otsu mask
    if level == 'max':
        level = len(slide.level_dimensions) - 1
    # note the shape of img_RGB is the transpose of slide.level_dimensions
    img_RGB = np.transpose(np.array(slide.read_region((0, 0),level,slide.level_dimensions[level]).convert('RGB')),
                           axes=[1, 0, 2])

    tissue_mask = get_mask_image(img_RGB, RGB_min, image_type = image_type)
    return tissue_mask, level

def extract_patches(slide, patch_size, max_patches_per_slide=2000, dezoom_factor=1.0, image_type = 'svs'):
    mask, mask_level = get_mask(slide, image_type = image_type)
    mask = binary_dilation(mask, iterations=3)
    mask = binary_erosion(mask, iterations=3)

    mask_level = len(slide.level_dimensions) - 1
    
    PATCH_LEVEL = 0
    BACKGROUND_THRESHOLD = .2

    ratio_x = slide.level_dimensions[PATCH_LEVEL][0] / slide.level_dimensions[mask_level][0]
    ratio_y = slide.level_dimensions[PATCH_LEVEL][1] / slide.level_dimensions[mask_level][1]

    xmax, ymax = slide.level_dimensions[PATCH_LEVEL]

    # handle slides with 40 magnification at base level
    resize_factor = float(slide.properties.get('aperio.AppMag', 20)) / 20.0
    resize_factor = resize_factor * dezoom_factor
    patch_size_resized = (int(resize_factor * patch_size[0]), int(resize_factor * patch_size[1]))
    i = 0

    indices = [(x, y) for x in range(0, xmax, patch_size_resized[0]) for y in
                range(0, ymax, patch_size_resized[1])]

    patches = []
    coordinates = []

    for x, y in stqdm(indices):
        # check if in background mask
        x_mask = int(x / ratio_x)
        y_mask = int(y / ratio_y)
        if mask[x_mask, y_mask] == 1:
            patch = slide.read_region((x, y), PATCH_LEVEL, patch_size_resized).convert('RGB')
            try:
                mask_patch = get_mask_image(np.array(patch))
                mask_patch = binary_dilation(mask_patch, iterations=3)
            except Exception as e:
                print("error with slide patch {}".format(i))
                print(e)
            if (mask_patch.sum() > BACKGROUND_THRESHOLD * mask_patch.size) and not (is_low_contrast(patch)):
                if resize_factor != 1.0:
                    patch = patch.resize(patch_size)

                coordinates.append((x,y))
                patches.append(patch)
                i += 1
        if i >= max_patches_per_slide:
            break

    return patches, coordinates