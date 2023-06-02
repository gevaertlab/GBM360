"""
General utility functions
"""

import os
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, SequentialSampler, DataLoader
from stqdm import stqdm
import seaborn as sns

from resnet import resnet50
from pathology_models import AggregationModel, Identity, TanhAttention
from get_patch_img import extract_patches
from dataset import PatchDataset
from pathlib import Path

def get_class():
    class2idx = {
    'Normal': 0,
    'NPC' : 1,
    'OPC' : 2,
    'AC' : 3,
    'MESlike': 4,
    'MEShypoxia':5
    }
    id2class = {v: k for k, v in class2idx.items()}  
    return class2idx, id2class

def get_color_ids():
    color_ids = {
    'Normal': 1, 
    'NPC' : 0,
    'OPC' : 3,
    'AC' : 2,
    'MESlike': 5,
    'MEShypoxia': 4
    }
    clusters_colors = {}
    for k, v in color_ids.items():
        clusters_colors[k] = sns.color_palette()[v]
    return color_ids, clusters_colors

def check_device(use_GPU):
    device = 'cpu'
    if use_GPU:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
    return device

def load_model(checkpoint: str, config=None):
    resnet = resnet50(pretrained=False)
    aggregator = None
    if config['aggregator']== 'identity':
        aggregator = Identity()
    elif config['aggregator'] == "attention":
        aggregator = TanhAttention(dim=2048)
    model = AggregationModel(resnet=resnet, aggregator=aggregator, 
                             aggregator_dim=config['aggregator_hdim'],resnet_dim=2048, 
                             out_features=config['num_classes'])
    
    model.load_state_dict(torch.load(checkpoint, map_location=torch.device('cpu')))
    
    return model


def predict_cell(model, val_dataloader, device='cpu'):
    
    model.to(torch.device(device))
    ## Validation
    model.eval()
    results = {
        'coordinates': [],
        'label': []
    }
    
    for batch_dict in stqdm(val_dataloader):
        inputs = batch_dict['image'].to(device)
        coordinates = batch_dict['coordinates']
        # forward
        with torch.no_grad():
            outputs, _ = model.forward(inputs)
        
        outputs = outputs.detach().cpu().numpy()

        tumor_arr = outputs[:, :6]
        class_weights = [1.0, 0.6, 1.4 , 0.5, 1.4, 1.8]
        tumor_arr = tumor_arr * class_weights

        pred_list = np.argmax(tumor_arr, axis=1)
        coordinates_list = [(x, y) for x, y in zip(coordinates[0], coordinates[1])]
        results['coordinates'].append(coordinates_list)
        results['label'].append(pred_list)

    return results

def predict_survival(model, val_dataloader, device='cpu'):

    model.to(torch.device(device))
    ## Validation

    model.eval()

    results = {
        'coordinates': [],
        'risk_score': []
    }

    for batch_dict in stqdm(val_dataloader):
        inputs = batch_dict['image'].to(device)
        coordinates = batch_dict['coordinates']
        # forward
        with torch.no_grad():
            outputs, _ = model.forward(inputs)

        output_list = outputs.detach().cpu().numpy()
        output_list = np.concatenate(output_list, axis=0)
        coordinates_list = [(x.item(), y.item()) for x, y in zip(coordinates[0], coordinates[1])]
        results['coordinates'].append(coordinates_list)
        results['risk_score'].append(output_list)
    
    return results


def read_patches(slide, max_patches_per_slide = np.inf):
    patches, coordinates = extract_patches(slide, patch_size=(112,112), max_patches_per_slide=max_patches_per_slide)
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

def save_uploaded_file(uploaded_file):
    with open(os.path.join("temp",uploaded_file.name),"wb") as f:
         f.write(uploaded_file.getbuffer())
    return os.path.join("temp",uploaded_file.name)

def clear(file):
    os.remove(file)


def style_table(df):
    # style
    th_props = [
    ('font-size', '18pt'),
    ('text-align', 'center'),
    ('font-weight', 'bold'),
    ('color', '#6d6d6d'),
    ('background-color', '#f7ffff')
    ]
                                
    td_props = [
    ('font-size', '18pt')
    ]
                                    
    styles = [
    dict(selector="th", props=th_props),
    dict(selector="td", props=td_props)
    ]

    df_style = df.style.set_properties(**{
    'font-size': '16pt',
    'text-align': 'center',
    'format': '{:.3f}'
    }).set_table_styles(styles)

    return df_style

def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()





