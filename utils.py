import os
import tqdm

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, SequentialSampler, DataLoader
from stqdm import stqdm

from resnet import resnet50
from pathology_models import AggregationModel, Identity, TanhAttention
from get_patch_img import extract_patches
from dataset import PatchDataset

def load_model(checkpoint: str, cuda=False, config=None):
    resnet = resnet50(pretrained=config['pretrained'])
    aggregator = None
    if config['aggregator']== 'identity':
        aggregator = Identity()
    elif config['aggregator'] == "attention":
        aggregator = TanhAttention(dim=2048)
    model = AggregationModel(resnet=resnet, aggregator=aggregator, 
                             aggregator_dim=config['aggregator_hdim'],resnet_dim=2048, 
                             out_features=config['num_classes'])
    
    model.load_state_dict(torch.load(checkpoint, map_location=torch.device('cpu')))

    if cuda:
        model = model.cuda()
    
    return model

def predict_cell(model, val_dataloader, device='cpu'):
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
        pred_list = np.argmax(outputs, axis=1)
        coordinates_list = [(x, y) for x, y in zip(coordinates[0], coordinates[1])]
        results['coordinates'].append(coordinates_list)
        results['label'].append(pred_list)
    
    return results

def predict_survival(model, val_dataloader, device='cpu'):
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
        coordinates_list = [(x, y) for x, y in zip(coordinates[0], coordinates[1])]
        results['coordinates'].append(coordinates_list)
        results['risk_score'].append(output_list)
    
    return results


def read_patches(slide):
    patches, coordinates = extract_patches(slide, patch_size=(112,112), max_patches_per_slide=100)
    data_transforms = transforms.Compose([
        transforms.Resize(46),
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