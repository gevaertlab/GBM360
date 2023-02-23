from torch.utils.data import Dataset

class PatchDataset(Dataset):
    """
    csv_path must contain csv with header
    case, wsi_file_name, attr1,...,attrk
    """

    def __init__(self, patches, coordinates,
                transforms=None):
                
        self.patches = patches
        self.coordinates = coordinates
        self.transforms = transforms
    
    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):    
        img = self.patches[idx]

        if self.transforms is not None:
            img = self.transforms(img)
        
        result = {}
        result['coordinates'] = self.coordinates[idx]
        result['image'] = img

        return result 