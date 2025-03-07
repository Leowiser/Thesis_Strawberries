import os
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from utils import *

# Custom dataset class for RGB images
class RgbDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, apply_mask=False, mask_method=1):
        super().__init__()
        self.img_dir = img_dir
        self.transform = transform
        self.apply_mask = apply_mask
        self.mask_method = mask_method
        self.img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir)]
        self.mask_paths = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir)]
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        image_path = self.img_paths[idx]
        image = Image.open(image_path)
        if self.apply_mask == True:
            # Open mask
            mask_path = self.mask_paths[idx]
            mask = read_mask(mask_path)
            
            # Apply mask to image
            image = Image.fromarray(np.array(image)*np.expand_dims(np.where(mask==2, self.mask_method, mask), axis=-1).astype('uint8'))
        
        # Convert to tensor. Tensors are needed already in the custom collate_fn and at the transforms (specified in self.transform).
        image = transforms.functional.to_tensor(image)
        
        if self.transform:
            image = self.transform(image)
            
        return image
    
    
    

# Custom dataset class for HSI data
class HsiDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, apply_mask=False, mask_method=1):
        super().__init__()
        self.img_dir = img_dir
        self.transform = transform
        self.apply_mask = apply_mask
        self.mask_method = mask_method
        self.img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir)]
        self.mask_paths = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir)]
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        image_path = self.img_paths[idx]
        hsi_np = LoadHSI(image_path, return_wlens=False)
        
        # Set all negative values to 0 (these are noise) and normalize the hsi
        hsi_np = np.maximum(hsi_np, 0)
        hsi_np = hsi_np / np.max(hsi_np)
        
        if self.apply_mask == True:
            # Open mask
            mask_path = self.mask_paths[idx]
            mask = read_mask(mask_path)
            
            # Apply mask to image
            hsi_np = hsi_np * np.where(mask == 2, self.mask_method, mask)
        
        # At this point hsi_np is a numpy.ndarray with shape (CxHxW).
        # However it will need to be changed to torch.Tensor (with transforms.functional.to_tensor or transforms.ToTensor).
        # Those methods need the input ndarray to have shape (HxWxC). So we transpose hsi_np.
        hsi_np = hsi_np.transpose((1,2,0))
        
        # Convert to tensor. Tensors are needed already in the custom collate_fn and at the transforms (specified in self.transform).
        hsi_np = transforms.functional.to_tensor(hsi_np)
        
        if self.transform:
            hsi_np = self.transform(hsi_np)

        return hsi_np
    
    
    
    
# Custom collate function that performs padding
def padded_collate(batch):

    # Find the max height and width in the batch
    max_height = max(img.shape[1] for img in batch)
    max_width = max(img.shape[2] for img in batch)
    
    # Pad all images to the max height and width - image will be in top-left after padding
    padded_batch = [
        transforms.functional.pad(img, (0, 0, max_width - img.shape[2], max_height - img.shape[1]))
        for img in batch
    ]
    
    return torch.stack(padded_batch)
