import numpy as np
import h5py
from PIL import Image

def LoadHSI(path_to_hdf5, return_wlens = False, print_info = False):
    # Open the HDF5 file
    # we handle two type, h5 and hdf5
    
    filetype = path_to_hdf5.split('.')[-1]
    with h5py.File(path_to_hdf5, 'r') as f:
        if filetype =='h5':
            hypercube_dataset = 'Hypercube'
            wavelength_dataset = 'Wavelengths'
        else:
            hypercube_dataset = 'hypercube'
            wavelength_dataset = 'wavelength_nm'
        if print_info:
            print("Datasets in the file:")
            for name in f.keys():
                print(name)
            
                data = f[name]
                print("Attributes of the dataset:")
                for key in data.attrs.keys():
                    print(f"{key}: {data.attrs[key]}")

        if filetype =='h5':
            dataset = f[hypercube_dataset]
            hcube = f[hypercube_dataset][:]
            hcube = np.transpose(hcube, (2, 0, 1))
            wlens = f[wavelength_dataset][:]
        else:
            dataset = f[hypercube_dataset] 
            hcube = f[hypercube_dataset][:]
            wlens = np.array(dataset.attrs[wavelength_dataset])
        
        if not return_wlens:
            return hcube
        else:
            return hcube, wlens
        

def read_mask(file_path):
    mask = np.array(Image.open(file_path), dtype=np.float32)  
    if len(mask.shape)>2 and mask.shape[2] == 4:
        # normalize the values, ensure they are integers for segmentation labels, and then restrict range
        mask = mask[:, :, 0]
        mask = ((mask / 255.0) * 2).round().astype(int)
    
    return mask