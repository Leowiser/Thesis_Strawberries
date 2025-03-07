import os
from utils import *
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def load_and_flatten_hsi(img_dir, mask_dir=None, individual_normalize=False, apply_mask=False, mask_method=1):
    """
    Transforms the 3D hyperspectral images into a 2D array by flattening the spatial dimensions. The resulting "rows" are the pixels
    and the "columns" store their values for the different spectral bands. 
    Can be used with a single- or multiple HSI-s. If multiple HSI-s are provided, they are stacked together.

    Parameters:
    - img_dir: str, path to the folder containing the HSI-s
    - mask_dir: str, path to the folder containing the masks for the HSI-s
    - individual_normalize: bool, whether to normalize each HSI individually before flattening and stacking
    - apply_mask: bool, whether to apply the mask to the HSI-s
    - mask_method: int, 0 for keeping only leaf, 1 for keeping leaf+stem after applying the mask

    Returns:
    - numpy array, shape (n_pixels, n_bands), the flattened and stacked HSI pixels
    """
    all_pixels = []
    
    for file in os.listdir(img_dir):
        hsi_np = LoadHSI(os.path.join(img_dir, file))
        
        # Set all negative values to 0 (these are noise)
        hsi_np = np.maximum(hsi_np, 0)
        
        # Load mask if required
        if apply_mask and mask_dir:
            mask_file = os.path.join(mask_dir, os.path.splitext(file)[0] + ".png")    # Find the mask for the HSI (same name)
            mask_np = read_mask(mask_file)
            hsi_np = hsi_np * np.where(mask_np == 2, mask_method, mask_np)

        # Flatten: (bands, height, width) → (height*width, bands)
        hsi_np = hsi_np.reshape(hsi_np.shape[0], -1).T
        
        # Apply mask if required - Remove background (zero) pixels
        if apply_mask:
            hsi_np = hsi_np[~np.all(hsi_np == 0, axis=1)]
        
        # Normalize each image individually if required
        if individual_normalize:
            hsi_np = hsi_np / np.max(hsi_np)

        all_pixels.append(hsi_np)
    
    # Stack all pixels together
    return np.vstack(all_pixels)



def hsi_transform_to_pca_space(hsi_np, pca, scaler, mask_np=None, mask_method=1):
    """
    Apply pre-trained PCA on HSI to reduce spectral bands, i.e. transform data to PCA space.
    If a mask is provided, only the valid non-background pixels are transformed to the PCA space with the pre-trained PCA model. Also,
    the background pixels will be set to 0 in the PCA space as well.

    Parameters:
    - hsi_np: numpy array, Hyperspectral image with shape (bands, height, width).
    - pca: Pre-fitted PCA model.
    - scaler: Pre-fitted StandardScaler model.
    - mask_np: numpy array, mask to apply on the HSI.
    - mask_method: int, 0 for keeping only leaf, 1 for keeping leaf+stem after applying the mask.

    Returns:
    - numpy array of PCA-transformed HSI with shape (pca_components, height, width).
    """
    # Set all negative values to 0 (these are noise)
    # hsi_np = np.maximum(hsi_np, 0)    # Q: Interestingly if we do this, the reconstruction errors show vertical lines
    # A: Because we would need to change negative values to 0 in the function that calculates reconstruction error as well to the
    # "original" hsi_np that we compare the reconstructed to. → Better apply np.maximum(hsi_np, 0) before calling the function and not inside it.
    
    # Flatten: (bands, height, width) → (height*width, bands)
    hsi_flattened = hsi_np.reshape(hsi_np.shape[0], -1).T
    
    # Apply mask if provided (exclude background from PCA)
    if mask_np is not None:
        mask_np = np.where(mask_np == 2, mask_method, mask_np)
        valid_indices = mask_np.flatten() != 0
        hsi_valid = hsi_flattened[valid_indices]    # Keep only non-background pixels
    else:
        hsi_valid = hsi_flattened

    # Standardize using the previously fitted scaler
    hsi_valid_scaled = scaler.transform(hsi_valid)

    # Apply the pre-trained PCA model
    hsi_pca_valid = pca.transform(hsi_valid_scaled)

    # Return to the original number of pixels, with 0s for background pixels if mask was applied and PCA-transformed pixels for the rest
    if mask_np is not None:
        hsi_pca = np.zeros((hsi_flattened.shape[0], pca.n_components_))    # Initialize empty array with the shape of the original pixels
        hsi_pca[valid_indices] = hsi_pca_valid    # Insert only valid transformed pixels
    else:
        hsi_pca = hsi_pca_valid    # No mask applied, just return transformed pixels

    # Reshape back to (pca_components, height, width)
    return hsi_pca.T.reshape(pca.n_components_, hsi_np.shape[1], hsi_np.shape[2])



def reconstruct_hsi_from_pca_space(hsi_pca, pca, scaler, mask_np=None, mask_method=1):
    """
    Back-transform PCA-transformed HSI to original space.
    If a mask is provided, only the valid non-background pixels are reconstructed from the PCA space to the original space, with background
    pixels set to 0.
    
    Parameters:
    - hsi_pca: numpy array, PCA-transformed HSI with shape (pca_components, height, width).
    - pca: Pre-fitted PCA model.
    - scaler: Pre-fitted StandardScaler model.
    - mask_np: numpy array, mask to apply on the HSI.
    - mask_method: int, 0 for keeping only leaf, 1 for keeping leaf+stem after applying the mask.

    Returns:
    - numpy array of back-transformed (reconstructed) HSI with shape (bands, height, width).
    """
    # Flatten: (pca_components, height, width) → (height*width, pca_components)
    hsi_pca_flattened = hsi_pca.reshape(pca.n_components_, -1).T
    
    # Apply mask if provided (exclude background from PCA)
    if mask_np is not None:
        mask_np = np.where(mask_np == 2, mask_method, mask_np)
        valid_indices = mask_np.flatten() != 0
        hsi_valid = hsi_pca_flattened[valid_indices]    # Keep only non-background pixels
    else:
        hsi_valid = hsi_pca_flattened

    # Apply inverse PCA
    hsi_valid_reconstructed = pca.inverse_transform(hsi_valid)

    # Apply inverse scaling
    hsi_valid_reconstructed = scaler.inverse_transform(hsi_valid_reconstructed)
    
    # Reconstruct full spatial structure if mask was applied
    if mask_np is not None:
        hsi_reconstructed = np.zeros((hsi_pca_flattened.shape[0], hsi_valid_reconstructed.shape[1]))    # Initialize empty array of shape (original pixels, original bands)
        hsi_reconstructed[valid_indices] = hsi_valid_reconstructed    # Insert only valid transformed pixels
    else:
        hsi_reconstructed = hsi_valid_reconstructed    # No mask applied, just return reconstructed pixels

    # Reshape back to (bands, height, width)
    return hsi_reconstructed.T.reshape(-1, hsi_pca.shape[1], hsi_pca.shape[2])



def compress_and_reconstruct_hsi_pca(hsi_np, pca, scaler, mask_np=None, mask_method=1):
    '''
    Preform PCA compression and reconstruction right after on a HSI data.
    
    Parameters:
    - hsi_np: numpy array, Hyperspectral image with shape (bands, height, width).
    - pca: Pre-fitted PCA model.
    - scaler: Pre-fitted StandardScaler model.
    - mask_np: numpy array, mask to apply on the HSI.
    - mask_method: int, 0 for keeping only leaf, 1 for keeping leaf+stem after applying the mask.    
    '''
    # Transform data to PCA space
    hsi_pca = hsi_transform_to_pca_space(hsi_np, pca, scaler, mask_np, mask_method)
    
    # Reconstruct data from PCA space
    hsi_reconstructed = reconstruct_hsi_from_pca_space(hsi_pca, pca, scaler, mask_np, mask_method)
    
    return hsi_reconstructed



def plot_pca_reconstruction_error(hsi_np, pca, scaler, mask_np=None, mask_method=1):
    """
    Reconstruct an input HSI using the pre-trained PCA and plot the reconstruction error.
    Should be used with a single HSI file.

    Parameters:
    - hsi_np: numpy array, Hyperspectral image with shape (bands, height, width)
    - pca: pre-fitted PCA model
    - scaler: pre-fitted StandardScaler model
    - mask_np: numpy array, mask to apply on the HSI
    - mask_method: int, 0 for keeping only leaf, 1 for keeping leaf+stem after applying the mask

    Returns:
    - Plot ("heatmap") of the sum of reconstruction errors across the bands
    """
    # Apply PCA and reconstruct
    hsi_pca = hsi_transform_to_pca_space(hsi_np, pca, scaler, mask_np, mask_method)
    hsi_reconstructed = reconstruct_hsi_from_pca_space(hsi_pca, pca, scaler, mask_np, mask_method)
    
    # Compute reconstruction error
    if mask_np is not None:
        mask_np = np.where(mask_np == 2, mask_method, mask_np)
        reconstruction_error = np.abs(hsi_np - hsi_reconstructed) * mask_np
    else:
        reconstruction_error = np.abs(hsi_np - hsi_reconstructed)

    # Sum reconstruction error across the bands
    pixel_errors = np.sum(reconstruction_error, axis=0)

    # Plot pixel errors
    plt.figure(figsize=(8, 6))
    plt.imshow(pixel_errors, cmap='hot')
    plt.colorbar(label='Total Reconstruction Error')
    plt.title('Total Reconstruction Error per Pixel')
    plt.axis('off')
    plt.show()