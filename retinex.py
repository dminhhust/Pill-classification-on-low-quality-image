import numpy as np
import cv2
import os
from pathlib import Path

def singleScaleRetinex(img, variance):
    retinex = np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), variance))
    return retinex

def multiScaleRetinex(img, variance_list):
    retinex = np.zeros_like(img)
    for variance in variance_list:
        retinex += singleScaleRetinex(img, variance)
    retinex = retinex / len(variance_list)
    return retinex

def MSR(img, variance_list):
    img = np.float64(img) + 1.0
    img_retinex = multiScaleRetinex(img, variance_list)

    for i in range(img_retinex.shape[2]):
        unique, count = np.unique(np.int32(img_retinex[:, :, i] * 100), return_counts=True)
        for u, c in zip(unique, count):
            if u == 0:
                zero_count = c
                break            
        low_val = unique[0] / 100.0
        high_val = unique[-1] / 100.0
        for u, c in zip(unique, count):
            if u < 0 and c < zero_count * 0.1:
                low_val = u / 100.0
            if u > 0 and c < zero_count * 0.1:
                high_val = u / 100.0
                break            
        img_retinex[:, :, i] = np.maximum(np.minimum(img_retinex[:, :, i], high_val), low_val)
        
        img_retinex[:, :, i] = (img_retinex[:, :, i] - np.min(img_retinex[:, :, i])) / \
                               (np.max(img_retinex[:, :, i]) - np.min(img_retinex[:, :, i])) \
                               * 255
    img_retinex = np.uint8(img_retinex)        
    return img_retinex

def SSR(img, variance):
    img = np.float64(img) + 1.0
    img_retinex = singleScaleRetinex(img, variance)
    for i in range(img_retinex.shape[2]):
        unique, count = np.unique(np.int32(img_retinex[:, :, i] * 100), return_counts=True)
        for u, c in zip(unique, count):
            if u == 0:
                zero_count = c
                break            
        low_val = unique[0] / 100.0
        high_val = unique[-1] / 100.0
        for u, c in zip(unique, count):
            if u < 0 and c < zero_count * 0.1:
                low_val = u / 100.0
            if u > 0 and c < zero_count * 0.1:
                high_val = u / 100.0
                break            
        img_retinex[:, :, i] = np.maximum(np.minimum(img_retinex[:, :, i], high_val), low_val)
        
        img_retinex[:, :, i] = (img_retinex[:, :, i] - np.min(img_retinex[:, :, i])) / \
                               (np.max(img_retinex[:, :, i]) - np.min(img_retinex[:, :, i])) \
                               * 255
    img_retinex = np.uint8(img_retinex)        
    return img_retinex

def process_images_in_folder(input_folder, output_folder, method='both', variance_list=[15, 80, 30], variance=300):
    """
    Process all images in a folder with Retinex enhancement
    
    Parameters:
    - input_folder: Path to folder containing input images
    - output_folder: Path to folder where processed images will be saved
    - method: 'msr', 'ssr', or 'both' (default: 'both')
    - variance_list: List of variances for MSR (default: [15, 80, 30])
    - variance: Variance value for SSR (default: 300)
    """
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Supported image extensions
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    
    # Get all image files in the input folder
    image_files = []
    for file in os.listdir(input_folder):
        file_path = Path(file)
        if file_path.suffix.lower() in supported_extensions:
            image_files.append(file)
    
    if not image_files:
        print(f"No supported images found in {input_folder}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    for idx, image_file in enumerate(image_files, 1):
        input_path = os.path.join(input_folder, image_file)
        print(f"Processing {idx}/{len(image_files)}: {image_file}")
        
        # Read image
        img = cv2.imread(input_path)
        if img is None:
            print(f"  Warning: Could not read {image_file}, skipping...")
            continue
        
        # Process based on selected method
        if method.lower() == 'msr' or method.lower() == 'both':
            # Apply MSR
            img_msr = MSR(img, variance_list)
            # Save MSR result
            base_name = os.path.splitext(image_file)[0]
            msr_output = os.path.join(output_folder, f"{base_name}_msr.jpg")
            cv2.imwrite(msr_output, img_msr)
            print(f"  Saved MSR: {msr_output}")
        
        if method.lower() == 'ssr' or method.lower() == 'both':
            # Apply SSR
            img_ssr = SSR(img, variance)
            # Save SSR result
            base_name = os.path.splitext(image_file)[0]
            ssr_output = os.path.join(output_folder, f"{base_name}_ssr.jpg")
            cv2.imwrite(ssr_output, img_ssr)
            print(f"  Saved SSR: {ssr_output}")
    
    print(f"\nProcessing complete! Processed {len(image_files)} images.")
    print(f"Results saved to: {output_folder}")

# Main execution
if __name__ == "__main__":
    # Configuration
    input_folder = "train"  # Folder containing your images
    output_folder = "train"  # Folder where enhanced images will be saved
    method = "both"  # Options: "msr", "ssr", or "both"
    
    # Variance parameters (you can adjust these)
    variance_list = [15, 80, 30]  # For MSR
    variance = 300  # For SSR
    
    # Process all images
    process_images_in_folder(input_folder, output_folder, method, variance_list, variance)