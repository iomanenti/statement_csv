import fitz  # PyMuPDF
from PIL import Image
import numpy as np
from skimage import color, filters, morphology, transform
import argparse
import os
import io
from tqdm import tqdm

def pdf_to_images(pdf_path):
    doc = fitz.open(pdf_path)
    images = []
    for i in range(len(doc)):
        try:
            page = doc[i]
            # Use get_pixmap on newer PyMuPDF, fallback to getPixmap for older versions
            get_pix = getattr(page, "get_pixmap", None) or getattr(page, "getPixmap")
            pix = get_pix(matrix=fitz.Matrix(300/72, 300/72), alpha=False)  # Scale to 300 DPI
            
            # Use pix.tobytes() which is safer
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data))
            images.append(np.array(image))
        except Exception as e:
            print(f"Error processing page {i+1}: {e}")
            continue
    doc.close()
    return images

def binarize_image(image_array):
    # Convert to grayscale if not already
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        gray_image = color.rgb2gray(image_array)
    else:
        gray_image = image_array
    
    # Ensure gray_image is float for thresholding
    if gray_image.dtype != np.float64:
        gray_image = gray_image.astype(np.float64) / 255.0 if gray_image.max() > 1 else gray_image.astype(np.float64)

    # Apply Otsu's thresholding for binarization, with a fallback for uniform images
    try:
        thresh = filters.threshold_otsu(gray_image)
    except ValueError: # Handle cases where threshold_otsu might fail (e.g., uniform image)
        thresh = 0.5 # Default to a mid-point threshold

    binary_image = gray_image > thresh
    return binary_image

def deskew_image(image_array):
    # Convert to grayscale if not already
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        gray_image = color.rgb2gray(image_array)
    else:
        gray_image = image_array

    # Estimate skew angle (example using moments, more robust methods exist)
    # This is a simplified example; real deskewing often involves Hough transforms or similar.
    # For simplicity, we'll assume a small skew and try to correct it.
    # A more advanced implementation would involve detecting text lines.
    
    # For demonstration, let's assume a fixed small angle or use a placeholder.
    # In a real scenario, you'd use a library like `imutils` or implement a more complex algorithm.
    # For now, we'll just return the original image, as scikit-image doesn't have a direct deskew function.
    # You might need to implement this using OpenCV or a more specialized library.
    return image_array

def remove_noise(image_array):
    # Apply a median filter to remove salt-and-pepper noise
    # For Gaussian noise, a Gaussian filter would be more appropriate
    denoised_image = filters.median(image_array)
    return denoised_image

def normalize_image(image_array):
    # Normalize image to 0-1 range (if not already) and then stretch contrast
    normalized_image = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))
    # You can add more advanced contrast stretching here if needed
    return normalized_image

def scale_image(image_array, dpi=300):
    # PyMuPDF already handles scaling to 300 DPI during conversion, 
    # but this function can be used for further scaling if needed.
    # For demonstration, we'll assume the input is already at a good DPI.
    # If you need to scale, you would calculate the new dimensions.
    return image_array

def thin_and_skeletonize_image(image_array):
    # Convert to grayscale and then to binary if not already
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        gray_image = color.rgb2gray(image_array)
    else:
        gray_image = image_array
    
    # Ensure image is binary (True/False) for skeletonization
    if gray_image.dtype != bool:
        thresh = filters.threshold_otsu(gray_image)
        binary_image = gray_image > thresh
    else:
        binary_image = gray_image

    # Apply skeletonization
    skeleton = morphology.skeletonize(binary_image)
    return skeleton

def thicken_lines(image_array, iterations=1, kernel_size=2):
    binary_image = image_array.astype(bool)

    inverted_binary_image = np.logical_not(binary_image)

    # Version-agnostic boolean footprint
    kernel_size = max(1, int(kernel_size))
    footprint = np.ones((kernel_size, kernel_size), dtype=bool)

    thickened_inverted_image = inverted_binary_image
    for _ in range(max(1, int(iterations))):
        thickened_inverted_image = morphology.binary_dilation(thickened_inverted_image, footprint=footprint)

    thickened_image = np.logical_not(thickened_inverted_image)
    return thickened_image

def enhance_image(image_array):
    # Apply enhancements in a typical order
    enhanced_image = normalize_image(image_array)
    enhanced_image = remove_noise(enhanced_image)
    # Deskewing is complex and often requires a binary image or text detection
    # For now, we'll call it but note its limitation.
    enhanced_image = deskew_image(enhanced_image)
    enhanced_image = binarize_image(enhanced_image)
    enhanced_image = thicken_lines(enhanced_image, iterations=1, kernel_size=3)  # thicken black lines
    # Scaling is handled by PyMuPDF initially, but can be reapplied if needed
    enhanced_image = scale_image(enhanced_image)
    # Thinning/Skeletonization is typically for specific OCR tasks, apply after binarization
    #enhanced_image = thin_and_skeletonize_image(enhanced_image)
    return enhanced_image

def main():
    parser = argparse.ArgumentParser(description="Process PDF file, slice into images, and apply image enhancements.")
    parser.add_argument("pdf_file_path", type=str, help="The path to the PDF file to process.")
    args = parser.parse_args()

    pdf_path = args.pdf_file_path

    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        return

    # Create output directory
    output_dir = os.path.join("temp", os.path.splitext(os.path.basename(pdf_path))[0])
    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing PDF: {pdf_path}")
    images = pdf_to_images(pdf_path)

    if not images:
        print("No images could be extracted from the PDF.")
        return

    print(f"Extracted {len(images)} pages. Applying enhancements...")
    for i, img_array in enumerate(tqdm(images, desc="Enhancing Pages")):
        # The description will be updated by tqdm, so the print statement below is optional
        # print(f"Enhancing page {i+1}...")
        enhanced_img_array = enhance_image(img_array)
        
        # Convert boolean array from skeletonization back to uint8 for saving
        if enhanced_img_array.dtype == bool:
            enhanced_img_array = (enhanced_img_array * 255).astype(np.uint8)
        
        # If the image is grayscale (2D array), convert to RGB for saving as JPEG
        if len(enhanced_img_array.shape) == 2:
            enhanced_img_array = color.gray2rgb(enhanced_img_array)

        # Ensure dtype uint8 for PIL
        if enhanced_img_array.dtype != np.uint8:
            enhanced_img_array = (np.clip(enhanced_img_array, 0, 1) * 255).astype(np.uint8)

        output_image_path = os.path.join(output_dir, f"page_{i+1}_enhanced.jpg")
        Image.fromarray(enhanced_img_array).save(output_image_path)
        # This print is also redundant if tqdm is used
        # print(f"Saved enhanced image for page {i+1} to {output_image_path}")

    print("PDF processing and image enhancement complete.")

if __name__ == "__main__":
    main()
