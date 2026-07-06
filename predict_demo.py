import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt

from models import SAM
from segment_anything.utils.transforms import ResizeLongestSide

# Set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHTS_PATH = "our_sam_loaded.pth"

def generate_synthetic_image():
    """Generates a synthetic image with a white circle on a black background."""
    print("No image provided. Generating a synthetic image (dummy_input.png)...")
    # 800x800 black image
    img = np.zeros((800, 800, 3), dtype=np.uint8)
    # Draw a white circle in the center with radius 150
    cv2.circle(img, (400, 400), 150, (255, 255, 255), -1)
    # Save it
    cv2.imwrite("dummy_input.png", img)
    return "dummy_input.png", np.array([[400, 400]]), np.array([1])

def preprocess_image(image, transform):
    """Transforms, normalizes, and pads the input image to 1024x1024."""
    # 1. Transform the image resolution using official ResizeLongestSide
    input_image = transform.apply_image(image)
    input_image_torch = torch.as_tensor(input_image, device=DEVICE)
    input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()
    
    # 2. Normalize color values using official SAM pixel mean and std
    pixel_mean = torch.tensor([123.675, 116.28, 103.53], device=DEVICE).view(-1, 1, 1)
    pixel_std = torch.tensor([58.395, 57.12, 57.375], device=DEVICE).view(-1, 1, 1)
    input_image_torch = (input_image_torch - pixel_mean) / pixel_std
    
    # 3. Pad the image to 1024x1024 square
    h, w = input_image_torch.shape[-2:]
    padh = 1024 - h
    padw = 1024 - w
    input_image_torch = F.pad(input_image_torch, (0, padw, 0, padh))
    
    # Add batch dimension: (1, 3, 1024, 1024)
    return input_image_torch[None, :, :, :], (h, w)

def postprocess_mask(low_res_masks, input_size, original_size):
    """Upscales masks back to original image resolution."""
    # 1. Upscale low-res masks (256x256) back to 1024x1024
    masks = F.interpolate(low_res_masks, (1024, 1024), mode="bilinear", align_corners=False)
    # 2. Crop the padding
    masks = masks[..., :input_size[0], :input_size[1]]
    # 3. Resize back to original image size
    masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
    return masks

def main():
    parser = argparse.ArgumentParser(description="Run inference using the custom SAM model.")
    parser.add_argument("--image", type=str, default=None, help="Path to the input image.")
    parser.add_argument("--point", type=int, nargs=2, default=None, metavar=("X", "Y"), 
                        help="Prompt point coordinates (X, Y) on the original image.")
    args = parser.parse_args()

    # Verify model weights exist
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(
            f"Weights file '{WEIGHTS_PATH}' not found. Please run 'python load_weights.py' first!"
        )

    # 1. Load or generate input image and prompt point
    if args.image is None:
        image_path, point_coords, point_labels = generate_synthetic_image()
    else:
        image_path = args.image
        if args.point is None:
            # Default to the middle of the loaded image
            img_temp = cv2.imread(image_path)
            h, w = img_temp.shape[:2]
            point_coords = np.array([[w // 2, h // 2]])
            print(f"No prompt point specified. Defaulting to center of image: {point_coords[0]}")
        else:
            point_coords = np.array([args.point])
        point_labels = np.array([1]) # 1 = foreground point

    # Load original image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_size = image.shape[:2]

    print(f"Loaded image: '{image_path}' with resolution {original_size[1]}x{original_size[0]}")
    print(f"Prompt point coordinate (X, Y): {point_coords[0]}")

    # 2. Preprocess input image and prompt coordinates
    transform = ResizeLongestSide(1024)
    input_image_torch, input_size = preprocess_image(image, transform)

    # Transform coordinates
    transformed_coords = transform.apply_coords(point_coords, original_size)
    transformed_coords_torch = torch.as_tensor(transformed_coords, dtype=torch.float, device=DEVICE)
    transformed_labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=DEVICE)
    # Add batch dimension: (1, N, 2) and (1, N)
    points = (transformed_coords_torch[None, :, :], transformed_labels_torch[None, :])

    # 3. Load SAM model and mapped weights
    print("Loading SAM model and weights...")
    model = SAM().to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.eval()

    # 4. Perform Inference
    print("Running inference...")
    with torch.no_grad():
        low_res_masks, iou_predictions = model(
            image=input_image_torch,
            points=points,
            multimask_output=False
        )

    # 5. Post-process mask back to original resolution
    masks = postprocess_mask(low_res_masks, input_size, original_size)
    # Threshold logits at 0.0 to get binary mask
    binary_mask = (masks[0, 0] > 0.0).cpu().numpy()
    iou_score = iou_predictions[0, 0].item()

    print(f"Inference complete. Predicted IoU score: {iou_score:.4f}")

    # -------------------------------------------------------------------------
    # PRINT MATRIX REPRESENTATION
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("MATRIX REPRESENTATION OF OUTPUT")
    print("="*80)
    print(f"Mask Matrix Shape: {binary_mask.shape} (H x W)")
    print(f"Total foreground pixels predicted: {np.sum(binary_mask)}")
    
    # Print a 15x15 slice around the prompt point to show matrix values
    px, py = point_coords[0]
    slice_y_start = max(0, py - 7)
    slice_y_end = min(original_size[0], py + 8)
    slice_x_start = max(0, px - 7)
    slice_x_end = min(original_size[1], px + 8)
    
    matrix_slice = binary_mask[slice_y_start:slice_y_end, slice_x_start:slice_x_end].astype(int)
    print(f"\n15x15 Binary Mask Slice around the prompt point (Y: {slice_y_start}-{slice_y_end-1}, X: {slice_x_start}-{slice_x_end-1}):")
    print("-" * 50)
    # Format grid printout for easy readability
    for row in matrix_slice:
        print(" ".join(map(str, row)))
    print("-" * 50)
    print("Legend: 1 = Mask (foreground), 0 = Background")

    # -------------------------------------------------------------------------
    # SAVE/VISUALIZE PLOT
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("VISUALIZING AND SAVING PLOT")
    print("="*80)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Left: Original image with point prompt
    axes[0].imshow(image)
    axes[0].scatter(px, py, color="red", marker="o", s=100, edgecolors="white", label="Prompt Point")
    axes[0].set_title("Original Image & Prompt Point")
    axes[0].legend()
    axes[0].axis("on")
    
    # Right: Mask overlay
    axes[1].imshow(image)
    # Create colored semi-transparent mask overlay (blue color)
    mask_overlay = np.zeros((*binary_mask.shape, 4))
    mask_overlay[binary_mask] = [0.0, 0.5, 1.0, 0.6]  # Blue with 60% opacity
    axes[1].imshow(mask_overlay)
    axes[1].scatter(px, py, color="red", marker="o", s=100, edgecolors="white")
    axes[1].set_title(f"Segmented Mask (IoU: {iou_score:.4f})")
    axes[1].axis("on")
    
    plt.tight_layout()
    output_filename = "output_visualization.png"
    plt.savefig(output_filename, dpi=150)
    print(f"Visualization saved successfully as '{output_filename}'!")

if __name__ == "__main__":
    main()
