import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

from models import SAM
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

# Set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_OFFICIAL = "checkpoint/sam_vit_b_01ec64.pth"
CHECKPOINT_CUSTOM = "our_sam_loaded.pth"

# Prompt data
points_coords = []
points_labels = []
current_box = None

# Model inputs and outputs
image = None
original_size = None
transform = None
input_image_torch = None
input_size = None

official_features = None
custom_features = None

official_model = None
custom_model = None

# Plot axes and visuals
fig = None
ax_img = None
ax_custom = None
ax_official = None

plot_points = []
plot_box_rect = None

def generate_synthetic_image():
    """Generates a synthetic image with a white circle on a black background."""
    print("No image provided. Generating synthetic image (dummy_input.png)...")
    img = np.zeros((800, 800, 3), dtype=np.uint8)
    cv2.circle(img, (400, 400), 150, (255, 255, 255), -1)
    cv2.imwrite("dummy_input.png", img)
    return "dummy_input.png"

def preprocess_image(img, tf):
    """Transforms, normalizes, and pads the input image to 1024x1024."""
    input_image = tf.apply_image(img)
    input_image_torch = torch.as_tensor(input_image, device=DEVICE)
    input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()
    
    pixel_mean = torch.tensor([123.675, 116.28, 103.53], device=DEVICE).view(-1, 1, 1)
    pixel_std = torch.tensor([58.395, 57.12, 57.375], device=DEVICE).view(-1, 1, 1)
    input_image_torch = (input_image_torch - pixel_mean) / pixel_std
    
    h, w = input_image_torch.shape[-2:]
    padh = 1024 - h
    padw = 1024 - w
    input_image_torch = F.pad(input_image_torch, (0, padw, 0, padh))
    return input_image_torch[None, :, :, :], (h, w)

def postprocess_mask(low_res_masks, in_sz, orig_sz):
    """Upscales masks back to original image resolution."""
    masks = F.interpolate(low_res_masks, (1024, 1024), mode="bilinear", align_corners=False)
    masks = masks[..., :in_sz[0], :in_sz[1]]
    masks = F.interpolate(masks, orig_sz, mode="bilinear", align_corners=False)
    return masks

def update_predictions():
    """Runs prompt encoding and decoding for both models, updating UI."""
    global plot_points, plot_box_rect
    
    # 1. Clear previous overlays on subplots
    for artist in plot_points:
        artist.remove()
    plot_points = []
    
    if plot_box_rect is not None:
        plot_box_rect.remove()
        plot_box_rect = None

    # 2. Draw current prompts on the input canvas
    # Points
    for pt, lbl in zip(points_coords, points_labels):
        color = "red" if lbl == 1 else "blue"
        p_plot = ax_img.scatter(pt[0], pt[1], color=color, marker="o", s=80, edgecolors="white", zorder=5)
        plot_points.append(p_plot)
        
    # Bounding Box
    if current_box is not None:
        x1, y1, x2, y2 = current_box
        plot_box_rect = ax_img.add_patch(
            plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                fill=False, edgecolor="green", linewidth=2.5, zorder=5
            )
        )

    # 3. Setup prompt tensors if prompts are present
    coords_torch, labels_torch, box_torch = None, None, None
    
    if len(points_coords) > 0:
        pts = np.array(points_coords)
        lbls = np.array(points_labels)
        
        transformed_coords = transform.apply_coords(pts, original_size)
        transformed_coords_torch = torch.as_tensor(transformed_coords, dtype=torch.float, device=DEVICE)
        transformed_labels_torch = torch.as_tensor(lbls, dtype=torch.int, device=DEVICE)
        
        coords_torch = transformed_coords_torch[None, :, :]
        labels_torch = transformed_labels_torch[None, :]

    if current_box is not None:
        box_np = np.array([current_box])
        transformed_box = transform.apply_boxes(box_np, original_size)
        box_torch = torch.as_tensor(transformed_box, dtype=torch.float, device=DEVICE)

    # 4. Predict and display if there's any active prompt
    if coords_torch is not None or box_torch is not None:
        points_input = (coords_torch, labels_torch) if coords_torch is not None else None
        
        with torch.no_grad():
            # A. Official Model
            official_sparse, official_dense = official_model.prompt_encoder(
                points=points_input,
                boxes=box_torch,
                masks=None
            )
            official_low_res, official_iou = official_model.mask_decoder(
                image_embeddings=official_features,
                image_pe=official_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=official_sparse,
                dense_prompt_embeddings=official_dense,
                multimask_output=False
            )
            official_masks_full = postprocess_mask(official_low_res, input_size, original_size)
            official_mask_np = (official_masks_full[0, 0] > 0.0).cpu().numpy()
            official_iou_val = official_iou[0, 0].item()

            # B. Custom Model
            custom_sparse, custom_dense = custom_model.prompt_encoder(
                points=points_input,
                boxes=box_torch,
                masks=None
            )
            custom_low_res, custom_iou = custom_model.mask_decoder(
                image_embeddings=custom_features,
                image_pe=custom_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=custom_sparse,
                dense_prompt_embeddings=custom_dense,
                multimask_output=False
            )
            custom_masks_full = postprocess_mask(custom_low_res, input_size, original_size)
            custom_mask_np = (custom_masks_full[0, 0] > 0.0).cpu().numpy()
            custom_iou_val = custom_iou[0, 0].item()

        # Update Mask Visuals
        # Custom Overlay
        ax_custom.images[0].set_data(image)
        custom_overlay = np.zeros((*custom_mask_np.shape, 4))
        custom_overlay[custom_mask_np] = [1.0, 0.5, 0.0, 0.6] # Orange with 60% opacity
        ax_custom.images[1].set_data(custom_overlay)
        ax_custom.set_title(f"Custom Model (IoU: {custom_iou_val:.3f})")

        # Official Overlay
        ax_official.images[0].set_data(image)
        official_overlay = np.zeros((*official_mask_np.shape, 4))
        official_overlay[official_mask_np] = [0.0, 0.5, 1.0, 0.6] # Blue with 60% opacity
        ax_official.images[1].set_data(official_overlay)
        ax_official.set_title(f"Official Model (IoU: {official_iou_val:.3f})")
    else:
        # Clear overlays if no prompts
        empty_overlay = np.zeros((*original_size, 4))
        ax_custom.images[0].set_data(image)
        ax_custom.images[1].set_data(empty_overlay)
        ax_custom.set_title("Custom Model")
        
        ax_official.images[0].set_data(image)
        ax_official.images[1].set_data(empty_overlay)
        ax_official.set_title("Official Model")

    # Refresh window
    fig.canvas.draw_idle()

press_x = None
press_y = None
press_button = None

def on_press(event):
    global press_x, press_y, press_button
    if event.inaxes != ax_img:
        return
    press_x = event.xdata
    press_y = event.ydata
    press_button = event.button

def on_release(event):
    global points_coords, points_labels, press_x, press_y, press_button
    if event.inaxes != ax_img:
        return
    if press_x is None or press_y is None or event.xdata is None or event.ydata is None:
        return
    
    dx = abs(event.xdata - press_x)
    dy = abs(event.ydata - press_y)
    
    # If mouse moved less than 5 pixels, treat as a single click point prompt
    if dx < 5 and dy < 5:
        if press_button == 1:  # Left click (Positive)
            points_coords.append([press_x, press_y])
            points_labels.append(1)
            print(f"Added positive prompt point at: ({press_x:.1f}, {press_y:.1f})")
            update_predictions()
        elif press_button == 3:  # Right click (Negative)
            points_coords.append([press_x, press_y])
            points_labels.append(0)
            print(f"Added negative prompt point at: ({press_x:.1f}, {press_y:.1f})")
            update_predictions()
            
    # Reset press variables
    press_x = None
    press_y = None
    press_button = None

def on_select(eclick, erelease):
    """Callback for RectangleSelector (handles box dragging)."""
    global current_box
    
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    
    if None in (x1, y1, x2, y2):
        return

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    
    if dx >= 5 or dy >= 5:
        # It's a drag bounding box prompt
        xmin, xmax = min(x1, x2), max(x1, x2)
        ymin, ymax = min(y1, y2), max(y1, y2)
        current_box = [xmin, ymin, xmax, ymax]
        print(f"Set box prompt: [{xmin:.1f}, {ymin:.1f}, {xmax:.1f}, {ymax:.1f}]")
        update_predictions()

def on_key(event):
    """Handles key press events (press 'c' to clear prompts)."""
    global points_coords, points_labels, current_box
    if event.key == "c":
        points_coords = []
        points_labels = []
        current_box = None
        print("Cleared all prompts!")
        update_predictions()

def main():
    global image, original_size, transform, input_image_torch, input_size
    global official_features, custom_features, official_model, custom_model
    global fig, ax_img, ax_custom, ax_official

    parser = argparse.ArgumentParser(description="Interactive comparison UI between Custom and Official SAM.")
    parser.add_argument("--image", type=str, default=None, help="Path to custom image.")
    args = parser.parse_args()

    # 1. Load weights
    if not os.path.exists(CHECKPOINT_OFFICIAL):
        raise FileNotFoundError(f"Official weights not found at '{CHECKPOINT_OFFICIAL}'!")
    if not os.path.exists(CHECKPOINT_CUSTOM):
        raise FileNotFoundError(f"Custom weights not found at '{CHECKPOINT_CUSTOM}'! Please run load_weights.py first.")

    # 2. Get image path
    if args.image is None:
        image_path = generate_synthetic_image()
    else:
        image_path = args.image

    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_size = image.shape[:2]

    # Preprocess
    transform = ResizeLongestSide(1024)
    input_image_torch, input_size = preprocess_image(image, transform)

    # 3. Initialize models and precompute features
    print("Initializing Official SAM model...")
    official_model = sam_model_registry["vit_b"](checkpoint=CHECKPOINT_OFFICIAL).to(DEVICE)
    official_model.eval()

    print("Initializing Custom SAM model...")
    custom_model = SAM().to(DEVICE)
    # Map and load weights dynamically
    official_state = official_model.state_dict()
    our_state = {}
    for k, v in official_state.items():
        if "image_encoder.blocks." in k and ".attn.qkv." in k:
            prefix = k.split(".attn.qkv.")[0]
            suffix = k.split(".attn.qkv.")[1]
            q, k_t, v_t = torch.chunk(v, 3, dim=0)
            our_state[f"{prefix}.attn.q_proj.{suffix}"] = q
            our_state[f"{prefix}.attn.k_proj.{suffix}"] = k_t
            our_state[f"{prefix}.attn.v_proj.{suffix}"] = v_t
        elif "image_encoder.blocks." in k and ".attn.proj." in k:
            prefix = k.split(".attn.proj.")[0]
            suffix = k.split(".attn.proj.")[1]
            our_state[f"{prefix}.attn.out_proj.{suffix}"] = v
        else:
            our_state[k] = v
    custom_model.load_state_dict(our_state)
    custom_model.eval()

    print("Precomputing image features (this might take a few seconds)...")
    with torch.no_grad():
        official_features = official_model.image_encoder(input_image_torch)
        custom_features = custom_model.image_encoder(input_image_torch)

    # 4. Set up interactive Matplotlib subplots
    fig, (ax_img, ax_custom, ax_official) = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Interactive SAM Prompt Comparison: Custom vs Official", fontsize=16)

    # Draw image on input canvas
    ax_img.imshow(image)
    ax_img.set_title("Prompt canvas (Left Click=Pos, Right Click=Neg, Drag=Box)")
    ax_img.axis("on")

    # Draw initial blank images and overlay layers
    empty_overlay = np.zeros((*original_size, 4))
    
    ax_custom.imshow(image)
    ax_custom.imshow(empty_overlay)
    ax_custom.set_title("Custom Model")
    ax_custom.axis("off")

    ax_official.imshow(image)
    ax_official.imshow(empty_overlay)
    ax_official.set_title("Official Model")
    ax_official.axis("off")

    # 5. Connect event handlers
    # Left click & drag selector
    selector = RectangleSelector(
        ax_img, on_select, useblit=True,
        button=[1], minspanx=5, minspany=5,
        props=dict(facecolor='green', edgecolor='green', alpha=0.2, fill=True)
    )
    
    # Press & Release connectors
    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("button_release_event", on_release)
    # Key press connector ('c' to clear)
    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.tight_layout()
    print("\n" + "="*80)
    print("INTERACTIVE COMPARISON GUI CONTROLS:")
    print("="*80)
    print("- Left Click: Add a positive (foreground) prompt point (Red)")
    print("- Right Click: Add a negative (background) prompt point (Blue)")
    print("- Left Click & Drag: Draw a prompt bounding box (Green)")
    print("- Press 'c' on Keyboard: Clear all prompt points and boxes")
    print("="*80 + "\n")
    
    plt.show()

if __name__ == "__main__":
    main()
