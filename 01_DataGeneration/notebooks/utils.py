import json 

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_timeseries_with_metadata(data, timestamps, save_img_path, save_label_path,
                                  figsize=(12, 4), dpi=100, padding=2):
    """
    Plot time series WITHOUT peaks (clean image for training)
    Save transformation metadata for reverse mapping
    
    Returns:
        metadata dict with all info needed to convert back to original coordinates
    """
    # Calculate adaptive y-limits
    data_min = data.min()
    data_max = data.max()
    y_range = data_max - data_min
    

    y_min_plot = data_min - padding 
    y_max_plot = data_max + padding
    
    # Get timestamp range  
    timestamp_min = timestamps.iloc[0]
    timestamp_max = timestamps.iloc[-1]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Plot ONLY the time series (no peaks)
    ax.plot(timestamps, data, color="black", linewidth=1.5)
    
    # Set BOTH x and y limits explicitly
    ax.set_xlim(timestamp_min, timestamp_max)
    ax.set_ylim(y_min_plot, y_max_plot)
    
    # Grid and formatting
    ax.grid(True, which="both", ls="--", c="gray", alpha=0.5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    plt.tight_layout(pad=0)
    plt.savefig(save_img_path, bbox_inches='tight', pad_inches=0, dpi=dpi, facecolor='white')
    plt.close()
    
    # *** CRITICAL: Get actual image dimensions from saved file ***
    img = Image.open(save_img_path)
    img_width, img_height = img.size
    print(f"Actual image size: {img_width}x{img_height} pixels")
    
    # Create metadata dictionary
    metadata = {
        'image_path': save_img_path,
        'label_path': save_label_path,
        'image_width': img_width,
        'image_height': img_height,
        'timestamp_min': timestamp_min,
        'timestamp_max': timestamp_max,
        'data_min': float(data_min),
        'data_max': float(data_max),
        'y_min_plot': float(y_min_plot),
        'y_max_plot': float(y_max_plot),
        'y_range_plot': float(y_max_plot - y_min_plot),
        'num_datapoints': len(data),
        'figsize': figsize,
        'dpi': dpi
    }
    
    # Save metadata as JSON (need to convert timestamps to string for JSON)
    metadata_json = metadata.copy()
    metadata_json['timestamp_min'] = str(timestamp_min)
    metadata_json['timestamp_max'] = str(timestamp_max)

    with open(save_label_path, 'w') as f:
        json.dump(metadata_json, f, indent=2)
    
    return metadata


def create_bounding_boxes(peaks_data, timestamps, data, metadata, 
                         box_width_ratio=0.01, box_height_ratio=0.1):
    """
    Create bounding boxes centered at peak locations
    
    Args:
        peaks_data: list of (timestamp, amplitude, label) tuples
        timestamps: array of timestamps from the original data
        data: the actual EDI data array
        metadata: metadata dict from plot_timeseries_with_metadata
        box_width_ratio: bbox width as fraction of image width (default 0.02 = 2%)
        box_height_ratio: bbox height as fraction of image height (default 0.1 = 10%)
    
    Returns:
        list of dicts with bounding box info in both pixel and normalized coords
    """
    img_width = metadata['image_width']
    img_height = metadata['image_height']
    
    timestamp_min = metadata['timestamp_min']
    timestamp_max = metadata['timestamp_max']
    
    y_min_plot = metadata['y_min_plot']
    y_max_plot = metadata['y_max_plot']
    y_range_plot = metadata['y_range_plot']
    
    boxes = []
    
    for peak_timestamp, peak_amplitude, label in peaks_data:
        # ===== X-COORDINATE CALCULATION =====
        # Map timestamp to x-coordinate (0 to 1)
        # timestamp_min maps to x=0, timestamp_max maps to x=1
        timestamp_range = timestamp_max - timestamp_min
        
        x_fraction = float(peak_timestamp - timestamp_min) / float(timestamp_range)
        
        # Clamp to [0, 1]
        x_fraction = max(0.0, min(1.0, x_fraction))
        
        # ===== Y-COORDINATE CALCULATION =====
        # Map amplitude to y-coordinate (0 to 1)
        # y_min_plot maps to y=1 (bottom), y_max_plot maps to y=0 (top) - flipped!
        y_fraction = (peak_amplitude - y_min_plot) / y_range_plot
        y_fraction_flipped = 1.0 - y_fraction  # Flip because image y goes top-down
        
        # Clamp to [0, 1]
        y_fraction_flipped = max(0.0, min(1.0, y_fraction_flipped))
        
        # ===== PIXEL COORDINATES =====
        x_center_px = x_fraction * img_width
        y_center_px = y_fraction_flipped * img_height
        
        # Calculate box dimensions in pixels
        box_width_px = box_width_ratio * img_width
        box_height_px = box_height_ratio * img_height
        
        # Normalized coordinates for YOLO format
        x_center_norm = x_fraction
        y_center_norm = y_fraction_flipped
        width_norm = box_width_ratio
        height_norm = box_height_ratio
        
        boxes.append({
            'label': label,
            'peak_timestamp': str(peak_timestamp),
            'peak_amplitude': float(peak_amplitude),
            'bbox_normalized': {
                'x_center': x_center_norm,
                'y_center': y_center_norm,
                'width': width_norm,
                'height': height_norm
            },
            'bbox_pixels': {
                'x_center': x_center_px,
                'y_center': y_center_px,
                'width': box_width_px,
                'height': box_height_px
            }
        })
    
    return boxes


def save_yolo_labels(boxes, save_path, class_mapping={'brb_peak': 0}):
    """
    Save bounding boxes in YOLO format
    """
    with open(save_path, 'w') as f:
        for box in boxes:
            label = box['label']
            class_id = class_mapping.get(label, 0)
            
            bbox = box['bbox_normalized']
            f.write(f"{class_id} {bbox['x_center']:.6f} {bbox['y_center']:.6f} "
                   f"{bbox['width']:.6f} {bbox['height']:.6f}\n")


def visualize_bounding_boxes(image_path, boxes, save_path=None):
    """
    Visualize the bounding boxes on the image for verification
    """

    
    img = Image.open(image_path)
    fig, ax = plt.subplots(1, figsize=(14, 5))
    ax.imshow(img)
    
    # Color mapping for visualization
    color_map = {'brb_peak': 'red', 'sigh': 'blue', 'apnea': 'green'}
    
    for box in boxes:
        bbox_px = box['bbox_pixels']
        label = box['label']
        color = color_map.get(label, 'yellow')
        
        # Calculate corner position (top-left)
        x_min = bbox_px['x_center'] - bbox_px['width'] / 2
        y_min = bbox_px['y_center'] - bbox_px['height'] / 2
        
        # Draw rectangle
        rect = patches.Rectangle(
            (x_min, y_min), bbox_px['width'], bbox_px['height'],
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add label
        ax.text(bbox_px['x_center'], y_min - 5, f"{label}\n({bbox_px['x_center']:.0f}, {bbox_px['y_center']:.0f})", 
                color=color, fontsize=8, ha='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # plt.title('Bounding Boxes Visualization (for verification only)')
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()
    plt.close()


def bbox_to_original_coordinates(bbox_normalized, metadata, timestamp_values=None):
    """
    Convert bounding box from normalized YOLO coordinates to original time series values
    
    Args:
        bbox_normalized: [x_center, y_center, width, height] in 0-1 range
        metadata: metadata dict from plot_timeseries_with_metadata
        timestamp_values: actual timestamp array (optional, for precise timestamp recovery)
    
    Returns:
        dict with original coordinates
    """
    x_center_norm, y_center_norm, width_norm, height_norm = bbox_normalized
    
    # Image dimensions
    img_width = metadata['image_width']
    img_height = metadata['image_height']
    
    # Convert normalized coordinates to pixel coordinates
    x_center_px = x_center_norm * img_width
    y_center_px = y_center_norm * img_height
    width_px = width_norm * img_width
    height_px = height_norm * img_height
    
    # Convert x (pixel) to timestamp
    # x_pixel=0 corresponds to timestamp_min, x_pixel=img_width corresponds to timestamp_max
    timestamp_range = metadata['timestamp_range']
    timestamp_min = metadata['timestamp_min']
    
    # Fractional position along x-axis
    x_fraction = x_center_px / img_width
    
    # Calculate timestamp (if timestamp_range is numeric)
    if isinstance(timestamp_range, (int, float)):
        timestamp_original = timestamp_min + (x_fraction * timestamp_range)
    else:
        # For datetime timestamps, you'll need to handle separately
        timestamp_original = x_fraction  # Return fraction for now
    
    # Convert y (pixel) to amplitude
    # NOTE: Image y-coordinates are top-down, so we need to flip
    y_min_plot = metadata['y_min_plot']
    y_max_plot = metadata['y_max_plot']
    y_range_plot = metadata['y_range_plot']
    
    # Flip y coordinate (image coordinates are top-down)
    y_fraction = 1.0 - (y_center_px / img_height)
    
    # Calculate original amplitude
    amplitude_original = y_min_plot + (y_fraction * y_range_plot)
    
    # If timestamp_values array is provided, find the closest actual timestamp
    if timestamp_values is not None:
        # Find the index in the original data
        data_index = int(x_fraction * len(timestamp_values))
        data_index = min(data_index, len(timestamp_values) - 1)  # Clamp to valid range
        timestamp_original = timestamp_values[data_index]
    
    return {
        'timestamp': timestamp_original,
        'amplitude': amplitude_original,
        'bbox_pixel': {
            'x_center': x_center_px,
            'y_center': y_center_px,
            'width': width_px,
            'height': height_px
        },
        'x_fraction': x_fraction,
        'data_index': int(x_fraction * metadata['num_datapoints'])
    }


def load_metadata(image_path):
    """
    Load metadata from JSON file
    """
    metadata_path = image_path.replace('.png', '_metadata.json')
    with open(metadata_path, 'r') as f:
        return json.load(f)