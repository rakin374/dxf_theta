import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from google.colab import files
from sklearn.cluster import KMeans
from scipy.ndimage import label, find_objects, binary_erosion

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def display_image(image_array, title="Image", size=(6,6)):
    """
    Display a given image array (NumPy array) with a title.
    For images with transparency, a white background is used.
    """
    if image_array.shape[-1] == 4:
        rgb = image_array[..., :3].copy()
        alpha = image_array[..., 3]
        rgb[alpha==0] = [255, 255, 255]
        img_to_show = rgb
    else:
        img_to_show = image_array
    plt.figure(figsize=size)
    plt.imshow(img_to_show)
    plt.title(title)
    plt.axis('off')
    plt.show()

def get_unique_colors(image):
    """
    Return a list of unique RGB colors in the input image.
    If the image is already flattened (i.e. shape (N,3)), it returns unique rows.
    """
    if image.ndim == 2:
        arr = image
    else:
        arr = image.reshape(-1, image.shape[2])
    return np.unique(arr, axis=0)

def display_palette(palette, title="Palette"):
    """
    Displays a list of RGB colors as swatches in a single row.
    """
    swatch_height = 50
    swatch_width = 50
    n_colors = len(palette)
    swatches = np.zeros((swatch_height, swatch_width * n_colors, 3), dtype=np.uint8)
    for i, color in enumerate(palette):
        swatches[:, i*swatch_width:(i+1)*swatch_width, :] = np.array(color, dtype=np.uint8)
    plt.figure(figsize=(n_colors, 1.5))
    plt.imshow(swatches)
    plt.title(title)
    plt.axis('off')
    plt.show()

def compute_circumference(mask):
    """
    Compute the perimeter of a connected component using binary erosion.
    """
    eroded = binary_erosion(mask, structure=np.ones((3,3)))
    edge_mask = mask & ~eroded
    return np.sum(edge_mask)

def get_component_image(element, has_alpha=False):
    """
    Given an element (with keys "bbox", "mask", "color"), create an image of its bounding box
    with the component painted in its color.
    """
    y_min, y_max, x_min, x_max = element["bbox"]
    height = y_max - y_min
    width = x_max - x_min
    if has_alpha:
        comp_img = np.dstack((
            255*np.ones((height, width), dtype=np.uint8),
            255*np.ones((height, width), dtype=np.uint8),
            255*np.ones((height, width), dtype=np.uint8),
            np.zeros((height, width), dtype=np.uint8)
        ))
        comp_img[element["mask"]] = list(element["color"]) + [255]
    else:
        comp_img = 255 * np.ones((height, width, 3), dtype=np.uint8)
        comp_img[element["mask"]] = element["color"]
    return comp_img

def display_components_grid(layers_dict, has_alpha=False):
    """
    Display a grid where each row corresponds to a layer and each cell shows the component image.
    """
    layer_numbers = sorted(layers_dict.keys())
    max_cols = max(len(layers_dict[layer]) for layer in layer_numbers)
    
    fig, axes = plt.subplots(nrows=len(layer_numbers), ncols=max_cols, figsize=(max_cols*3, len(layer_numbers)*3))
    if len(layer_numbers) == 1:
        axes = np.expand_dims(axes, axis=0)
    if max_cols == 1:
        axes = np.expand_dims(axes, axis=1)
    
    for row_idx, layer in enumerate(layer_numbers):
        components = layers_dict[layer]
        for col_idx in range(max_cols):
            ax = axes[row_idx, col_idx]
            if col_idx < len(components):
                comp_img = get_component_image(components[col_idx], has_alpha=has_alpha)
                ax.imshow(comp_img)
                ax.set_title(f"ID {components[col_idx]['id']}")
            else:
                ax.axis('off')
            if col_idx == 0:
                ax.set_ylabel(f"Layer {layer}", rotation=0, size='large', labelpad=40)
            ax.axis('off')
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# Step 1: File Upload (with option to reuse previous upload)
# -----------------------------------------------------------------------------
if 'uploaded' in globals() and uploaded:
    use_previous = input("A file has already been uploaded. Do you want to use it? (Y/n): ").strip().lower()
    if use_previous in ['n', 'no']:
        uploaded = files.upload()
else:
    print("Please upload your image file.")
    uploaded = files.upload()

filename = list(uploaded.keys())[0]
img = Image.open(filename)

# Check if the image has an alpha channel (transparent background)
has_alpha = False
if img.mode in ('RGBA', 'LA'):
    has_alpha = True

if has_alpha:
    img_np = np.array(img)
    rgb_img = img_np[..., :3]
    alpha_channel = img_np[..., 3]
    fg_mask = alpha_channel > 0
    display_img = rgb_img.copy()
    display_img[~fg_mask] = [255, 255, 255]
else:
    img_np = np.array(img.convert('RGB'))
    rgb_img = img_np
    fg_mask = np.ones(rgb_img.shape[:2], dtype=bool)
    display_img = rgb_img

display_image(display_img, title="Original Image")

# -----------------------------------------------------------------------------
# Step 2: Count Unique Colors (Foreground Only)
# -----------------------------------------------------------------------------
unique_colors = get_unique_colors(rgb_img[fg_mask].reshape(-1, 3))
print(f"The original image has {len(unique_colors)} unique colors (in the foreground).")

# -----------------------------------------------------------------------------
# Step 3: Determine Required Colors for p% Representation & Produce Elbow Plot
# -----------------------------------------------------------------------------
p = 90  # percentage of variance to be explained
pixels = rgb_img[fg_mask].reshape(-1, 3)
mean_pixel = np.mean(pixels, axis=0)
total_variance = np.sum((pixels - mean_pixel)**2)
max_clusters = min(20, len(unique_colors))
inertias = []
explained_variances = []
print("Calculating K-Means for different numbers of clusters...")
for k in range(1, max_clusters+1):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(pixels)
    inertia = kmeans.inertia_
    ev = 1 - inertia/total_variance
    inertias.append(inertia)
    explained_variances.append(ev)
k_needed = next(k for k, ev in enumerate(explained_variances, start=1) if ev >= p/100)
print(f"To represent {p}% variance, you need at least {k_needed} colors.")

plt.figure(figsize=(8,6))
plt.plot(range(1, max_clusters+1), inertias, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Plot')
plt.axvline(k_needed, color='red', linestyle='--', label=f'k_needed = {k_needed}')
plt.legend()
plt.show()

plt.figure(figsize=(8,6))
plt.plot(range(1, max_clusters+1), [ev*100 for ev in explained_variances], marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Explained Variance (%)')
plt.title('Explained Variance')
plt.axhline(p, color='green', linestyle='--', label=f'{p}%')
plt.axvline(k_needed, color='red', linestyle='--', label=f'k_needed = {k_needed}')
plt.legend()
plt.show()

# -----------------------------------------------------------------------------
# Step 4: Input Target Number of Colors and K-Means Color Reduction (Foreground)
# -----------------------------------------------------------------------------
target_n = int(input("Enter target number of colors (or use the suggested k_needed): "))
if target_n < 1 or target_n > len(unique_colors):
    print("Invalid number of target colors.")
    raise SystemExit

print("Clustering foreground pixels...")
kmeans = KMeans(n_clusters=target_n, random_state=42, n_init='auto')
kmeans.fit(pixels)
cluster_centers = np.round(kmeans.cluster_centers_).astype(np.uint8)
labels = kmeans.labels_
new_rgb_img = rgb_img.copy()
foreground_indices = np.where(fg_mask)
new_rgb_img[foreground_indices] = cluster_centers[labels]

if has_alpha:
    new_img_np = np.dstack((new_rgb_img, alpha_channel))
    display_new = new_rgb_img.copy()
    display_new[~fg_mask] = [255, 255, 255]
else:
    new_img_np = new_rgb_img
    display_new = new_img_np

display_image(display_new, title=f"Reconstructed with {target_n} Colors")
palette = [tuple(c) for c in cluster_centers]
display_palette(palette, title="Reduced Color Palette")

# -----------------------------------------------------------------------------
# Step 5: Extract Connected Components (Foreground Only)
# -----------------------------------------------------------------------------
elements_by_color = []
all_elements = []
element_id = 0
if has_alpha:
    image_for_cc = new_img_np[..., :3]
else:
    image_for_cc = new_img_np

for color in palette:
    mask = np.all(image_for_cc == color, axis=-1)
    mask = mask & fg_mask  # only foreground pixels
    total_pixels = np.sum(mask)
    print(f"\nColor {color} -> {total_pixels} pixels")
    labeled_array, num_features = label(mask, structure=np.ones((3,3), dtype=int))
    print(f"  Found {num_features} connected component(s).")
    slices = find_objects(labeled_array)
    color_elements = []
    for comp_index, sl in enumerate(slices, start=1):
        comp_mask = (labeled_array[sl] == comp_index)
        comp_size = np.sum(comp_mask)
        circumference = compute_circumference(comp_mask)
        bbox = (sl[0].start, sl[0].stop, sl[1].start, sl[1].stop)
        elem = {
            "id": element_id,
            "color": color,
            "area": comp_size,
            "circumference": circumference,
            "bbox": bbox,
            "mask": comp_mask,
            "enveloped_by": None
        }
        color_elements.append(elem)
        all_elements.append(elem)
        print(f"    Element {comp_index} (ID={element_id}): area={comp_size}, circ={circumference}")
        element_id += 1
    elements_by_color.append(color_elements)

# -----------------------------------------------------------------------------
# Step 6: Establish Envelope Relationships
# -----------------------------------------------------------------------------
for elem in all_elements:
    y_min, y_max, x_min, x_max = elem["bbox"]
    candidate_parents = []
    for other in all_elements:
        if other["id"] == elem["id"]:
            continue
        oy_min, oy_max, ox_min, ox_max = other["bbox"]
        if (oy_min <= y_min < y_max <= oy_max) and (ox_min <= x_min < x_max <= ox_max):
            candidate_parents.append(other)
    if candidate_parents:
        parent = min(candidate_parents, key=lambda f: f["area"])
        elem["enveloped_by"] = parent["id"]

# -----------------------------------------------------------------------------
# Step 7: Compute Complexity and Assign Layers
# Smaller numerical layers (before inversion) indicate front.
# -----------------------------------------------------------------------------
for e in all_elements:
    e["complexity"] = e["circumference"] / e["area"] if e["area"] > 0 else float('inf')

max_circ_elem = max(all_elements, key=lambda e: e["circumference"])
max_circ_elem["layer"] = 1

non_enveloped = [x for x in all_elements if x["enveloped_by"] is None and x["id"] != max_circ_elem["id"]]
non_enveloped_sorted = sorted(non_enveloped, key=lambda x: x["complexity"])
layer_counter = 2
for e in non_enveloped_sorted:
    e["layer"] = layer_counter
    layer_counter += 1

for e in all_elements:
    if e["enveloped_by"] is not None:
        parent = next((p for p in all_elements if p["id"] == e["enveloped_by"]), None)
        if parent and "layer" in parent:
            e["layer"] = parent["layer"] + 1
        else:
            e["layer"] = layer_counter

print("\nOriginal layer assignment (smaller = front):")
for e in sorted(all_elements, key=lambda x: x["layer"]):
    print(f"ID {e['id']}: layer={e['layer']}, color={e['color']}, area={e['area']}")

# Invert layer numbering so that numerically smaller layers become the back.
max_layer = max(e["layer"] for e in all_elements)
for e in all_elements:
    e["layer"] = max_layer + 1 - e["layer"]

# Build a dictionary grouping elements by layer.
layers_dict = {}
for e in all_elements:
    layers_dict.setdefault(e["layer"], []).append(e)
all_layer_numbers = sorted(layers_dict.keys())

print("\nRevised layer assignment (smaller = back):")
for e in sorted(all_elements, key=lambda x: x["layer"]):
    print(f"ID {e['id']}: layer={e['layer']}, color={e['color']}, area={e['area']}")

# -----------------------------------------------------------------------------
# Step 8: Display Component Grid (Grouped by Layer)
# -----------------------------------------------------------------------------
display_components_grid(layers_dict, has_alpha=has_alpha)

# -----------------------------------------------------------------------------
# Step 9: Build Tree Data Structure from Envelope Relationships
# -----------------------------------------------------------------------------
class TreeNode:
    def __init__(self, element):
        self.element = element
        self.children = []

# Create a node for each element.
nodes = {e["id"]: TreeNode(e) for e in all_elements}
roots = []
for e in all_elements:
    if e["enveloped_by"] is not None:
        parent_id = e["enveloped_by"]
        nodes[parent_id].children.append(nodes[e["id"]])
    else:
        roots.append(nodes[e["id"]])

# -----------------------------------------------------------------------------
# Step 10: Visualize the Tree Structure with Thumbnails
# -----------------------------------------------------------------------------
def layout_tree(node, depth=0, next_x=[0], positions=None):
    """
    Recursively assign (x, y) positions to nodes.
    Leaves get the next available x value; internal nodes get the average of their children's x.
    """
    if positions is None:
        positions = {}
    if not node.children:
        x = next_x[0]
        positions[node] = (x, depth)
        next_x[0] += 1
    else:
        for child in node.children:
            layout_tree(child, depth+1, next_x, positions)
        child_xs = [positions[child][0] for child in node.children]
        x = sum(child_xs) / len(child_xs)
        positions[node] = (x, depth)
    return positions

def visualize_tree_forest(roots, node_width=50, node_height=50, h_spacing=50, v_spacing=100):
    """
    Visualize a forest of trees (if more than one root) by computing positions for each node,
    then drawing edges and placing a thumbnail image at each node.
    """
    all_positions = {}
    offset = 0
    for root in roots:
        positions = layout_tree(root, depth=0, next_x=[0])
        for node, (x, y) in positions.items():
            all_positions[node] = (x + offset, y)
        tree_width = max(x for (x, y) in positions.values()) + 1
        offset += tree_width + 1  # add gap between trees
    
    # Scale positions to pixel coordinates.
    scaled_positions = {node: (x*(node_width+h_spacing), y*(node_height+v_spacing))
                        for node, (x, y) in all_positions.items()}
    max_x = max(x for (x, y) in scaled_positions.values()) + node_width
    max_y = max(y for (x, y) in scaled_positions.values()) + node_height
    
    fig, ax = plt.subplots(figsize=(max_x/100, max_y/100))
    ax.set_xlim(0, max_x)
    ax.set_ylim(0, max_y)
    ax.invert_yaxis()
    ax.axis('off')
    
    # Draw edges.
    for node, (x, y) in scaled_positions.items():
        for child in node.children:
            child_x, child_y = scaled_positions[child]
            ax.plot([x+node_width/2, child_x+node_width/2],
                    [y+node_height, child_y], 'k-', lw=1)
    
    # Draw nodes as thumbnails.
    for node, (x, y) in scaled_positions.items():
        comp_img = get_component_image(node.element, has_alpha=has_alpha)
        thumb = Image.fromarray(comp_img).resize((node_width, node_height))
        thumb = np.array(thumb)
        extent = [x, x+node_width, y, y+node_height]
        ax.imshow(thumb, extent=extent)
        ax.text(x+node_width/2, y-5, f"ID {node.element['id']}",
                ha='center', va='top', fontsize=8)
    plt.show()

visualize_tree_forest(roots)
