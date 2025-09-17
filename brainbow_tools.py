import numpy as np 
import os
from skimage import io, measure, morphology, filters, exposure
import napari
import pandas as pd
from openpyxl import load_workbook
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
from skimage.segmentation import find_boundaries
from sklearn.cluster import Birch
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def rgb_to_maxwell_triangle(r, g, b):
    """
    Convert RGB values to Maxwell triangle coordinates.
    Handles both scalar and array inputs correctly.
    """
    total = r + g + b
    
    # Handle both scalar and array cases
    if np.isscalar(total):
        if total == 0:
            return 0.0, 0.0
    else:
        # For arrays, create masks for zero totals
        zero_mask = total == 0
        non_zero_mask = ~zero_mask
    
    # Initialize normalized values
    r_norm = np.zeros_like(r, dtype=float)
    g_norm = np.zeros_like(g, dtype=float)
    b_norm = np.zeros_like(b, dtype=float)
    
    # Calculate normalized values only for non-zero totals
    if np.isscalar(total):
        r_norm = r / total
        g_norm = g / total
        b_norm = b / total
    else:
        r_norm[non_zero_mask] = r[non_zero_mask] / total[non_zero_mask]
        g_norm[non_zero_mask] = g[non_zero_mask] / total[non_zero_mask]
        b_norm[non_zero_mask] = b[non_zero_mask] / total[non_zero_mask]
    
    # Convert to Maxwell triangle coordinates
    x = 0.5 * (2 * g_norm + r_norm)
    y = (np.sqrt(3) / 2) * r_norm
    
    return x, y

def create_maxwell_triangle(points, point_alpha=1, point_size=5, unmixing_triangle=None, ax=None):
    """
    Create a Maxwell triangle plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    if unmixing_triangle is not None:
        # Plot unmixing triangle
        triangle_vertices = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])
        transformed_vertices = np.dot(unmixing_triangle, triangle_vertices.T).T
        triangle = patches.Polygon(transformed_vertices, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(triangle)
    
    # Plot points
    ax.scatter(points[:, 0], points[:, 1], alpha=point_alpha, s=point_size)
    
    # Set triangle boundaries
    ax.plot([0, 0.5], [0, np.sqrt(3)/2], 'k-')
    ax.plot([0.5, 1], [np.sqrt(3)/2, 0], 'k-')
    ax.plot([1, 0], [0, 0], 'k-')
    
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 0.95)
    ax.set_aspect('equal')
    ax.axis('off')
    
    return ax

def unmix_rgb(rgb_values):
    """
    Unmix RGB values using matrix inversion.
    """
    # Create ideal color matrix (RGB for pure colors)
    ideal_colors = np.array([
        [1, 0, 0],  # Red
        [0, 1, 0],  # Green
        [0, 0, 1]   # Blue
    ])
    
    # Calculate unmixing matrix
    unmixing_matrix = np.linalg.inv(ideal_colors)
    
    # Apply unmixing
    unmixed = np.dot(unmixing_matrix, rgb_values.T).T
    
    # Ensure no negative values
    unmixed[unmixed < 0] = 0
    
    return unmixed, unmixing_matrix

def normalisation_canal_per_quantile(data, lower_percentages, upper_percentages, channel_axis=1, clip=True, return_quantiles=False, skew_norm=None):
    """
    Normalize data by channel using quantiles.
    """
    if skew_norm is None:
        skew_norm = [0, 0]
    
    normalized_data = np.zeros_like(data)
    quantiles = []
    
    for i in range(data.shape[channel_axis]):
        channel_data = data[:, i] if channel_axis == 1 else data[i, :]
        
        # Calculate quantiles
        lower_q = np.percentile(channel_data, lower_percentages)
        upper_q = np.percentile(channel_data, upper_percentages[i] if hasattr(upper_percentages, '__iter__') else upper_percentages)
        
        quantiles.append((lower_q, upper_q))
        
        # Normalize channel
        channel_norm = (channel_data - lower_q) / (upper_q - lower_q)
        
        if clip:
            channel_norm = np.clip(channel_norm, 0, 1)
        
        if channel_axis == 1:
            normalized_data[:, i] = channel_norm
        else:
            normalized_data[i, :] = channel_norm
    
    if return_quantiles:
        return normalized_data, quantiles
    return normalized_data

def extract_data_image(brainbow_path, label_mask_path, output_csv_path):
    """
    Processes a Brainbow image to detect labeled cells and save cell data to a CSV file.
    
    Parameters:
    - brainbow_path (str): Path to the Brainbow image (e.g., TIFF file).
    - label_mask_path (str): Path to the label mask image (e.g., TIFF file).
    - output_csv_path (str): Path to save the output CSV file containing cell properties.
    
    Returns:
    - pd.DataFrame: DataFrame containing properties of detected cells.
    """
    
    # Load the Brainbow image
    brainbow = io.imread(brainbow_path)

    # Load the label mask that we'll create with proper labels
    label_mask = io.imread(label_mask_path)

    # Convert the label mask into a binary mask for cell regions
    binary_mask = (label_mask == 0)  # True where cells are (black in our visual mask)
    labeled_cells = measure.label(binary_mask)

    # Measure properties for each labeled region
    props = measure.regionprops_table(labeled_cells, brainbow,   
                                     properties=('label', 'centroid', 'area', 
                                                'mean_intensity'))
    # Create a DataFrame to store the properties of each detected cell
    df = pd.DataFrame(props) 

    print(f"Number of ROIs detected: {len(df)}")
    
    # Save the DataFrame to the provided output CSV path
    df.to_csv(output_csv_path, index=False)
    print(f"Results saved to: {output_csv_path}")
    
    return df

def convert_area_to_microns(input_csv_path, output_csv_path):
    """
    Converts the 'area' column in a CSV file from pixel^2 to microns^2.
    
    Parameters:
    - input_csv_path (str): Path to the input CSV file with cell data.
    - output_csv_path (str): Path to save the output CSV file with the converted area values.
    
    Returns:
    - pd.DataFrame: DataFrame with the 'area' column converted to microns^2.
    """
    
    # Load the CSV file containing cell data
    df = pd.read_csv(input_csv_path)

    # Conversion factor: 1 px² = 0.010609 µm² (since 1 px = 0.103 µm)
    pixel_area_um2 = 0.103 ** 2

    # Replace 'area' column with values in µm²
    df['area'] = df['area'] * pixel_area_um2

    # Save the updated DataFrame to the specified output CSV path
    df.to_csv(output_csv_path, index=False)

    print(f"Done! 'area' column now in µm². File saved as '{output_csv_path}'")

    return df

def plot_cell_size_distribution(df, area_col='area', bin_edges=None, title='Cell Size Distribution', base_name=None, pdf=None):
    """
    Plots a histogram (as percentages) of cell area distribution from a DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing cell measurement data.
    area_col : str
        Name of the column containing cell area values in µm².
    bin_edges : list of float or None
        Edges for histogram bins. If None, uses default bins.
    title : str
        Title for the plot.
    """
    if bin_edges is None:
        bin_edges = [0, 1, 5, 10, 20, 50, 100, 250, 500]

    # Compute histogram and convert to percentages
    areas = df[area_col]
    hist_counts, _ = np.histogram(areas, bins=bin_edges)
    hist_percentages = (hist_counts / len(df)) * 100

    # Create human-readable bin labels
    bin_labels = []
    for i in range(len(bin_edges) - 1):
        start = bin_edges[i]
        end = bin_edges[i + 1]
        label = f'{start}–{end} µm²' if end != bin_edges[-1] else f'>{start} µm²'
        bin_labels.append(label)

    plot_title = title
    if base_name:
        plot_title = f"{title}\n({base_name})"

    # Plot
    fig=plt.figure(figsize=(8, 5))
    plt.bar(bin_labels, hist_percentages, color='black', edgecolor='black')
    plt.xlabel('Cell Area (µm²)', fontsize=14)
    plt.ylabel('Percentage of Cells (%)', fontsize=14)
    plt.title(plot_title, fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    if pdf:
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
        
def plot_clone_size_distribution_per_color(massive_df, color, output_folder, base_name=None):
    """
    Plots the distribution of clone sizes (number of cells per clone) for a specific color.
    
    Parameters:
    - massive_df (pd.DataFrame): DataFrame containing massive cell data for one color.
    - color (str): The color category being plotted.
    - output_folder (str): Folder to save the plot.
    - base_name (str): Base name for the plot (image identifier).
    
    Returns:
    - str: Path to the saved plot.
    """
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    from collections import Counter

    if massive_df is None or massive_df.empty:
        print(f"No massive cells data for {color} to plot clone size distribution.")
        return None

    # Extract the clone sizes (number of individual cells per clone)
    clone_sizes = massive_df['individual_cells_count'].values

    if len(clone_sizes) == 0:
        print(f"No clone sizes found for {color}.")
        return None

    # Define size bins
    bin_edges = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 50, 1000]
    bin_labels = [
        '1', '2', '3', '4', '5', '6', '7', '8', '9', 
        '10-14', '15-19', '20-29', '30-49', '50+'
    ] 

    # Compute histogram and convert to percentages
    hist_counts, _ = np.histogram(clone_sizes, bins=bin_edges)
    hist_percentages = (hist_counts / len(clone_sizes)) * 100

    # Create plot
    plt.figure(figsize=(10, 6))
    plt.bar(bin_labels, hist_percentages, color=color.lower(), alpha=0.7, edgecolor='black')

    # Set title and labels
    title = f'Clone Size Distribution - {color}'
    if base_name:
        title += f' ({base_name})'
    plt.title(title, fontsize=14)
    plt.xlabel('Cells per Clone', fontsize=12)
    plt.ylabel('Percentage of Clones (%)', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)

    # Add text labels to bars
    for i, val in enumerate(hist_percentages):
        if val > 0:  # Only label bars with values
            plt.text(i, val + 0.5, f'{val:.1f}%', ha='center', fontsize=9)

    # Save plot
    plot_filename = f"{base_name}_{color}_clone_size_distribution.png" if base_name else f"{color}_clone_size_distribution.png"
    plot_path = os.path.join(output_folder, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved clone size distribution for {color}: {plot_path}")
    return plot_path

def plot_percentage_cells_in_clone_bins_per_color(regionprops_excel_path, output_folder, base_name=None):
    """
    Plots the percentage of cells that fall into each clone size bin for each color separately.
    
    Parameters:
    - regionprops_excel_path (str): Path to the regionprops Excel file
    - output_folder (str): Folder to save the plot
    - base_name (str): Base name for the plot title and filename
    
    Returns:
    - str: Path to the saved plot
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from collections import defaultdict

    # Define size bins
    bin_edges = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 50, 1000]
    bin_labels = [
        '1', '2', '3', '4', '5', '6', '7', '8', '9', 
        '10-14', '15-19', '20-29', '30-49', '50+'
    ]

    # Define color categories and their plotting colors
    color_categories = ['Red', 'Green', 'Blue', 'Yellow', 'Magenta', 'Cyan', 'White', 'Black']
    color_map = {
        'Red': 'red',
        'Green': 'green',
        'Blue': 'blue',
        'Yellow': 'gold',
        'Magenta': 'magenta',
        'Cyan': 'cyan',
        'White': 'lightgray',
        'Black': 'black'
    }

    # Dictionary to store cell counts per bin for each color
    color_bin_data = {color: defaultdict(int) for color in color_categories}
    color_total_cells = {color: 0 for color in color_categories}

    # Read data from Excel
    with pd.ExcelFile(regionprops_excel_path) as xls:
        for color in color_categories:
            sheet_name = f"{color}_Massive_Cells"
            if sheet_name in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name)
                if not df.empty and 'individual_cells_count' in df.columns:
                    # For each clone, add its cell count to the appropriate bin
                    for _, row in df.iterrows():
                        clone_size = row['individual_cells_count']
                        color_total_cells[color] += clone_size
                        
                        # Find the appropriate bin
                        for i in range(len(bin_edges) - 1):
                            if bin_edges[i] <= clone_size < bin_edges[i+1]:
                                color_bin_data[color][bin_labels[i]] += clone_size
                                break
                        else:
                            # Handle clones larger than the last bin edge
                            color_bin_data[color][bin_labels[-1]] += clone_size

    # Calculate percentages for each color
    color_percentages = {}
    for color in color_categories:
        if color_total_cells[color] > 0:
            color_percentages[color] = {}
            for bin_label in bin_labels:
                if bin_label in color_bin_data[color]:
                    color_percentages[color][bin_label] = (color_bin_data[color][bin_label] / color_total_cells[color]) * 100
                else:
                    color_percentages[color][bin_label] = 0

    # Create plot with subplots for each color
    fig, axes = plt.subplots(2, 4, figsize=(20, 12))
    axes = axes.flatten()
    
    for i, color in enumerate(color_categories):
        ax = axes[i]
        
        if color in color_percentages and color_total_cells[color] > 0:
            # Get percentages for this color
            percentages = [color_percentages[color][label] for label in bin_labels]
            
            # Plot
            x_pos = np.arange(len(bin_labels))
            ax.bar(x_pos, percentages, color=color_map[color], edgecolor='black', alpha=0.7)
            
            # Format subplot
            ax.set_title(f'{color} Cells\n(Total: {color_total_cells[color]} cells)', fontsize=12)
            ax.set_xlabel('Clone Size (cells per clone)', fontsize=10)
            ax.set_ylabel('Percentage of Cells (%)', fontsize=10)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(bin_labels, rotation=45, fontsize=8)
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for j, val in enumerate(percentages):
                if val > 1:  # Only label significant values
                    ax.text(j, val + 0.5, f'{val:.1f}%', ha='center', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{color} Cells', fontsize=12)
    
    # Set main title
    main_title = 'Percentage of Cells in Clone Size Bins by Color'
    if base_name:
        main_title += f'\n({base_name})'
    fig.suptitle(main_title, fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save plot
    plot_filename = f"percentage_cells_in_clone_bins_per_color_{base_name}.png" if base_name else "percentage_cells_in_clone_bins_per_color.png"
    plot_path = os.path.join(output_folder, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved percentage cells in clone bins per color plot: {plot_path}")
    return plot_path

def process_brainbow_image(brainbow_path, label_mask_path, output_csv_path, base_name=None, pdf=None, excel_path=None):
    # Load the Brainbow image and label mask
    brainbow = io.imread(brainbow_path)
    label_mask = io.imread(label_mask_path)

    # Create a binary mask (True where cell, False otherwise)
    binary_mask = (label_mask == 0)
    
    # Label the binary mask for regionprops extraction
    labeled_cells = measure.label(binary_mask)

    # Extract region properties from labeled cells and the Brainbow image
    df = pd.read_csv(output_csv_path)  # Assuming this CSV contains previously extracted data

    # Extract pixel intensities for each cell
    red_intensities = df['mean_intensity-0'].values
    green_intensities = df['mean_intensity-1'].values
    blue_intensities = df['mean_intensity-2'].values
    
    # Stack intensities into a single array
    colors = np.vstack([red_intensities, green_intensities, blue_intensities]).T
    
    # NEW: Maxwell triangle-based color classification
    print("Starting Maxwell triangle-based color classification...")
    
    # 1. Log transform and handle invalid values
    ll = np.log(colors)
    ll[np.isnan(ll)] = 0
    ll[np.isinf(ll)] = 0
    ll[ll < 0] = 0

    # 2. Compute histogram for density estimation
    hist, edges = np.histogramdd(ll, bins=101)

    # 3. Get bin indices for each dimension
    bin_indices = np.vstack([
        np.searchsorted(edges[dim], ll[:, dim], side='right') - 1
        for dim in range(ll.shape[1])
    ]).T

    # 4. Ensure indices are within valid range
    for dim in range(bin_indices.shape[1]):
        bin_indices[:, dim] = np.clip(bin_indices[:, dim], 0, hist.shape[dim] - 1)

    # 5. Retrieve densities using the bin indices
    densities = hist[tuple(bin_indices.T)]
    
    # 6. Select cells with sufficient density (remove outliers)
    min_density = 0.1  # Adjust as needed
    selected_cells = densities > min_density
    
    # 7. Normalize selected cells
    selected_colors = colors[selected_cells]
    selected_cells_normed = selected_colors / np.sum(selected_colors, axis=1)[:, None]
    selected_cells_normed[np.sum(selected_colors, axis=1) == 0] = 0

    # 8. Unmix RGB values
    unmixed_colors, unmixing_matrix = unmix_rgb(selected_colors)
    
    # 9. Normalize unmixed colors by quantile
    lower_percentages = 0
    upper_percentages = [99., 99.7, 99.7]
    skew_norm = [0, 0]
    
    colors_unmixed_quantile, quantiles = normalisation_canal_per_quantile(
        unmixed_colors,
        lower_percentages,
        upper_percentages,
        channel_axis=1,
        clip=True,
        return_quantiles=True,
        skew_norm=skew_norm
    )
    
    # 10. Remove points on the border of the triangle
    colors_mask = colors_unmixed_quantile[np.min(colors_unmixed_quantile, axis=-1) > 0]
    
    # 11. Convert to Maxwell triangle coordinates
    xy_coords = rgb_to_maxwell_triangle(*colors_mask.T)
    xy_coords = np.column_stack(xy_coords)
    
    # 12. Cluster using Birch
    birch = Birch(n_clusters=7, branching_factor=10, threshold=0.05)
    birch.fit(xy_coords)
    
    # 13. Predict clusters for all points
    birch_predict = birch.predict(xy_coords)
    
    # 14. Map clusters to colors based on their position in the Maxwell triangle
    # Define expected positions of pure colors in Maxwell triangle
    color_positions = {
        'Red': (0.5, np.sqrt(3)/2),
        'Green': (1, 0),
        'Blue': (0, 0),
        'Yellow': (0.75, np.sqrt(3)/4),
        'Magenta': (0.25, np.sqrt(3)/4),
        'Cyan': (0.5, 0),
        'White': (0.5, np.sqrt(3)/6)
        }

    # Calculate cluster centroids
    cluster_centroids = {}
    for cluster_id in range(7):
        cluster_points = xy_coords[birch_predict == cluster_id]
        if len(cluster_points) > 0:
            cluster_centroids[cluster_id] = np.mean(cluster_points, axis=0)

    # Assign colors based on closest expected position
    cluster_colors = {}
    for cluster_id, centroid in cluster_centroids.items():
        min_dist = float('inf')
        assigned_color = 'Black'
        for color, pos in color_positions.items():
            dist = np.linalg.norm(centroid - pos)
            if dist < min_dist:
                min_dist = dist
                assigned_color = color
        cluster_colors[cluster_id] = assigned_color
    
    # 15. Assign colors to selected cells
    color_assignments = np.full(len(colors), 'Black')  # Default to black
    # Create a new mask that combines both selection criteria
    final_selection_mask = selected_cells.copy()
    final_selection_mask[selected_cells] = (np.min(colors_unmixed_quantile, axis=-1) > 0)

    # Assign colors only to cells that passed both filters
    color_assignments[final_selection_mask] = [cluster_colors.get(cluster, 'Black') for cluster in birch_predict]
    
    # 16. Update DataFrame with color assignments
    df['Color'] = color_assignments
    
    # Plot the Maxwell triangle with clustering results
    fig = plt.figure(figsize=(10, 8))
    ax = create_maxwell_triangle(
        xy_coords[:int(len(xy_coords)*0.1)],  # Display only 10% of points for clarity
        point_alpha=0.5,
        point_size=5,
        ax=plt.gca()
    )
    
    # Color points by cluster
    for cluster_id in range(7):
        cluster_points = xy_coords[birch_predict == cluster_id]
        if len(cluster_points) > 0:
            ax.scatter(
                cluster_points[:int(len(cluster_points)*0.1), 0],
                cluster_points[:int(len(cluster_points)*0.1), 1],
                alpha=0.5,
                s=5,
                label=cluster_colors.get(cluster_id, f'Cluster {cluster_id}'),
                color=cluster_colors.get(cluster_id, 'black').lower()
                )
    
    ax.legend()
    plt.title(f'Maxwell Triangle with Clustering - {base_name}' if base_name else 'Maxwell Triangle with Clustering')
    
    if pdf:
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
    
    # Continue with the rest of the processing (masks, regionprops, etc.)
    image_base = os.path.splitext(os.path.basename(brainbow_path))[0]
    color_masks_folder = os.path.join(os.path.dirname(brainbow_path), f"{image_base}_color_masks")
    os.makedirs(color_masks_folder, exist_ok=True)
    print(f"Created color masks folder: {color_masks_folder}")

    # Generate and save masks for ALL color categories
    color_masks = {}
    color_categories = ['Red', 'Green', 'Blue', 'Yellow', 'Magenta', 'Cyan', 'White', 'Black']
    
    # Create a dilation kernel (1-pixel disk)
    dilation_kernel = morphology.disk(1)
    
    # Define boundary colors for massive cells
    boundary_colors = {
        'Red': [255, 0, 0],        # Red
        'Green': [0, 255, 0],      # Green
        'Blue': [0, 0, 255],       # Blue
        'Yellow': [255, 255, 0],   # Yellow
        'Magenta': [255, 0, 255],  # Magenta
        'Cyan': [0, 255, 255],     # Cyan
        'White': [255, 255, 255],  # White
        'Black': [0, 0, 0]         # Black
    }
    
    for color in color_categories:
        # Create binary mask for this color
        color_mask = np.zeros_like(label_mask, dtype=np.uint16)
        color_cells = df[df['Color'] == color]['label']
        
        for label in color_cells:
            color_mask[labeled_cells == label] = 1
        
        # Dilate the mask to merge adjacent cells
        dilated_mask = morphology.binary_dilation(color_mask, dilation_kernel).astype(np.uint16)
        
        # Label the dilated regions to identify massive cells
        labeled_mask, num_massive_cells = measure.label(dilated_mask, return_num=True)
        
        # Find boundaries of massive cells
        massive_boundaries = find_boundaries(labeled_mask, mode='outer')
        
        # Create RGB version of the mask with colored boundaries
        rgb_mask = np.zeros((*color_mask.shape, 3), dtype=np.uint8)
        
        # Fill the cells with white
        rgb_mask[color_mask.astype(bool)] = [255, 255, 255]
        
        # Add colored boundaries for massive cells
        boundary_color = boundary_colors[color]
        rgb_mask[massive_boundaries] = boundary_color
        
        # Save the RGB mask with boundaries
        color_mask_path = os.path.join(color_masks_folder, f"{color.lower()}_mask_with_boundaries.tif")
        io.imsave(color_mask_path, rgb_mask)
        
        color_masks[color] = color_mask_path
        print(f"Saved {color} mask with boundaries: {os.path.basename(color_mask_path)}")
        print(f"Number of {color} cells detected: {len(color_cells)}")
        print(f"Number of {color} clones after merging: {num_massive_cells}")

    regionprops_excel_path = analyze_color_masks_regionprops(
        brainbow_path, labeled_cells, df, color_masks_folder, brainbow
    )
    
    if regionprops_excel_path:
        print(f"Regionprops analysis completed: {os.path.basename(regionprops_excel_path)}")
    
    # Generate and save massive cell distribution plot
    massive_dist_plot_path = plot_massive_cell_distribution(
        regionprops_excel_path,
        color_masks_folder,
        base_name=image_base
    )
    
    # Plot clone size distribution per color
    if regionprops_excel_path:
        # Read the Excel file
        xls = pd.ExcelFile(regionprops_excel_path)
        for color in color_categories:
            sheet_name = f"{color}_Massive_Cells"
            if sheet_name in xls.sheet_names:
                df_color = pd.read_excel(xls, sheet_name)
                if not df_color.empty and 'individual_cells_count' in df_color.columns:
                    # Plot clone size distribution for this color
                    _ = plot_clone_size_distribution_per_color(
                        df_color, color, color_masks_folder, base_name=image_base
                        )
    
    # Save the color classified DataFrame to CSV
    df.to_csv(output_csv_path, index=False)
    
    if regionprops_excel_path:
        print(f"Regionprops analysis completed: {os.path.basename(regionprops_excel_path)}")
    
    # Massive cell distribution plot
    massive_dist_plot_path = plot_massive_cell_distribution(
        regionprops_excel_path,
        color_masks_folder,
        base_name=image_base
    )
    
    # Add plot clone orientation
    orientation_plot_path = plot_clone_orientation(
        regionprops_excel_path,
        color_masks_folder,
        base_name=image_base
    )
    
    # Add plot for color frequency vs clone size
    color_freq_plot_path = plot_color_frequency_vs_clone_size(
        df,
        regionprops_excel_path,
        color_masks_folder,
        base_name=image_base
    )
    
    percentage_plot_path = plot_percentage_cells_in_clone_bins_per_color(
        regionprops_excel_path,
        color_masks_folder,
        base_name=image_base
    )
    
    return df, color_masks, regionprops_excel_path


def display_images_and_masks(brainbow_path, black_cells_mask_path):
    # Load the Brainbow image and the black cells mask
    brainbow = io.imread(brainbow_path)
    black_cells_mask = io.imread(black_cells_mask_path)

    # Transpose the Brainbow image if necessary
    if brainbow.ndim == 3 and brainbow.shape[0] <= 4:
        brainbow = np.transpose(brainbow, (1, 2, 0))

    # Apply contrast stretching to each channel of the Brainbow image
    image_stretched = np.zeros_like(brainbow, dtype=np.float32)
    for i in range(brainbow.shape[2]):
        p1, p99 = np.percentile(brainbow[..., i], (1, 99.5))
        image_stretched[..., i] = exposure.rescale_intensity(
            brainbow[..., i], in_range=(p1, p99), out_range=(0, 1)
        )

    # Apply gamma correction
    gamma = 0.6
    image_gamma = image_stretched ** gamma
    image_final = np.clip(image_gamma * 1.2, 0, 1)

    # Extract the base name of the files for display
    brainbow_name = os.path.basename(brainbow_path)
    mask_name = os.path.basename(black_cells_mask_path)

    # Display the images using Napari with their respective names
    viewer = napari.Viewer()
    viewer.add_image(image_final, name=f'Brainbow Enhanced: {brainbow_name}', rgb=True, contrast_limits=(0, 1))
    viewer.add_image(black_cells_mask, name=f'Black Cells Mask: {mask_name}', colormap='gray')

    napari.run()


def plot_color_distribution_by_area(df, bin_edges=None, bin_labels=None, color_list=None, plot_colors=None, base_name=None, pdf=None):
    """
    Plots the cell color distribution by area bin for the given DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame containing cell area and color information.
    - bin_edges (list): List of area bin edges in µm². Default is [0, 0.1, 0.5, 1, 5, 10, 20, 50, 100].
    - bin_labels (list): List of area bin labels. Default is ['0–0.1 µm²', '0.1–0.5 µm²', '0.5–1 µm²', '1–5 µm²', 
      '5–10 µm²', '10–20 µm²', '20–50 µm²', '>50 µm²'].
    - color_list (list): List of color categories. Default is ['Red', 'Green', 'Blue', 'Yellow', 'Magenta', 'Cyan', 'Black', 'White'].
    - plot_colors (dict): Dictionary mapping color categories to hex color codes.
    
    Returns:
    - pd.DataFrame: A DataFrame with the color distribution percentages for each area bin.
    """
    # Set default values if not provided
    if bin_edges is None:
        bin_edges = [0, 0.1, 0.5, 1, 5, 10, 20, 50, 100]
    if bin_labels is None:
        bin_labels = ['0–0.1 µm²', '0.1–0.5 µm²', '0.5–1 µm²', '1–5 µm²', '5–10 µm²', '10–20 µm²', '20–50 µm²', '>50 µm²']
    if color_list is None:
        color_list = ['Red', 'Green', 'Blue', 'Yellow', 'Magenta', 'Cyan', 'Black', 'White']
    if plot_colors is None:
        plot_colors = {
            'Red': '#FF0000',
            'Green': '#00FF00',
            'Blue': '#0000FF',
            'Yellow': '#FFFF00',
            'Magenta': '#FF00FF',
            'Cyan': '#00FFFF',
            'Black': '#333333',
            'White': '#FFFFFF'
        }

    # Assign each cell to an area bin
    df['area_bin'] = pd.cut(df['area'], bins=bin_edges, labels=bin_labels, right=False, include_lowest=True)

    # Calculate color proportions per area bin
    area_color_counts = {}
    area_totals = {}

    for area_bin in bin_labels:
        bin_df = df[df['area_bin'] == area_bin]
        bin_total = len(bin_df)
        area_totals[area_bin] = bin_total
        if bin_total > 0:
            counts = bin_df['Color'].value_counts().reindex(color_list).fillna(0)
            area_color_counts[area_bin] = (counts / bin_total) * 100
        else:
            area_color_counts[area_bin] = pd.Series(0, index=color_list)

    # Convert to DataFrame
    percentage_df = pd.DataFrame(area_color_counts)

    # Plot grouped bar chart
    fig, ax = plt.subplots(figsize=(14, 8))
    n_colors = len(color_list)
    n_bins = len(bin_labels)
    group_width = 0.8
    bar_width = group_width / n_colors
    max_percentage = percentage_df.max().max()
    y_limit = max_percentage * 1.2

    for i, color in enumerate(color_list):
        positions = np.arange(n_bins) + (i - n_colors/2 + 0.5) * bar_width
        percentages = percentage_df.loc[color].values
        bars = ax.bar(positions, percentages, width=bar_width,
                      color=plot_colors[color], edgecolor='black', linewidth=0.5, label=color)
        for j, bar in enumerate(bars):
            height = bar.get_height()
            if height > 1:
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Cell Area', fontsize=14)
    ax.set_ylabel('Percentage within Each Area Bin (%)', fontsize=14)
    plot_title = 'Cell Color Distribution by Area'
    if base_name:
        plot_title += f'\n({base_name})'
    ax.set_title(plot_title, fontsize=16)    
    ax.set_xticks(np.arange(n_bins))
    ax.set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=12)
    ax.legend(title='Cell Colors', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    # Add total cell counts below x-axis
    for i, area_bin in enumerate(bin_labels):
        total = area_totals[area_bin]
        ax.text(i, -max_percentage * 0.02, f'n={total}', ha='center', va='top', fontsize=10, rotation=45)

    plt.tight_layout()
    plt.ylim(0, y_limit)
    if pdf:
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
    return percentage_df
     
    
def analyze_color_masks_regionprops(brainbow_path, labeled_cells, df, color_masks_folder, brainbow_image):
    """
    Apply regionprops on each color mask and save results to Excel file.
    Parameters:
    - brainbow_path (str): Path to the original brainbow image
    - labeled_cells (numpy.ndarray): Labeled mask of all cells
    - df (pd.DataFrame): DataFrame with cell classifications
    - color_masks_folder (str): Path to the color masks folder
    - brainbow_image (numpy.ndarray): Original brainbow image for intensity measurements
    Returns:
    - str: Path to the created Excel file
    """
    import pandas as pd
    from skimage import measure, morphology
    import numpy as np
    import os
    from openpyxl import Workbook
    from openpyxl.utils.dataframe import dataframe_to_rows

    # Create Excel file path
    image_base = os.path.splitext(os.path.basename(brainbow_path))[0]
    excel_path = os.path.join(color_masks_folder, f"{image_base}_massive_cells_analysis.xlsx")

    # Create dilation kernel (same as used in mask creation)
    dilation_kernel = morphology.disk(1)

    # Dictionary to store results for each color
    color_results = {}
    color_categories = ['Red', 'Green', 'Blue', 'Yellow', 'Magenta', 'Cyan', 'White', 'Black']

    # Conversion factor: 1 px² = 0.010609 µm² (since 1 px = 0.103 µm)
    pixel_area_um2 = 0.103 ** 2

    for color in color_categories:
        print(f"Analyzing {color} massive cells...")

        # Create binary mask for this color
        color_mask = np.zeros_like(labeled_cells, dtype=np.uint16)
        color_cells = df[df['Color'] == color]['label']

        if len(color_cells) == 0:
            print(f"No {color} cells found, skipping...")
            continue

        # Create mask for individual cells of this color
        for label in color_cells:
            color_mask[labeled_cells == label] = 1

        # Dilate the mask to merge adjacent cells (create massive cells)
        dilated_mask = morphology.binary_dilation(color_mask, dilation_kernel).astype(np.uint16)

        # Label the dilated regions to identify massive cells
        labeled_massive, num_massive = measure.label(dilated_mask, return_num=True)

        if num_massive == 0:
            print(f"No massive {color} cells found after dilation")
            continue

        # Extract regionprops for massive cells
        massive_props = measure.regionprops_table(
            labeled_massive,
            brainbow_image,
            properties=('label', 'area', 'centroid', 'mean_intensity', 'max_intensity',
                        'min_intensity', 'perimeter', 'eccentricity', 'solidity')
        )

        # Convert to DataFrame
        massive_df = pd.DataFrame(massive_props)

        if len(massive_df) == 0:
            continue

        # Convert area to microns
        massive_df['area_um2'] = massive_df['area'] * pixel_area_um2

        # Count individual cells and compute orientation
        cell_counts = []
        orientations = []

        for massive_label in massive_df['label']:
            try:
                # Get the region of this massive cell
                massive_region = (labeled_massive == massive_label)

                # Count how many original individual cells are within this region
                individual_cells_in_massive = np.unique(labeled_cells[massive_region])
                individual_cells_in_massive = individual_cells_in_massive[individual_cells_in_massive > 0]  # Remove background

                # Count only cells of the current color
                cells_of_color = df[df['Color'] == color]['label'].values
                cells_in_this_massive = len(np.intersect1d(individual_cells_in_massive, cells_of_color))
                cell_counts.append(cells_in_this_massive)

                # Compute clone orientation
                row = massive_df[massive_df['label'] == massive_label].iloc[0]
                centroid = (row['centroid-0'], row['centroid-1'])
                orientation = compute_clone_orientation(
                    df,
                    centroid,
                    individual_cells_in_massive
                )
                orientations.append(orientation)
            except Exception as e:
                print(f"Error processing massive cell {massive_label}: {str(e)}")
                orientations.append(None)

        massive_df['individual_cells_count'] = cell_counts
        massive_df['orientation'] = orientations

        # Add color information
        massive_df['color'] = color
        massive_df['orientation_angle'] = orientations

        # Reorder columns for better readability
        column_order = ['label', 'color', 'area', 'area_um2', 'individual_cells_count', 'orientation',
                       'centroid-0', 'centroid-1', 'mean_intensity', 'max_intensity',
                       'min_intensity', 'perimeter', 'eccentricity', 'solidity']

        # Only include columns that exist
        existing_columns = [col for col in column_order if col in massive_df.columns]
        massive_df = massive_df[existing_columns]

        # Round numerical values for better display
        numeric_columns = ['area_um2', 'centroid-0', 'centroid-1', 'mean_intensity',
                          'max_intensity', 'min_intensity', 'perimeter', 'eccentricity', 'solidity', 'orientation']

        for col in numeric_columns:
            if col in massive_df.columns:
                massive_df[col] = pd.to_numeric(massive_df[col], errors='coerce').round(3)

        color_results[color] = massive_df
        print(f"Found {len(massive_df)} massive {color} cells containing {sum(cell_counts)} individual cells")

    # Save results to Excel file
    if color_results:
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Create summary sheet
            summary_data = []
            for color, df_color in color_results.items():
                summary_data.append({
                    'Color': color,
                    'Number_of_Massive_Cells': len(df_color),
                    'Total_Individual_Cells': df_color['individual_cells_count'].sum(),
                    'Total_Area_um2': df_color['area_um2'].sum(),
                    'Average_Area_um2': df_color['area_um2'].mean(),
                    'Average_Cells_per_Massive': df_color['individual_cells_count'].mean(),
                    'Median_Cells_per_Massive': df_color['individual_cells_count'].median()
                })

            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)

            # Create individual sheets for each color
            for color, df_color in color_results.items():
                sheet_name = f"{color}_Massive_Cells"
                df_color.to_excel(writer, sheet_name=sheet_name, index=False)

                # Auto-adjust column widths
                worksheet = writer.sheets[sheet_name]
                for column in worksheet.columns:
                    max_length = max(len(str(cell.value)) for cell in column)
                    worksheet.column_dimensions[column[0].column_letter].width = min(max_length + 2, 50)

        print(f"Massive cells analysis saved to: {excel_path}")
    else:
        print("No massive cells found for any color")
        return None

    return excel_path



def plot_massive_cell_distribution(regionprops_excel_path, color_masks_folder, base_name=None):
    """
    Plots the distribution of clone sizes (cells per massive cell) for each color category.
    Each bar represents clones with the exact same number of individual cells.
    
    Parameters:
    - regionprops_excel_path (str): Path to the Excel file with massive cell data
    - color_masks_folder (str): Folder to save the plot
    - base_name (str): Base name for the plot title and filename
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from collections import Counter
    
    # Define color mapping
    color_map = {
        'Red': 'red',
        'Green': 'green',
        'Blue': 'blue',
        'Yellow': 'yellow',
        'Magenta': 'magenta',
        'Cyan': 'cyan',
        'White': 'gray',
        'Black': 'black'
    }
    
    # Define color order for consistent plotting
    color_order = ['Red', 'Green', 'Blue', 'Yellow', 'Magenta', 'Cyan', 'White', 'Black']
    
    # Read data from Excel
    all_data = {}
    
    with pd.ExcelFile(regionprops_excel_path) as xls:
        # Read summary sheet to get total clone counts per color
        summary_df = pd.read_excel(xls, sheet_name='Summary')
        color_totals = dict(zip(summary_df['Color'], summary_df['Number_of_Massive_Cells']))
        
        for color in color_order:
            sheet_name = f"{color}_Massive_Cells"
            if sheet_name in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name)
                if 'individual_cells_count' in df.columns:
                    counts = df['individual_cells_count']
                    all_data[color] = counts
    
    # Create figure with subplots for each color
    fig, axs = plt.subplots(2, 4, figsize=(24, 12))
    axs = axs.flatten()  # Flatten to 1D array for easier indexing
    
    # Plot distribution for each color
    for i, color in enumerate(color_order):
        ax = axs[i]
        if color in all_data and len(all_data[color]) > 0:
            counts = all_data[color]
            total_clones = color_totals[color]
            
            # Count frequency of each unique clone size
            clone_size_counts = Counter(counts)
            
            # Get unique clone sizes and their frequencies
            unique_sizes = sorted(clone_size_counts.keys())
            frequencies = [clone_size_counts[size] / total_clones for size in unique_sizes]
            
            # Determine appropriate bar width based on data density
            if len(unique_sizes) == 1:
                # Only one unique size - use a reasonable fixed width
                bar_width = max(1, unique_sizes[0] * 0.1)
            else:
                # Multiple sizes
                size_range = max(unique_sizes) - min(unique_sizes)
                if size_range <= 50:
                    bar_width = 0.8  # Fine resolution for small ranges
                elif size_range <= 500:
                    bar_width = min(5, size_range / len(unique_sizes) * 0.8)  # Adaptive for medium ranges
                else:
                    bar_width = min(50, size_range / len(unique_sizes) * 0.8)  # Adaptive for large ranges
            
            # Plot bars at exact clone sizes
            bars = ax.bar(unique_sizes, frequencies, width=bar_width, 
                         color=color_map[color], alpha=0.7, edgecolor='white', linewidth=0.5)
            
            # Add text for detailed statistics
            mean_size = counts.mean()
            median_size = counts.median()
            max_size = counts.max()
            min_size = counts.min()
            std_size = counts.std()
            
            stats_text = f'n={total_clones}\nRange: {min_size:.0f}-{max_size:.0f}\nMean: {mean_size:.1f}±{std_size:.1f}\nMedian: {median_size:.1f}'
            ax.text(0.95, 0.95, stats_text, 
                    transform=ax.transAxes, ha='right', va='top',
                    bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'), fontsize=9)
            
            # Set title and labels
            ax.set_title(f'{color}', fontsize=14, color=color_map[color], fontweight='bold')
            ax.set_xlabel('Cells per Clone', fontsize=12)
            
            # Only add ylabel to leftmost plots
            if i % 4 == 0:
                ax.set_ylabel('Frequency', fontsize=12)
            
            # Set x-axis limits with padding
            x_padding = max(1, (max_size - min_size) * 0.05)
            ax.set_xlim(max(0, min_size - x_padding), max_size + x_padding)
            
            # Set y-axis limits
            max_freq = max(frequencies)
            ax.set_ylim(0, max_freq * 1.15)
            
            # Add grid for readability
            ax.grid(axis='y', linestyle='--', alpha=0.3)
            
            # Smart x-axis ticks
            size_range = max_size - min_size
            if size_range <= 20:
                # Show every value for small ranges
                ax.set_xticks(unique_sizes)
            elif size_range <= 100:
                # Show every 5th or 10th value
                step = 5 if size_range <= 50 else 10
                tick_start = int(min_size / step) * step
                tick_end = int(max_size / step + 1) * step
                ax.set_xticks(range(tick_start, tick_end + 1, step))
            else:
                # Show fewer ticks for large ranges
                step = 50 if size_range <= 500 else 100
                tick_start = int(min_size / step) * step
                tick_end = int(max_size / step + 1) * step
                ax.set_xticks(range(tick_start, tick_end + 1, step))
            
            if size_range > 50:
                ax.tick_params(axis='x', rotation=45)
            
            # Highlight the most common clone size
            if frequencies:
                max_freq_idx = frequencies.index(max_freq)
                most_common_size = unique_sizes[max_freq_idx]
                bars[max_freq_idx].set_alpha(1.0)  # Make the mode more prominent
                       
        else:
            # Hide empty subplots
            ax.set_visible(False)
    
    # Set main title
    main_title = 'Clone Size Distribution by Color'
    if base_name:
        main_title += f' - {base_name}'
    fig.suptitle(main_title, fontsize=18, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
    
    # Save plot
    plot_filename = f"clone_size_distribution_{base_name}.png" if base_name else "clone_size_distribution.png"
    plot_path = os.path.join(color_masks_folder, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved clone size distribution plot: {plot_path}")
    
    # Print debug information for each color
    print("\nClone size distribution summary:")
    for color in color_order:
        if color in all_data and len(all_data[color]) > 0:
            counts = all_data[color]
            clone_size_counts = Counter(counts)
            unique_sizes = sorted(clone_size_counts.keys())
            
            print(f"\n{color}:")
            print(f"  Total clones: {len(counts)}")
            print(f"  Unique clone sizes: {len(unique_sizes)}")
            print(f"  Size range: {min(unique_sizes)} - {max(unique_sizes)} cells")
            
            # Show most common sizes
            most_common = clone_size_counts.most_common(3)
            print(f"  Most common sizes:")
            for size, count in most_common:
                freq = count / len(counts)
                print(f"    {size} cells: {count} clones ({freq:.1%})")
    
    return plot_path


def compute_clone_orientation(individual_cells_df, massive_cell_centroid, cell_labels):
    """
    Compute the orientation of a cellular clone using PCA.
    
    Parameters:
    - individual_cells_df (pd.DataFrame): DataFrame containing all individual cells' data
    - massive_cell_centroid (tuple): (x, y) coordinates of the clone's centroid
    - cell_labels (list): Labels of cells belonging to the clone
    
    Returns:
    - float: Orientation angle in degrees (0-180° relative to horizontal axis)
    """
    # Filter cells belonging to this clone
    clone_cells = individual_cells_df[individual_cells_df['label'].isin(cell_labels)]
    if len(clone_cells) < 2:
        return np.nan  # Not enough points for PCA
    
    # Center coordinates around clone centroid
    points = np.array([
        clone_cells['centroid-0'] - massive_cell_centroid[0],
        clone_cells['centroid-1'] - massive_cell_centroid[1]
    ]).T
    
    # Compute covariance matrix
    cov_matrix = np.cov(points.T)
    
    # Perform PCA
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Get main principal component (largest eigenvalue)
    main_component = eigenvectors[:, np.argmax(eigenvalues)]
    
    # Calculate angle relative to horizontal axis (0-180°)
    angle_rad = np.arctan2(main_component[1], main_component[0])
    angle_deg = np.degrees(angle_rad) % 180
    
    return angle_deg


def plot_clone_orientation(regionprops_excel_path, output_folder, base_name=None):
    import pandas as pd
    import matplotlib.pyplot as plt
    import os
    import numpy as np

    # Define size categories
    size_categories = {
        "Small (2-4 cells)": (2, 4),
        "Medium (5-10 cells)": (5, 10),
        "Large (11+ cells)": (11, 1000)
    }

    # Define color categories to process
    color_categories = ['Red', 'Green', 'Blue', 'Yellow', 'Magenta', 'Cyan', 'White', 'Black']

    # Define color mapping for clone categories
    color_map = {
        'Red': 'red',
        'Green': 'green',
        'Blue': 'blue',
        'Yellow': 'gold',
        'Magenta': 'magenta',
        'Cyan': 'cyan',
        'White': 'lightgray',
        'Black': 'black'
    }

    # Read data from Excel
    color_data = {}
    with pd.ExcelFile(regionprops_excel_path) as xls:
        for color in color_categories:
            sheet_name = f"{color}_Massive_Cells"
            if sheet_name in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet_name)
                if 'orientation' in df.columns and 'individual_cells_count' in df.columns:
                    valid_df = df.dropna(subset=['orientation'])
                    valid_df = valid_df[valid_df['orientation'].between(0, 180)]
                    color_data[color] = valid_df

    # Generate separate plot for each color
    plot_paths = []
    for color, df in color_data.items():
        fig = plt.figure(figsize=(14, 12))
        fig.suptitle(f'Clone Orientation - {color} ({base_name})' if base_name else f'Clone Orientation - {color}', fontsize=16)

        # Create subplots without colorbar space
        ax1 = fig.add_subplot(221, projection='polar')
        ax2 = fig.add_subplot(222, projection='polar')
        ax3 = fig.add_subplot(223, projection='polar')
        ax4 = fig.add_subplot(224, projection='polar')
        axes = [ax1, ax2, ax3, ax4]

        plt.subplots_adjust(
            left=0.05,
            right=0.95,  # Removed colorbar space
            top=0.92,
            bottom=0.08,
            wspace=0.3,
            hspace=0.4
        )

        # Configure polar plots
        for ax in axes:
            ax.set_theta_zero_location("E")
            ax.set_theta_direction(1)
            ax.set_thetalim(0, np.pi)
            ax.set_rlabel_position(0)
            ax.grid(True, alpha=0.3)
            ax.set_thetagrids(np.arange(0, 181, 45), labels=['0°', '45°', '90°', '135°', '180°'])

        # Set titles
        ax1.set_title("Small Clones (2-4 cells)", pad=20)
        ax2.set_title("Medium Clones (5-10 cells)", pad=20)
        ax3.set_title("Large Clones (11+ cells)", pad=20)
        ax4.set_title("All Clones", pad=20)

        # Use clone-specific color
        bar_color = color_map.get(color, 'blue')

        size_angles = {size_name: [] for size_name in size_categories}
        all_angles = []
        angles = df['orientation'].tolist()
        counts = df['individual_cells_count'].tolist()

        for angle, count in zip(angles, counts):
            for size_name, (min_size, max_size) in size_categories.items():
                if min_size <= count <= max_size:
                    size_angles[size_name].append(angle)
            all_angles.append(angle)

        size_angles["All Clones"] = all_angles

        for ax, (size_name, angles) in zip(axes, size_angles.items()):
            if not angles:
                ax.set_title(f"{size_name}\n(No Data)", color='red')
                continue

            radians = np.deg2rad(angles)
            n_bins = 18
            bins = np.linspace(0, np.pi, n_bins + 1)
            hist, bin_edges = np.histogram(radians, bins=bins)
            hist = hist / len(angles)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            widths = (bin_edges[1] - bin_edges[0]) * 0.8

            # Plot with uniform color
            ax.bar(bin_centers, hist, width=widths, bottom=0.0, color=bar_color, alpha=0.7, edgecolor='black', linewidth=0.5)

            max_freq = max(hist) if len(hist) > 0 else 0.1
            ax.set_ylim(0, max_freq * 1.1)
            n_ticks = 5
            r_ticks = np.linspace(0, max_freq, n_ticks)
            ax.set_yticks(r_ticks)
            ax.set_yticklabels([f'{x:.2f}' for x in r_ticks], fontsize=8)

            if angles:
                mean_angle = np.mean(angles)
                ax.text(np.pi/2, ax.get_rmax()*0.7, f"n={len(angles)}\nMean: {mean_angle:.1f}°", ha='center', va='center', fontsize=9, bbox=dict(facecolor='white', alpha=0.8, pad=3))

        plot_filename = f"clone_orientation_{color}_{base_name}.png" if base_name else f"clone_orientation_{color}.png"
        plot_path = os.path.join(output_folder, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths.append(plot_path)
        print(f"Saved {color} clone orientation plot: {plot_path}")

    return plot_paths


def plot_color_frequency_vs_clone_size(df, regionprops_excel_path, output_folder, base_name=None):
    """
    Plots clone size vs. color frequency for an image.
    
    Parameters:
    - df: Main DataFrame with cell classifications
    - regionprops_excel_path: Path to massive cells analysis Excel
    - output_folder: Folder to save plot
    - base_name: Base name for plot title
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import os
    import numpy as np
    from scipy.stats import linregress

    # Get color frequencies from main DataFrame
    color_counts = df['Color'].value_counts()
    total_cells = len(df)
    color_freq = (color_counts / total_cells * 100).to_dict()

    # Read clone size data from Excel
    clone_size_data = {}
    color_categories = ['Red', 'Green', 'Blue', 'Yellow', 'Magenta', 'Cyan', 'White', 'Black']
    
    try:
        with pd.ExcelFile(regionprops_excel_path) as xls:
            for color in color_categories:
                sheet_name = f"{color}_Massive_Cells"
                if sheet_name in xls.sheet_names:
                    df_color = pd.read_excel(xls, sheet_name=sheet_name)
                    if 'individual_cells_count' in df_color.columns:
                        clone_size_data[color] = df_color['individual_cells_count'].mean()
    except Exception as e:
        print(f"Error reading Excel file: {str(e)}")
        return None

    # Prepare plot data
    plot_data = []
    for color, size in clone_size_data.items():
        if color in color_freq:
            plot_data.append({
                'color': color,
                'frequency': color_freq[color],
                'size': size
            })
    
    if not plot_data:
        print("No valid data for plotting")
        return None
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = {
        'Red': 'red', 'Green': 'green', 'Blue': 'blue', 
        'Yellow': 'gold', 'Magenta': 'magenta', 
        'Cyan': 'cyan', 'White': 'gray', 'Black': 'black'
    }
    
    for _, row in plot_df.iterrows():
        ax.scatter(
            row['frequency'], 
            row['size'],
            s=150,
            color=colors[row['color']],
            edgecolor='black',
            label=row['color']
        )
    
    # Add linear regression
    if len(plot_df) > 1:
        x = plot_df['frequency']
        y = plot_df['size']
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        line_x = np.linspace(min(x)-5, max(x)+5, 100)
        line_y = slope * line_x + intercept
        ax.plot(line_x, line_y, 'k--', alpha=0.7)
        ax.text(0.05, 0.95, 
                f"R² = {r_value**2:.3f}\np = {p_value:.4f}", 
                transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Color Frequency (%)', fontsize=14)
    ax.set_ylabel('Average Clone Size (cells/clone)', fontsize=14)
    title = 'Clone Size vs. Color Frequency'
    if base_name:
        title += f'\n{base_name}'
    ax.set_title(title, fontsize=16)
    ax.grid(alpha=0.3)
    ax.legend(title='Color Categories')
    
    # Save plot
    plot_filename = f"color_freq_vs_clone_size_{base_name}.png" if base_name else "color_freq_vs_clone_size.png"
    plot_path = os.path.join(output_folder, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved color frequency vs clone size plot: {plot_path}")
    return plot_path