# FILE : main.py

# USE :
# 1. Put the Brainbow images and their masks in a folder
# 2. Modify the input/output paths in __main__
# 3. Execut main.py

"""
MAIN SCRIPT FOR BATCH PROCESSING OF BRAINBOW IMAGES

This script processes Brainbow images and their corresponding mask files to analyze:
- Cell colors (8 possible colors: Red, Green, Blue, Yellow, Magenta, Cyan, White, Black)
- Clone sizes (groups of adjacent same-colored cells)
- Cell distributions and orientations

INPUT FILE ORGANIZATION:
- The input folder should contain:
  - Brainbow images: TIFF files named with patterns like "LocZP_*.tif" or "CORR_LocZP_*.tif"
  - Mask images: TIFF files named with patterns like "masque_*.tif"
  - Optional: Excel files with manual measurements for comparison (named with patterns that match the image names)

- The script will recursively search for image/mask pairs in the input folder and its subfolders

PROCESSING STEPS:
1. Find all image/mask pairs using filename patterns
2. For each pair:
   a. Extract cell data (area, centroid, intensity) and save to a temporary CSV
   b. Convert area from pixels to micrometers
   c. Classify cells into one of 8 colors based on RGB thresholds
   d. Merge adjacent cells of the same color into clones
   e. Generate plots and analysis for the image
3. Aggregate results from all images into a combined Excel file and PDF report

OUTPUTS:
After running the script, the output folder will contain:

FOR EACH IMAGE:
- A CSV file with cell measurements in micrometers (e.g., "[image_name]_um.csv")
- Color masks (TIFF images) for each color category in a "[image_name]_color_masks" subfolder
- An Excel file with clone analysis (region properties) in the color masks folder
- Various plots (PDF and PNG) showing:
  - Cell size distribution
  - Color distribution by area
  - Maxwell triangle projection of colors
  - Clone size distribution per color
  - Clone orientation analysis
  - Color frequency vs. clone size relationships

AGGREGATED ACROSS ALL IMAGES:
- combined_results.xlsx: Combined data from all processed images
- all_plots.pdf: PDF containing all generated plots
- Additional aggregated analysis files for:
  - Clone size evolution across time points
  - Clone orientation aggregation
  - Color frequency vs. clone size relationships
  - Clone size distribution per color
  - Percentage of cells in clone bins per color

Note: The script uses the brainbow_tools module for image processing functions.
"""

import os
from glob import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import numpy as np
import sys
sys.path.append(r'C:\Users\33672\Documents\Stage X\IPyprocess')
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
import brainbow_tools
from matplotlib.backends.backend_pdf import PdfPages
from brainbow_tools import extract_data_image, convert_area_to_microns, plot_cell_size_distribution, plot_color_distribution_by_area, plot_maxwell_triangle_projection, process_brainbow_image, display_images_and_masks


def find_image_mask_pairs(folder_path):
    """
    IMAGE/MASK CORRELATION: Recursively searches the folder to find matching Brainbow image (.tif) and binary mask (.tif) pairs.
    Uses filename-based matching logic.
    """
    
    # Include CORR files in search
    brainbow_files = glob(os.path.join(folder_path, "LocZP_*.tif")) + \
                     glob(os.path.join(folder_path, "CORR_LocZP_*.tif"))
    mask_files = glob(os.path.join(folder_path, "masque_*.tif"))

    pairs = []
    
    # Create mapping of simplified names to paths
    image_map = {}
    for img_path in brainbow_files:
        img_name = os.path.basename(img_path)
        base_name = re.sub(r'^(CORR_LocZP_|LocZP_)', '', img_name).replace(".tif", "")
        base_name = re.sub(r'_s\d+-s\d+$', '', base_name)  # Remove _sX-sXX suffix
        
        base_name = re.sub(r'_MIP|_Mid-Mid|_Mid|hR_Mid|hL_Mid|_1q', '', base_name)
        base_name = re.sub(r'_(ANT|MID|POST)$', '', base_name, flags=re.IGNORECASE)

        
        # Store both original and simplified versions
        image_map[base_name] = img_path
        print(f"Image: {img_name} -> Base: {base_name}")  # Debug info moved inside loop
    
    for mask_path in mask_files:
        mask_name = os.path.basename(mask_path)
        # Handle different mask prefixes including CORR
        if mask_name.startswith("masque_CORR_RoiSet_"):
            mask_base = mask_name.replace("masque_CORR_RoiSet_", "").replace(".tif", "")
        elif mask_name.startswith("masque_RoiSet_"):
            mask_base = mask_name.replace("masque_RoiSet_", "").replace(".tif", "")
        else:
            mask_base = mask_name.replace("masque_", "").replace(".tif", "")
        
        mask_base = re.sub(r'^(CORR_LocZP_|LocZP_)', '', mask_base)
        mask_base = re.sub(r'_s\d+-s\d+$', '', mask_base)
        
        mask_base = re.sub(r'_MIP|_Mid-Mid|_Mid|hR_Mid|hL_Mid|_1q', '', mask_base)
        mask_base = re.sub(r'_(ANT|MID|POST)$', '', mask_base, flags=re.IGNORECASE)

        print(f"Mask: {mask_name} -> Base: {mask_base}")
        
        # Try different matching strategies in order of specificity
        matched = False
        
        # Option 1: Exact match after cleaning both names
        if mask_base in image_map:
            pairs.append((image_map[mask_base], mask_path))
            print(f"  -> Matched with exact match: {os.path.basename(image_map[mask_base])}")
            matched = True

        if matched: continue      

        # Option 2: Exact match for CROP cases
        if "CROP" in mask_base:
            for img_base, img_path in image_map.items():
                # Compare only the CROP portion
                mask_crop_part = mask_base.split("EMX1TCyt5_")[-1]
                img_crop_part = img_base.split("EMX1TCyt5_")[-1]
                
                # Normalize by removing Mid-Mid and variations
                mask_crop_part = mask_crop_part.replace("Mid-Mid_", "").replace("Mid_", "")
                img_crop_part = img_crop_part.replace("Mid-Mid_", "").replace("Mid_", "")
                
                if mask_crop_part == img_crop_part:
                    pairs.append((img_path, mask_path))
                    print(f"  -> Matched with CROP logic: {os.path.basename(img_path)}")
                    matched = True
                    break
        
        if matched: continue

       # Option 3: Match by eXXX pattern with hemisphere and L/M distinction
        mask_e_match = re.search(r'e\d+[\-_]?\d*', mask_base)
        if mask_e_match and not matched:
            mask_e_part = mask_e_match.group(0).replace("-", "").replace("_", "")

            # Extract hemisphere (hL/hR) and L/M number
            mask_hemi_match = re.search(r'h[LR]', mask_base)
            mask_hemi = mask_hemi_match.group(0) if mask_hemi_match else None
            mask_lm_match = re.search(r'[LM]\d+', mask_base)
            mask_lm = mask_lm_match.group(0) if mask_lm_match else None

            for img_base, img_path in image_map.items():
                img_e_match = re.search(r'e\d+[\-_]?\d*', img_base)
                if img_e_match:
                    img_e_part = img_e_match.group(0).replace("-", "").replace("_", "")
                    if img_e_part == mask_e_part:
                        img_hemi_match = re.search(r'h[LR]', img_base)
                        img_hemi = img_hemi_match.group(0) if img_hemi_match else None
                        img_lm_match = re.search(r'[LM]\d+', img_base)
                        img_lm = img_lm_match.group(0) if img_lm_match else None

                        if mask_hemi and mask_lm and img_hemi == mask_hemi and img_lm == mask_lm:
                            pairs.append((img_path, mask_path))
                            matched = True
                            break

        if not matched:
            print(f"  -> No match found for {mask_name}")

    # Remove duplicate pairs and ensure 1:1 matching
    unique_pairs = []
    used_images = set()
    used_masks = set()

    for img, mask in pairs:
        if img not in used_images and mask not in used_masks:
            unique_pairs.append((img, mask))
            used_images.add(img)
            used_masks.add(mask)
        else:
            print(f"Skipping duplicate: {os.path.basename(img)} - {os.path.basename(mask)}")

    print(f"\nFinal pairs:")
    for i, (img, mask) in enumerate(unique_pairs, 1):
        print(f"{i}. {os.path.basename(img)} <-> {os.path.basename(mask)}")

    return unique_pairs


def extract_common_name(full_name):
    """
    Extracts the common part of the image name to match with Excel files.
    Handles both original and CORR filenames.
    """
    # Remove prefixes and extension
    common_name = re.sub(r'^(CORR_LocZP_|LocZP_|CORR_)', '', full_name).replace(".tif", "")

    # Remove slice information if present
    if "_s" in common_name:
        common_name = common_name.split("_s")[0]

    # Remove 40X information but keep the number
    common_name = re.sub(r'_40X-?(\d+)', r'_\1', common_name)
    
    common_name = re.sub(r'_MIP|_Mid-Mid|_Mid|hR_Mid|hL_Mid|_1q', '', common_name)
    common_name = re.sub(r'_Crop_|_CROP_', '_crop_', common_name)
    
    return common_name


def extract_common_name_from_excel(excel_filename):
    """
    Extracts the common part of the Excel filename to match with image files.
    Handles both original and CORR filenames.
    """
    # Remove the path if present and get just the filename
    filename = os.path.basename(excel_filename)
    
    common_name = filename.replace("RGBint_", "")
    common_name = common_name.replace("00RGBint_", "")
    common_name = re.sub(r'^(CORR_)', '', common_name)  # Handle CORR prefix
    
    common_name = common_name.replace(".xlsx", "")
    
    # Remove date pattern (YYYY-MM-DD format at the end)
    common_name = re.sub(r'_\d{4}-\d{2}-\d{2}$', '', common_name)
    
    # Apply the same transformations as the image function
    # Remove slice information if present
    if "_s" in common_name:
        common_name = common_name.split("_s")[0]
    
    # Remove 40X information but keep the number
    common_name = re.sub(r'_40X-?(\d+)', r'_\1', common_name)
    
    common_name = re.sub(r'_MIP|_Mid_Mid|_Mid|hR_Mid|hL_Mid|_1q', '', common_name)
    common_name = re.sub(r'_Crop_|_CROP_', '_crop_', common_name)
    
    return common_name


def format_sheet_name(full_name):
    """
    Format sheet name to be under 31 characters while keeping it identifiable
    Handles both original and CORR filenames.
    """
    # Remove extension and CORR prefixes
    base_name = os.path.splitext(full_name)[0]
    base_name = re.sub(r'^(CORR_LocZP_|LocZP_|CORR_)', '', base_name)
    
    formatted_name = (
        base_name.replace("EMX1TCyt5_", "")  # Remove prefix
                .replace("_40X-", "")
                .replace("_Post", "")
                .replace("_hR_Mid", "_hR_M")  # Shorten hR_Mid to hR
                .replace("_hL_Mid", "_hL_M")  # Shorten hL_Mid to hL
                .replace("_Mid-Mid", "_MM")  # Shorten Mid-Mid to M
                .replace("_Mid", "_M")      # Shorten Mid to M
    )
    
    # Remove slice information (s1-s10, s1-s7, etc.)
    if "_s" in formatted_name:
        formatted_name = formatted_name.split("_s")[0]
    
    # Shorten CROP indicators
    formatted_name = formatted_name.replace("_CROP_1q_", "_C_")
    
    # Remove underscores around L to make it cleaner
    formatted_name = formatted_name.replace("_L_", "_L")
    
    # Ensure it's under 31 characters
    if len(formatted_name) > 31:
        # If still too long, truncate but keep the key identifier (e-number)
        formatted_name = formatted_name[:31]
    
    return formatted_name


def find_matching_excel_file(image_path, input_folder):
    """
    Find the matching Excel file for a given image file.
    Uses the same eXXX pattern matching logic as the mask/image pairing.
    Handles both original and CORR filenames.
    """
    # Extract common name from image
    image_common_name = extract_common_name(os.path.basename(image_path))
    
    # Get all Excel files
    all_excel_files = glob(os.path.join(input_folder, "*.xlsx"))
    
    # Try to find a match
    for excel_file in all_excel_files:
        excel_common_name = extract_common_name_from_excel(excel_file)
        
        # Direct match
        if image_common_name == excel_common_name:
            return excel_file
        
        # Match by eXXXX pattern (same logic as your mask matching)
        image_e_match = re.search(r'(e\w+[\-_]?\w*)', image_common_name)
        excel_e_match = re.search(r'(e\w+[\-_]?\w*)', excel_common_name)
        
        if image_e_match and excel_e_match:
            # Normalize by removing dashes and underscores (same as your mask logic)
            image_e_part = image_e_match.group(1).replace("-", "").replace("_", "")
            excel_e_part = excel_e_match.group(1).replace("-", "").replace("_", "")
            
            if image_e_part == excel_e_part:
                # Additional check to make sure other parts are compatible
                # Remove the e-pattern from both names and compare the rest
                image_base = image_common_name.replace(image_e_match.group(1), "")
                excel_base = excel_common_name.replace(excel_e_match.group(1), "")
                
                # Remove crop-specific differences for comparison
                image_base_clean = re.sub(r'_Crop_\w+|_CROP_\w+', '', image_base)
                excel_base_clean = re.sub(r'_Crop_\w+|_CROP_\w+', '', excel_base)
                
                # If the bases match or are very similar, it's a match
                if (image_base_clean == excel_base_clean or 
                    image_base_clean in excel_base_clean or 
                    excel_base_clean in image_base_clean):
                    return excel_file
        
        # Fallback: partial match for cases with different naming conventions
        if image_common_name in excel_common_name or excel_common_name in image_common_name:
            return excel_file
    
    return None


def process_folder(input_folder, output_folder):
    """
    PROCESSING A COMPLETE FOLDER: Orchestrates the processing of all found image/mask pairs.
    Generates outputs in an organized structure.
    """
    
    os.makedirs(output_folder, exist_ok=True)
    pairs = find_image_mask_pairs(input_folder)
    if not pairs:
        print(f"No matching image/mask pairs found in {input_folder}")
        return
    print(f"Found {len(pairs)} image/mask pairs to process")
    combined_excel_path = os.path.join(output_folder, "combined_results.xlsx")
    pdf_path = os.path.join(output_folder, "all_plots.pdf")
    all_results = {}

    with PdfPages(pdf_path) as pdf:
        for i, (brainbow_path, mask_path) in enumerate(pairs, 1):
            full_name = os.path.basename(brainbow_path)
            short_name = format_sheet_name(full_name)
            image_base = os.path.splitext(full_name)[0]
            print(f"\nProcessing pair {i}/{len(pairs)}: {short_name}")

            try:
                temp_csv = os.path.join(output_folder, f"temp_{full_name}.csv")
                permanent_um_csv = os.path.join(output_folder, f"{image_base}_um.csv")

                df = extract_data_image(brainbow_path, mask_path, temp_csv)
                df_um = convert_area_to_microns(temp_csv, permanent_um_csv)

                plot_cell_size_distribution(df_um, pdf=pdf, base_name=short_name)
                common_name = extract_common_name(os.path.basename(brainbow_path))
                print(f"Common name extracted: {common_name}")

                excel_path = find_matching_excel_file(brainbow_path, input_folder)
                df_colors, color_masks, regionprops_excel_path = process_brainbow_image(
                    brainbow_path, mask_path, permanent_um_csv,
                    pdf=pdf, base_name=short_name, excel_path=excel_path
                )

                image_base = os.path.splitext(os.path.basename(brainbow_path))[0]
                color_masks_folder = os.path.join(os.path.dirname(brainbow_path), f"{image_base}_color_masks")
                regionprops_excel = os.path.join(color_masks_folder, f"{image_base}_massive_cells_analysis.xlsx")
                if os.path.exists(regionprops_excel):
                    print(f"Massive cells analysis available: {os.path.basename(regionprops_excel)}")

                plot_color_distribution_by_area(df_colors, pdf=pdf, base_name=short_name)
                plot_maxwell_triangle_projection(df_colors, pdf=pdf, base_name=short_name)

                cols_to_drop = ['area_bin', 'r_norm', 'g_norm', 'b_norm']
                existing_cols = [col for col in cols_to_drop if col in df_colors.columns]
                df_clean = df_colors.drop(columns=existing_cols)

                all_results[short_name] = df_clean
                os.remove(temp_csv)  # Only delete the temporary file
                print(f"Successfully processed {short_name}")
                for color, mask_path in color_masks.items():
                    print(f"Generated {color} mask: {os.path.basename(mask_path)}")

            except Exception as e:
                print(f"Error processing {short_name}: {str(e)}")
                import traceback
                traceback.print_exc()
                for f in [temp_csv]:
                    if os.path.exists(f):
                        os.remove(f)

    with pd.ExcelWriter(combined_excel_path, engine='openpyxl') as writer:
        for short_name, df in all_results.items():
            sheet_name = re.sub(r'[\\/*?:[\]]', '', short_name)[:31]
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            worksheet = writer.sheets[sheet_name]

            for column in worksheet.columns:
                max_length = max(len(str(cell.value)) for cell in column)
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column[0].column_letter].width = adjusted_width

    print(f"\nAll results saved to: {combined_excel_path}")
    print(f"All plots saved to: {pdf_path}")
           
  
def aggregate_clone_sizes(root_output_folder):
    """
    Collects clone size data from each time point folder and plots evolution over time
    with additional distribution plots.
    """
    root_folder_name = os.path.basename(root_output_folder)
    time_point_results = {}

    # Walk through all subdirectories to find massive_cells_analysis files
    for root, dirs, files in os.walk(root_output_folder):
        excel_files = [f for f in files if f.endswith("_massive_cells_analysis.xlsx")]
        
        for excel_file in excel_files:
            excel_path = os.path.join(root, excel_file)
            try:
                folder_name = os.path.basename(root)
                match = re.search(r'E(\d+)', folder_name)
                if not match:
                    continue
                
                time_point = f"E{match.group(1)}"
                
                # Read summary sheet
                summary_df = pd.read_excel(excel_path, sheet_name='Summary')
                
                # Initialize storage for this time point if needed
                if time_point not in time_point_results:
                    time_point_results[time_point] = {
                        'Red': {'means': [], 'medians': []}, 
                        'Green': {'means': [], 'medians': []},
                        'Blue': {'means': [], 'medians': []},
                        'Yellow': {'means': [], 'medians': []},
                        'Magenta': {'means': [], 'medians': []},
                        'Cyan': {'means': [], 'medians': []},
                        'White': {'means': [], 'medians': []},
                        'Black': {'means': [], 'medians': []}
                    }
                
                # Collect data for each color
                for _, row in summary_df.iterrows():
                    color = row['Color']
                    if color in time_point_results[time_point]:
                        time_point_results[time_point][color]['means'].append(row['Average_Cells_per_Massive'])
                        time_point_results[time_point][color]['medians'].append(row['Median_Cells_per_Massive'])
                        
            except Exception as e:
                print(f"Error processing {excel_file}: {str(e)}")
    
    if not time_point_results:
        print("No clone size data found")
        return
    
    # Color mapping
    color_map = {
        'Red': 'red', 'Green': 'green', 'Blue': 'blue', 
        'Yellow': 'yellow', 'Magenta': 'magenta', 'Cyan': 'cyan', 
        'White': 'gray', 'Black': 'black'
    }
    
    # Extract and sort time points
    time_points = sorted(list(time_point_results.keys()), key=lambda x: int(x[1:]))
    numeric_days = [int(tp[1:]) for tp in time_points]

    # PLOT 1: Distribution of means
    plt.figure(figsize=(14, 10))
    for i, color in enumerate(color_map.keys(), 1):
        plt.subplot(3, 3, i)
        all_means = []
        positions = []
        
        for j, tp in enumerate(time_points):
            if color in time_point_results[tp]:
                means = time_point_results[tp][color]['means']
                if means:
                    # Add slight jitter to x-position
                    x = [numeric_days[j] + np.random.uniform(-0.2, 0.2) for _ in means]
                    plt.scatter(x, means, alpha=0.6, color=color_map[color])
                    all_means.extend(means)
                    positions.append(numeric_days[j])
        
        if positions and all_means:
            # Add median line
            medians = [np.median(time_point_results[tp][color]['means']) for tp in time_points 
                       if color in time_point_results[tp] and time_point_results[tp][color]['means']]
            plt.plot(positions, medians, 'k-', lw=2, alpha=0.7)
            
        plt.title(color)
        plt.xlabel('Embryonic Day')
        plt.ylabel('Mean Cells per Clone')
        plt.xticks(numeric_days, time_points)
        plt.grid(alpha=0.2)
    
    plt.suptitle(f'Distribution of Mean Clone Sizes by Color - {root_folder_name}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path1 = os.path.join(root_output_folder, "mean_clone_size_distribution.png")
    plt.savefig(plot_path1, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved mean clone size distribution: {plot_path1}")

    # PLOT 2: Median evolution
    plt.figure(figsize=(12, 8))
    for color, color_name in color_map.items():
        median_values = []
        valid_days = []
        
        for tp in time_points:
            if color in time_point_results[tp] and time_point_results[tp][color]['medians']:
                median_of_medians = np.median(time_point_results[tp][color]['medians'])
                median_values.append(median_of_medians)
                valid_days.append(int(tp[1:]))
        
        if median_values:
            plt.plot(valid_days, median_values, label=color, 
                     color=color_name, marker='o', markersize=8, lw=2)
    
    plt.xlabel('Embryonic Day', fontsize=14)
    plt.ylabel('Median Clone Size (cells per clone)', fontsize=14)
    plt.title(f'Median Clone Size Evolution by Color - {root_folder_name}', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.xticks(numeric_days, time_points, fontsize=12)
    plt.yticks(fontsize=12)
    plot_path2 = os.path.join(root_output_folder, "median_clone_size_evolution.png")
    plt.savefig(plot_path2, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved median clone size evolution: {plot_path2}")

    # PLOT 3: Distribution of medians
    plt.figure(figsize=(14, 10))
    for i, color in enumerate(color_map.keys(), 1):
        plt.subplot(3, 3, i)
        all_medians = []
        positions = []
        
        for j, tp in enumerate(time_points):
            if color in time_point_results[tp]:
                medians = time_point_results[tp][color]['medians']
                if medians:
                    # Add slight jitter to x-position
                    x = [numeric_days[j] + np.random.uniform(-0.2, 0.2) for _ in medians]
                    plt.scatter(x, medians, alpha=0.6, color=color_map[color])
                    all_medians.extend(medians)
                    positions.append(numeric_days[j])
        
        if positions and all_medians:
            # Add median line
            median_line = [np.median(time_point_results[tp][color]['medians']) for tp in time_points 
                           if color in time_point_results[tp] and time_point_results[tp][color]['medians']]
            plt.plot(positions, median_line, 'k-', lw=2, alpha=0.7)
            
        plt.title(color)
        plt.xlabel('Embryonic Day')
        plt.ylabel('Median Cells per Clone')
        plt.xticks(numeric_days, time_points)
        plt.grid(alpha=0.2)
    
    plt.suptitle(f'Distribution of Median Clone Sizes by Color - {root_folder_name}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path3 = os.path.join(root_output_folder, "median_clone_size_distribution.png")
    plt.savefig(plot_path3, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved median clone size distribution: {plot_path3}")
    
    
    # NEW PLOT 4: Mean evolution
    plt.figure(figsize=(12, 8))
    for color, color_name in color_map.items():
        mean_values = []
        #std_errors = []
        valid_days = []
        
        for tp in time_points:
            if color in time_point_results[tp] and time_point_results[tp][color]['means']:
                means = time_point_results[tp][color]['means']
                mean_value = np.mean(means)
                std_error = np.std(means) / np.sqrt(len(means))  # Standard error
                
                mean_values.append(mean_value)
                #std_errors.append(std_error)
                valid_days.append(int(tp[1:]))
        
        if mean_values:
            plt.errorbar(valid_days, mean_values, 
                         label=color, color=color_name, 
                         marker='o', markersize=8, capsize=5, lw=2) # + yerr=std_errors if needed
    
    plt.xlabel('Embryonic Day', fontsize=14)
    plt.ylabel('Mean Cells per Clone', fontsize=14)
    plt.title(f'Mean Clone Size Evolution by Color - {root_folder_name}', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.xticks(numeric_days, time_points, fontsize=12)
    plt.yticks(fontsize=12)
    plot_path4 = os.path.join(root_output_folder, "mean_clone_size_evolution.png")
    plt.savefig(plot_path4, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved mean clone size evolution: {plot_path4}")
    
    
def aggregate_clone_orientation(root_output_folder):
    """Collects clone orientation data and saves plots in each embryonic day folder"""
    # Dictionary to store orientation data by type, day and color
    orientation_data = {}
    
    # Walk through all subdirectories to find regionprops Excel files
    for root, dirs, files in os.walk(root_output_folder):
        excel_files = [f for f in files if f.endswith("_massive_cells_analysis.xlsx")]
        
        for excel_file in excel_files:
            excel_path = os.path.join(root, excel_file)
            
            # Extract type (parent folder of day folder) and day (EXX folder)
            path_parts = root.split(os.sep)
            try:
                # Find index of day folder (EXX)
                day_idx = next(i for i, part in enumerate(path_parts) if re.match(r'E\d+', part))
                day = path_parts[day_idx]
                # Type is the parent folder of the day folder
                type_name = path_parts[day_idx-1] if day_idx > 0 else "Unknown_Type"
            except (StopIteration, IndexError):
                continue
            
            # Initialize data structure
            if type_name not in orientation_data:
                orientation_data[type_name] = {}
            if day not in orientation_data[type_name]:
                orientation_data[type_name][day] = {}
            
            # Read all color sheets from Excel
            xls = pd.ExcelFile(excel_path)
            for sheet_name in xls.sheet_names:
                if sheet_name.endswith("_Massive_Cells"):
                    color = sheet_name.replace("_Massive_Cells", "")
                    df = pd.read_excel(xls, sheet_name=sheet_name)
                    
                    if 'orientation' in df.columns and 'individual_cells_count' in df.columns:
                        # Store both orientation and clone size
                        orientations = df['orientation'].dropna()
                        sizes = df['individual_cells_count']
                        orientation_data[type_name][day].setdefault(color, []).extend(
                            list(zip(orientations, sizes))
                        )
    
    # Define size categories
    size_categories = {
        "Small (2-4 cells)": (2, 4),
        "Medium (5-10 cells)": (5, 10),
        "Large (11+ cells)": (11, 1000)
    }
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

    # Generate plots for each type, day and color
    for type_name, days_data in orientation_data.items():
        for day, color_data in days_data.items():
            for color, data in color_data.items():
                if not data:
                    continue
                    
                # Create output folder for this type/day
                output_folder = os.path.join(root_output_folder, type_name, day, "Clone_Orientation_Aggregated")
                os.makedirs(output_folder, exist_ok=True)
                
                # Split data by size categories
                size_groups = {size_name: [] for size_name in size_categories}
                all_angles = []
                
                for angle, size in data:
                    for size_name, (min_size, max_size) in size_categories.items():
                        if min_size <= size <= max_size:
                            size_groups[size_name].append(angle)
                    all_angles.append(angle)
                
                size_groups["All Clones"] = all_angles
                
                # Create plot for this type/day/color
                fig = plt.figure(figsize=(14, 12))
                fig.suptitle(f'Clone Orientation - {type_name} - {day} - {color}', fontsize=16)
                
                # Create subplots
                ax1 = fig.add_subplot(221, projection='polar')
                ax2 = fig.add_subplot(222, projection='polar')
                ax3 = fig.add_subplot(223, projection='polar')
                ax4 = fig.add_subplot(224, projection='polar')
                axes = [ax1, ax2, ax3, ax4]
                
                plt.subplots_adjust(
                    left=0.05,
                    right=0.95,
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
                    ax.set_thetagrids(np.arange(0, 181, 45), 
                                     labels=['0°', '45°', '90°', '135°', '180°'])
                
                # Set titles
                ax1.set_title("Small Clones (2-4 cells)", pad=20)
                ax2.set_title("Medium Clones (5-10 cells)", pad=20)
                ax3.set_title("Large Clones (11+ cells)", pad=20)
                ax4.set_title("All Clones", pad=20)
                
                # Plot each size group
                for ax, (size_name, angles) in zip(axes, size_groups.items()):
                    if not angles:
                        ax.set_title(f"{size_name}\n(No Data)", color='red')
                        continue
                        
                    radians = np.deg2rad(angles)
                    n_bins = 18
                    bins = np.linspace(0, np.pi, n_bins + 1)
                    hist, bin_edges = np.histogram(radians, bins=bins)
                    hist = hist / len(angles)  # Convert to frequency
                    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                    widths = (bin_edges[1] - bin_edges[0]) * 0.8
                    
                    # Plot with color-specific bar color
                    ax.bar(bin_centers, hist, width=widths, bottom=0.0, 
                           color=color_map[color], alpha=0.7, edgecolor='black', 
                           linewidth=0.5)
                    
                    # Add statistics
                    mean_angle = np.mean(angles)
                    ax.text(np.pi/2, ax.get_rmax()*0.7, 
                            f"n={len(angles)}\nMean: {mean_angle:.1f}°", 
                            ha='center', va='center', fontsize=9, 
                            bbox=dict(facecolor='white', alpha=0.8, pad=3))
                
                # Save plot
                plot_filename = f"{type_name}_{day}_{color}_orientation.png"
                plot_path = os.path.join(output_folder, plot_filename)
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Saved orientation plot: {plot_path}")
                
                
def aggregate_color_frequency_vs_clone_size(root_output_folder):
    """Aggregates color frequency vs clone size data across all images"""
    data_by_day = {}

    # Walk through output folders
    for root, dirs, files in os.walk(root_output_folder):
        # Look for regionprops Excel files
        excel_files = [f for f in files if f.endswith("_massive_cells_analysis.xlsx")]

        for excel_file in excel_files:
            excel_path = os.path.join(root, excel_file)

            # Extract embryonic day from path
            day_match = re.search(r'E(\d+)', root)
            if not day_match:
                continue
            day = f"E{day_match.group(1)}"

            # Modify the CSV path construction
            day_folder = os.path.dirname(root)  # Parent of color_masks folder
            base_name = excel_file.replace("_massive_cells_analysis.xlsx", "")
            csv_path = os.path.join(day_folder, f"{base_name}_um.csv")  # Now in correct location

            if not os.path.exists(csv_path):
                continue

            try:
                # Load data
                df = pd.read_csv(csv_path)
                color_counts = df['Color'].value_counts()
                total_cells = len(df)

                # Get clone sizes from Excel
                clone_size_data = {}
                with pd.ExcelFile(excel_path) as xls:
                    for color in ['Red', 'Green', 'Blue', 'Yellow', 'Magenta', 'Cyan', 'White', 'Black']:
                        sheet_name = f"{color}_Massive_Cells"
                        if sheet_name in xls.sheet_names:
                            df_color = pd.read_excel(xls, sheet_name=sheet_name)
                            if 'individual_cells_count' in df_color.columns and len(df_color) > 0:
                                clone_size_data[color] = df_color['individual_cells_count'].mean()

                # Add to day data
                for color, size in clone_size_data.items():
                    if color in color_counts:
                        freq = color_counts[color] / total_cells * 100
                        data_by_day.setdefault(day, []).append({
                            'color': color,
                            'frequency': freq,
                            'size': size,
                            'image': base_name
                        })

            except Exception as e:
                print(f"Error processing {excel_file}: {str(e)}")

    # Create output folder
    output_folder = os.path.join(root_output_folder, "Color_Frequency_Analysis")
    os.makedirs(output_folder, exist_ok=True)

    # Plot per day
    color_map = {
        'Red': 'red', 'Green': 'green', 'Blue': 'blue',
        'Yellow': 'gold', 'Magenta': 'magenta',
        'Cyan': 'cyan', 'White': 'gray', 'Black': 'black'
    }

    for day, data in data_by_day.items():
        if not data:
            continue

        df_day = pd.DataFrame(data)

        plt.figure(figsize=(12, 8))
        for color in df_day['color'].unique():
            df_color = df_day[df_day['color'] == color]
            plt.scatter(
                df_color['frequency'],
                df_color['size'],
                s=100,
                color=color_map[color],
                edgecolor='black',
                label=color
            )

        plt.xlabel('Color Frequency (%)', fontsize=14)
        plt.ylabel('Average Clone Size (cells/clone)', fontsize=14)
        plt.title(f'Clone Size vs. Color Frequency - {day}', fontsize=16)
        plt.grid(alpha=0.3)
        plt.legend(title='Color Categories')

        # Add linear regression
        if len(df_day) > 1:
            x = df_day['frequency']
            y = df_day['size']
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            line_x = np.linspace(min(x)-5, max(x)+5, 100)
            line_y = slope * line_x + intercept
            plt.plot(line_x, line_y, 'k--', alpha=0.7)
            plt.text(0.05, 0.95,
                    f"R² = {r_value**2:.3f}\np = {p_value:.4f}",
                    transform=plt.gca().transAxes,
                    bbox=dict(facecolor='white', alpha=0.8))

        plot_path = os.path.join(output_folder, f"color_freq_vs_clone_size_{day}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved aggregate plot for {day}: {plot_path}")


def aggregate_clone_size_distribution_per_color(root_output_folder):
    """Aggregates clone size distributions per color across all images"""
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import defaultdict

    # Define size bins (same as in plot_clone_size_distribution_per_color)
    bin_edges = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 50, 1000]
    bin_labels = [
        '1', '2', '3', '4', '5', '6', '7', '8', '9', 
        '10-14', '15-19', '20-29', '30-49', '50+'
    ]
    
    # Data structure: color -> bin -> list of percentages
    color_data = defaultdict(lambda: defaultdict(list))
    color_count = defaultdict(int)  # Count of images per color

    # Walk through all subfolders to find regionprops Excel files
    for root, dirs, files in os.walk(root_output_folder):
        for file in files:
            if file.endswith("_massive_cells_analysis.xlsx"):
                excel_path = os.path.join(root, file)
                color_count['total'] += 1
                
                # Define the color categories
                color_categories = ['Red', 'Green', 'Blue', 'Yellow', 
                                   'Magenta', 'Cyan', 'White', 'Black']
                
                # Open the Excel file
                with pd.ExcelFile(excel_path) as xls:
                    for color in color_categories:
                        sheet_name = f"{color}_Massive_Cells"
                        if sheet_name in xls.sheet_names:
                            df = pd.read_excel(xls, sheet_name=sheet_name)
                            if not df.empty and 'individual_cells_count' in df.columns:
                                # Compute histogram
                                hist_counts, _ = np.histogram(
                                    df['individual_cells_count'], 
                                    bins=bin_edges
                                )
                                percentages = (hist_counts / len(df)) * 100
                                
                                # Store percentages
                                for i, bin_label in enumerate(bin_labels):
                                    color_data[color][bin_label].append(percentages[i])
                                
                                color_count[color] += 1

    # Only proceed if we have data
    if not color_data:
        print("No clone size data found for aggregation")
        return

    # Create output folder
    output_folder = os.path.join(root_output_folder, "Aggregated_Clone_Size_Distributions")
    os.makedirs(output_folder, exist_ok=True)

    # Plot for each color
    for color, bin_data in color_data.items():
        if color_count[color] < 1:
            continue
            
        plt.figure(figsize=(12, 8))
        
        # Calculate average and standard error for each bin
        avg_percent = []
        std_err = []
        for bin_label in bin_labels:
            values = bin_data[bin_label]
            avg = np.mean(values)
            avg_percent.append(avg)
            std_err.append(np.std(values) / np.sqrt(len(values)) if len(values) > 1 else 0)

        # Plot with error bars
        x_pos = np.arange(len(bin_labels))
        plt.bar(x_pos, avg_percent, yerr=std_err, capsize=5, 
                color=color.lower(), alpha=0.7, edgecolor='black')
        
        # Format plot
        plt.title(f'Aggregated Clone Size Distribution - {color}\n({color_count[color]} images)', fontsize=16)
        plt.xlabel('Cells per Clone', fontsize=14)
        plt.ylabel('Average Percentage of Clones (%)', fontsize=14)
        plt.xticks(x_pos, bin_labels, rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, val in enumerate(avg_percent):
            if val > 0:
                plt.text(i, val + 1, f'{val:.1f}%', ha='center', fontsize=10)

        # Save plot
        plot_path = os.path.join(output_folder, f"{color}_aggregated_clone_size_distribution.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved aggregated clone size distribution for {color}: {plot_path}")

    # Create combined plot for all colors
    plt.figure(figsize=(14, 10))
    color_map = {
        'Red': 'red', 'Green': 'green', 'Blue': 'blue',
        'Yellow': 'yellow', 'Magenta': 'magenta', 'Cyan': 'cyan',
        'White': 'gray', 'Black': 'black'
    }
    
    # Prepare data for combined plot
    combined_data = {}
    for color in color_data.keys():
        if color_count[color] > 0:
            combined_data[color] = [np.mean(color_data[color][bin_label]) for bin_label in bin_labels]
    
    # Plot settings
    bar_width = 0.1
    x_pos = np.arange(len(bin_labels))
    
    for i, (color, percentages) in enumerate(combined_data.items()):
        offset = bar_width * (i - len(combined_data)/2)
        plt.bar(x_pos + offset, percentages, bar_width, 
                color=color_map[color], label=color, alpha=0.7)
    
    plt.title('Aggregated Clone Size Distribution by Color', fontsize=16)
    plt.xlabel('Cells per Clone', fontsize=14)
    plt.ylabel('Average Percentage of Clones (%)', fontsize=14)
    plt.xticks(x_pos, bin_labels, rotation=45)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save combined plot
    plot_path = os.path.join(output_folder, "combined_clone_size_distribution.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved combined aggregated clone size distribution: {plot_path}")
    
    
def aggregate_percentage_cells_in_clone_bins_per_color(root_output_folder):
    """
    Aggregates the percentage of cells in clone bins for each color separately across all images.
    
    Parameters:
    - root_output_folder (str): Root folder containing all processed images
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    from collections import defaultdict

    # Define size bins
    bin_edges = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 50, 1000]
    bin_labels = [
        '1', '2', '3', '4', '5', '6', '7', '8', '9', 
        '10-14', '15-19', '20-29', '30-49', '50+'
    ]

    # Define color categories
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

    # Dictionary to store data from all images for each color
    all_color_data = {color: defaultdict(list) for color in color_categories}

    # Walk through all subfolders to find regionprops Excel files
    for root, dirs, files in os.walk(root_output_folder):
        for file in files:
            if file.endswith("_massive_cells_analysis.xlsx"):
                excel_path = os.path.join(root, file)
                
                # Process this image for each color
                with pd.ExcelFile(excel_path) as xls:
                    for color in color_categories:
                        sheet_name = f"{color}_Massive_Cells"
                        if sheet_name in xls.sheet_names:
                            df = pd.read_excel(xls, sheet_name=sheet_name)
                            if not df.empty and 'individual_cells_count' in df.columns:
                                # Calculate percentages for this color in this image
                                bin_cell_counts = defaultdict(int)
                                total_cells = 0
                                
                                # For each clone, add its cell count to the appropriate bin
                                for _, row in df.iterrows():
                                    clone_size = row['individual_cells_count']
                                    total_cells += clone_size
                                    
                                    # Find the appropriate bin
                                    for i in range(len(bin_edges) - 1):
                                        if bin_edges[i] <= clone_size < bin_edges[i+1]:
                                            bin_cell_counts[bin_labels[i]] += clone_size
                                            break
                                    else:
                                        # Handle clones larger than the last bin edge
                                        bin_cell_counts[bin_labels[-1]] += clone_size
                                
                                # Calculate percentages for this color
                                if total_cells > 0:
                                    for bin_label in bin_labels:
                                        if bin_label in bin_cell_counts:
                                            percentage = (bin_cell_counts[bin_label] / total_cells) * 100
                                            all_color_data[color][bin_label].append(percentage)
                                        else:
                                            all_color_data[color][bin_label].append(0)

    # Create output folder
    output_folder = os.path.join(root_output_folder, "Aggregated_Cell_Percentage_In_Clone_Bins_Per_Color")
    os.makedirs(output_folder, exist_ok=True)

    # Create a figure with subplots for each color
    fig, axes = plt.subplots(2, 4, figsize=(20, 12))
    axes = axes.flatten()
    
    for i, color in enumerate(color_categories):
        ax = axes[i]
        
        if all_color_data[color] and any(all_color_data[color].values()):
            # Calculate average and standard error for each bin for this color
            avg_percent = []
            std_err = []
            for bin_label in bin_labels:
                values = all_color_data[color][bin_label]
                if values:
                    avg = np.mean(values)
                    avg_percent.append(avg)
                    std_err.append(np.std(values) / np.sqrt(len(values)) if len(values) > 1 else 0)
                else:
                    avg_percent.append(0)
                    std_err.append(0)
            
            # Plot with error bars
            x_pos = np.arange(len(bin_labels))
            ax.bar(x_pos, avg_percent, yerr=std_err, capsize=5, 
                   color=color_map[color], alpha=0.7, edgecolor='black')
            
            # Format subplot
            ax.set_title(f'{color} Cells', fontsize=12)
            ax.set_xlabel('Clone Size (cells per clone)', fontsize=10)
            ax.set_ylabel('Percentage of Cells (%)', fontsize=10)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(bin_labels, rotation=45, fontsize=8)
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for j, val in enumerate(avg_percent):
                if val > 1:  # Only label significant values
                    ax.text(j, val + 0.5, f'{val:.1f}%', ha='center', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{color} Cells', fontsize=12)
    
    # Set main title
    fig.suptitle('Aggregated Percentage of Cells in Clone Size Bins by Color', fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save plot
    plot_path = os.path.join(output_folder, "aggregated_percentage_cells_in_clone_bins_per_color.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved aggregated percentage cells in clone bins per color plot: {plot_path}")


def process_root_folder(root_input_folder, root_output_folder):
    """Process all subfolders within a root directory"""
    os.makedirs(root_output_folder, exist_ok=True)
    print(f"📁 Processing root folder: {root_input_folder}")
    
    # Walk through all subdirectories
    for dirpath, dirnames, filenames in os.walk(root_input_folder):
        # Skip directories without image files
        if not any(f.lower().endswith('.tif') for f in filenames):
            continue
            
        # Create corresponding output path
        rel_path = os.path.relpath(dirpath, root_input_folder)
        output_folder = os.path.join(root_output_folder, rel_path)
        
        print(f"\n🔍 Processing subfolder: {rel_path}")
        process_folder(dirpath, output_folder)
        
    aggregate_clone_sizes(root_output_folder)
    aggregate_clone_orientation(root_output_folder)
    aggregate_color_frequency_vs_clone_size(root_output_folder)
    aggregate_clone_size_distribution_per_color(root_output_folder)
    aggregate_percentage_cells_in_clone_bins_per_color(root_output_folder)
        
if __name__ == '__main__':
    input_root = r'C:\Users\33672\Documents\Stage X\test'
    output_root = r'C:\Users\33672\Documents\Stage X\test'
    process_root_folder(input_root, output_root)

