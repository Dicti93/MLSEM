import os
import glob
import numpy as np
import re
import pandas as pd

from pathlib import Path

import cv2
from shutil import rmtree


# ## Testing Split Integrity

# This function ensures the integrity of the data split between the training and test datasets. It verifies that there are no overlapping patients between the two sets, which is critical for unbiased model evaluation.

# ### Function Details:

# - **Input Paths**: Receives paths for training and test image directories.
# - **Extract Patient Codes**: Parses the filenames to extract unique patient identifiers, assuming the identifier is the first part of the filename before a delimiter.
# - **Check Uniqueness**: Compares the list of patient codes from both directories to ensure there is no overlap.
# - **Outputs**:
#   - The number of unique patients and the total number of images in both the training and test sets.
#   - Confirmation that there are no shared patients between the sets if the condition is met, ensuring that the training and test data are indeed disjoint.

# This verification helps prevent data leakage and ensures the model is tested on completely unseen data.

def extract_patient_codes(image_dir):
    """Extract and convert patient codes from image filenames in the given directory."""
    image_file_list = [f for f in glob.glob(os.path.join(image_dir, '*')) if os.path.isfile(f)]
    patient_codes = []
    for file in image_file_list:
        name = os.path.basename(file)
        patient_code = re.split('_| |\.', name)[0]
        if patient_code.isdigit():
            patient_codes.append(patient_code)
    return np.array(list(map(int, patient_codes)), dtype=np.int32)

def check_for_overlap(train_codes, test_codes):
    """Check for any overlap in patient codes between training and testing datasets."""
    unique_train = np.unique(train_codes)
    unique_test = np.unique(test_codes)
    print("\nNumber of patients in train set: ", len(unique_train))
    print("Number of patients in test set: ", len(unique_test))
    print("\nNumber of images in train set: ", len(train_codes))
    print("Number of images in test set: ", len(test_codes))
    
    if len(np.intersect1d(unique_train, unique_test)) == 0:
        print("\nPatients in train and test set don't overlap!")
    else:
        print("\nOverlap detected in patients between train and test sets!")

def test_split(train_image_dir, test_image_dir):
    """Verifies that the training and test sets form disjoint sets of subjects."""
    try:
        train_groups = extract_patient_codes(train_image_dir)
        test_groups = extract_patient_codes(test_image_dir)
        check_for_overlap(train_groups, test_groups)
    except ValueError as e:
        print(f"Error processing patient codes: {str(e)}")



## Renaming Image Files for Standardization

# This script section defines and applies a function to standardize the naming and format of ultrasound image files within specified directories.

# ### Functionality:

# - **Target Directories**: Processes images in both training (`TRAIN_IMAGE_DIR`) and testing (`TEST_IMAGE_DIR`) directories.
# - **File Processing**:
#   - Iterates over each `.bmp` file found in the specified directories.
#   - Standardizes filenames by replacing spaces with underscores and converting file formats from `.bmp` to `.png`.
#   - Renames files accordingly to ensure consistency across the dataset.
# - **Execution**: The renaming function is applied to each directory listed in the `directories` array, ensuring all relevant images are uniformly formatted.

def rename_files_in_directory(image_dir, src_ext="*.bmp", target_ext=".png"):
    dir_path = Path(image_dir)
    for file in dir_path.glob(src_ext):
        try:
            new_name = file.stem.replace(" ", "_") + target_ext
            new_file = file.with_name(new_name)
            file.rename(new_file)
        except Exception as e:
            print(f"Error renaming {file}: {e}")
    
    print(f"File renaming complete in {image_dir}.")


    

# ## Image Preprocessing for Appendicitis Ultrasound Images

# This script is designed to automate the preprocessing of ultrasound images of appendicitis. The main goals are to identify and crop the relevant part of each image and to resize it to a standard dimension. This standardization is crucial for consistent analysis in subsequent processing steps such as feature extraction or machine learning modeling.

# ### Process Overview

# 1. **Set Thresholds and Target Dimensions**: 
#    - `FOREGROUND_THRESHOLD`: This determines the minimum percentage of foreground pixels (pixels above a certain intensity) required for a row to be considered as containing relevant content.
#    - `SHADE_THRESHOLD`: This is the intensity above which a pixel is considered part of the foreground.
#    - `TARGET_DIMS`: The dimensions to which all images will be resized after cropping.

# 2. **Define Functions**:
#    - `find_image_border()`: Calculates the boundary of the area containing the relevant image content by evaluating pixel intensity across rows and columns.
#    - `preprocess_images()`: Handles the reading, processing, and saving of images. This includes reading images from a specified directory, applying the `find_image_border` function to determine the crop area, resizing the cropped area to the target dimensions, and saving the processed image to a new directory.

# 3. **Execution**:
#    - For each image in the specified directory, the script performs cropping based on calculated borders and resizes the image to predefined dimensions. The processed images are then saved in a separate directory within the original directory to ensure that the output is organized and easily accessible for future use.

# This preprocessing routine is specifically tuned for ultrasound images of appendicitis, focusing on isolating the region of interest (ROI) and standardizing the image size for any further analysis.

def find_image_border(img, threshold, shade_threshold, buffer=10):
    """Finds the border of the image based on foreground intensity, with added buffer for flexibility."""
    k = 0
    height, width = img.shape[:2]
    for row_idx, row in enumerate(img):
        foreground = np.sum(row > shade_threshold)
        if foreground / width > threshold:
            if k == 0:
                row_idx_old = row_idx
            k += 1
            if k > 20:  # Continue until we have 20 consecutive rows above the threshold
                return max(0, row_idx_old - buffer)  # Subtract buffer but prevent negative index
        else:
            k = 0
    return height

def crop_and_resize_image(img, threshold, shade_threshold, target_dims, buffer=10):
    """Crops and resizes the image based on the specified thresholds and dimensions with buffer zones."""
    top = find_image_border(img, threshold, shade_threshold, buffer)
    bottom = find_image_border(cv2.flip(img, 0), threshold, shade_threshold, buffer)
    bottom = img.shape[0] - bottom

    rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    left = find_image_border(rotated_img, threshold, shade_threshold, buffer)
    right = find_image_border(cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), threshold, shade_threshold, buffer)
    right = img.shape[1] - right

    cropped = img[top:bottom, left:right]
    resized = cv2.resize(cropped, target_dims)
    return resized

def process_images_in_directory(source_dir, target_dir, threshold, shade_threshold, target_dims, buffer=10):
    """Processes all images in the specified directory based on given parameters and saves them to the target directory."""
    target_path = Path(target_dir)
    if target_path.exists() and target_path.is_dir():
        rmtree(target_path)
    os.makedirs(target_path, exist_ok=True)

    for image_file in Path(source_dir).glob("*.png"):
        img = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        resized_img = crop_and_resize_image(img, threshold, shade_threshold, target_dims, buffer)
        cv2.imwrite(str(target_path / image_file.name), resized_img)

    print(f"Processed images are saved in {target_dir}")


# Preprocessing 'US_Number' in DataFrame:
# This section includes functions to clean and standardize the 'US_Number' column of a DataFrame.
# It first converts the 'US_Number' entries to numeric, handling any 'nan' strings and true NaN values by converting them to NaN and then dropping these rows.
# It then ensures all numbers are integers. Additionally, there is a function to convert the column to a string type if required.
# These steps help in maintaining data integrity and preparing the column for further analysis or processing.


def preprocess_us_number(df, us_number_col='US_Number'):
    """
    Preprocess the US_Number column in the DataFrame to ensure it contains only integers.

    Args:
    df (pd.DataFrame): DataFrame containing the US_Number column.
    us_number_col (str): Column name for US numbers.

    Returns:
    pd.DataFrame: The DataFrame with the preprocessed US_Number column.
    """
    # Drop rows with NaN values in the US_Number column, handling both 'nan' strings and true NaN values
    df[us_number_col] = pd.to_numeric(df[us_number_col], errors='coerce')  # Convert to numeric, coercing errors to NaN
    df = df.dropna(subset=[us_number_col])  # Drop rows with NaN values
    
    # Convert US_Number to integer
    df[us_number_col] = df[us_number_col].astype(int)

    return df

def convert_column_to_string(df, column_name):
    """
    Convert the specified column of a DataFrame to string type.

    Args:
    df (pd.DataFrame): The DataFrame containing the column.
    column_name (str): The name of the column to convert.

    Returns:
    pd.DataFrame: The DataFrame with the column converted to string type.
    """
    df[column_name] = df[column_name].astype(str)
    return df



# Image Label CSV Creation:
# This section of code modularly handles the extraction of labels and other pertinent details from images located in a specified directory,
# comparing them against a DataFrame that includes metadata such as Diagnosis, Severity, and Management. The data extraction function not 
# only gathers necessary image information but also tracks missing US_Numbers dynamically, enhancing data integrity checks. Subsequently, the 
# extracted information is compiled into a DataFrame and either appended to or used to create a CSV file, mapping image filenames to their 
# respective labels. This process is segmented into distinct functions: one for data extraction and missing count, and another for writing this 
# data to a CSV file. Separating these tasks into dedicated functions enhances the code's flexibility, maintainability, and testability, while 
# also providing clear, trackable logging of data discrepancies.

# Global dictionary to track missing US_Numbers
missing_counts = {
    'train': 0,
    'test': 0
}

def extract_image_labels(image_dir, df, file_extension='*.png'):
    """
    Extract labels from image filenames based on a DataFrame and count missing entries.

    Args:
    image_dir (str): Directory containing the images.
    df (pd.DataFrame): DataFrame with image identifiers and labels.
    file_extension (str): The extension of the files to look for.

    Returns:
    list of dicts: List containing the records for each image found.
    """

    records = []
    unlabeled_images = []  # List to track unlabeled image paths
    directory_key = 'train' if 'train' in str(image_dir).lower() else 'test'
    missing_counts[directory_key] = 0  # Reset count for each run

    for image_file in Path(image_dir).glob(file_extension):
        us_number = image_file.stem.split('_')[0].split('.')[0]
        if us_number in df['US_Number'].values:
            row = df[df['US_Number'] == us_number]
            records.append({
                'filename': image_file.name,
                'label': row['Diagnosis'].values[0],
                'severity': row['Severity'].values[0],
                'management': row['Management'].values[0]
            })
        else:
            print(f"US_Number {us_number} not found in DataFrame.")
            missing_counts[directory_key] += 1
            unlabeled_images.append(image_file)  # Add path to the list

    return records, unlabeled_images


def write_records_to_csv(records, csv_filename):
    """
    Write records to a CSV file.

    Args:
    records (list of dicts): Records to write to the CSV.
    csv_filename (str): Filename for the resulting CSV.
    """
    label_df = pd.DataFrame(records)
    label_df.to_csv(csv_filename, mode='a', header=not Path(csv_filename).exists(), index=False)
    print(f"CSV file updated: {csv_filename}")

def process_images_and_create_csv(image_dir, df, csv_filename='image_labels.csv'):
    """
    Process images in a directory and map their filenames to labels in a CSV file.

    Args:
    image_dir (str): Directory containing the images.
    df (pd.DataFrame): DataFrame with image identifiers and labels.
    csv_filename (str): Filename for the resulting CSV.
    """
    directory_key = 'train' if 'train' in str(image_dir).lower() else 'test'
    records, unlabeled_images = extract_image_labels(image_dir, df)
    write_records_to_csv(records, csv_filename)

    # Delete unlabeled images
    for image_path in unlabeled_images:
        os.remove(image_path)
        print(f"Deleted unlabeled image: {image_path}")

    print(f"Total missing US_Number in {Path(image_dir).name}: {missing_counts[directory_key]}")