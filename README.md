# Seminar Machine Learning in Applied Settings

This repository contains the seminar project on appendicitis classification. All scripts should be executed in Google Colab.

**Note**: Running the CNN scripts (`CNN.ipynb` and `SimpleCNN.ipynb`) may require substantial computational resources. It is recommended to have access to high-performance GPUs or TPUs for efficient execution.

## Folder Structure and Contents

### Seminar Introduction Files
- **ML_Seminar_Presenation.pdf**: Introduction Presentation to the seminar.
- **Seminar_ML_Syllabus2024.pdf**: Seminar Syllabus.

### Data and Description
- **data.csv**: Dataset for appendicitis classification.
- **Pediatric Appendicitis_ Dataset & Summary.xlsx**: Description of the data fields.

### Data Preparation
- **Data_Prep.ipynb**: 
  - Processes the dataset by cleaning and imputing missing values.
  - Converts categorical variables to numeric.
  - Visualizes missing data.
  - Prepares the dataset for further analysis or modeling in the context of appendicitis classification.
- **clean_data.csv**: New cleaned Dataset.

### EDA (Exploratory Data Analysis)
- **EDA.ipynb**: 
  - Processes the cleaned dataset by defining feature groups.
  - Describes the demographic variables.
  - Performs correlation and feature importance analysis.
  - Uses mutual information and PCA for supervised and unsupervised feature selection, respectively, to prepare the data for modeling in the context of appendicitis classification.

### Modelling
- **Modelling.ipynb**: 
  - Defines feature sets.
  - Implements machine learning models (Logistic Regression, Naive Bayes, SVM, Random Forest, Gradient Boosting, Neural Network).
  - Evaluates these models using cross-validation.
  - Plots the results for comparison in the context of appendicitis classification.

### CNN (Convolutional Neural Networks)
- **CNN.ipynb**: 
  - Trains a Convolutional Neural Network (CNN) to diagnose appendicitis by processing sequences of ultrasound images grouped by patient identification numbers.
  - Note: Executing this script may require significant computational resources.
- **SimpleCNN.ipynb**: 
  - Trains a Convolutional Neural Network (CNN) to diagnose appendicitis by processing individual ultrasound images independently.
  - Note: Executing this script may require significant computational resources.

## Usage Instructions

1. Clone the repository: `git clone https://github.com/Dicti93/MLSEM.git`
2. Upload the desired scripts to Google Colab and execute.

