# Seminar Machine Learning in Applied Settings

This repository contains the seminar project on appendicitis classification. All scripts should be executed in Google Colab.

## Folder Structure and Contents

### Seminar Introduction Files
- **ML_Seminar_Presenation.pdf**: Introduction Presentation to the seminar.
- **Seminar_ML_Syllabus2024.pdf**: Seminar Syllabus.

### Data and Description
- **data.csv**: Dataset for appendicitis classification.
- **Pediatric Appendicitis_ Dataset & Summary.xlsx**: Description of the data fields.

### Data Preparation
- **Data_Prep.ipynb**: This script processes the dataset by cleaning and imputing missing values, converts categorical variables to numeric, visualizes missing data, and prepares the dataset for further analysis or modeling in the context of appendicitis classification.
- **clean_data.csv**: New cleaned Dataset.

### EDA (Exploratory Data Analysis)
- **EDA.ipynb**: This script processes the cleaned dataset by defining feature groups, describes the demographic variables, performs correlation and feature importance analysis, and uses mutual information and PCA for supervised and unsupervised feature selection, respectively, to prepare the data for modeling in the context of appendicitis classification.

### Modelling
- **Modelling.ipynb**: This script, defines feature sets, implements machine learning models (Logistic Regression, Naive Bayes, SVM, Random Forest, Gradient Boosting, Neural Network), evaluates these models using cross-validation, and plots the results for comparison in the context of appendicitis classification.

### CNN (Convolutional Neural Networks)
- **CNN.ipynb**: This script trains a Convolutional Neural Network (CNN) to diagnose appendicitis by processing sequences of ultrasound images grouped by patient identification numbers.
- **SimpleCNN.ipynb**: This script trains a Convolutional Neural Network (CNN) to diagnose appendicitis by processing individual ultrasound images independently.

## Usage Instructions

1. Clone the repository: `git clone https://github.com/Dicti93/MLSEM.git`
2. Upload the desired scripts to Google Colab and execute.
