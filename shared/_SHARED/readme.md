# Task Overview

## 1. Creation of `severity_label.csv`

The dataset consists of two folders: one containing images and the other containing their respective labels. The goal is to create a `severity_label.csv` file to train the model and establish a general structure for the model architecture.

### Steps to Follow:

- **Generate `severity_label.csv`:** Utilize the labels from the label folder and `train.csv` to create the `severity_label.csv` file.
- **Feature Analysis:** `train.csv` contains various features. It's crucial to understand how these features, such as pixel damage, damage index are calculated using image labels.
- **Recreate `train.csv`:** Use the given 3 sets of images and labels uploaded in the data folder to replicate `train.csv`, enabling the creation of a comprehensive `severity_label.csv` for the entire dataset.
- **Reporting:** Upon successful completion, provide a report detailing how these values were determined.
- **File Consistency:** Ensure the features remain consistent with the original file. Upload the completed `severity_label.csv` to the 'report' folder.

**Note:** There are two training files: `train.csv` and `train_folds.csv`. Focus on creating `train.csv`. The `train_folds.csv` is generated after applying an algorithm called 'k-folds', with a corresponding script named 'make_folds' in the repository.

## 2. Analysis of Model Architecture and Code Structure

The objective is to understand the contents of this [repository](https://github.com/BloodAxe/xView2-Solution), which includes data augmentation, CSV creation, and the model structure.

### Areas of Focus:

1. **Model Creation:** Comprehend how the model was developed.
2. **Loss Function Initialization:** Understand the initialization process of the loss function.
3. **Data Preparation:** Analyze how data is prepared for the model.
4. **Feature Utilization:** Identify the features being passed to the model.

**Important:** Determine where in the repository these features are initialized and utilized. Upload the findings in the 'report' folder.

## **new severity_index.py and making_the_post_image_with_pre.py push** :

use 'making_the_post_image_with_pre.py' first before 'severity_index.py' and the result of the first file is the image which will be given as input to severity_index.py as post image to create the prediction.