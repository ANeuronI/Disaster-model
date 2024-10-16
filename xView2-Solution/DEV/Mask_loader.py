import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Define the directories
pre_disaster_dir = os.path.join('models', 'predict_37_weighted_dominantv2', 'localization')
post_disaster_dir = os.path.join('models', 'predict_37_weighted_dominantv2', 'damage')
save_folder = os.path.join('DEV', 'output', 'output_mask')

# Create the output folder if it doesn't exist
os.makedirs(save_folder, exist_ok=True)

# Get a list of all pre-disaster image files
pre_disaster_files = [f for f in os.listdir(pre_disaster_dir) if f.endswith('_pre_disaster_prediction.png')]

# Loop through each pre-disaster image
for pre_disaster_file in pre_disaster_files:
    # Extract the unique ID from the filename
    unique_id = pre_disaster_file.split('_')[2]  # Assumes filename format is consistent

    # Construct the corresponding post-disaster filename
    post_disaster_file = f'test_damage_{unique_id}_pre_disaster_prediction.png'
    post_disaster_img_path = os.path.join(post_disaster_dir, post_disaster_file)

    # Check if the corresponding post-disaster image exists
    if not os.path.exists(post_disaster_img_path):
        print(f"Post-disaster image not found for ID {unique_id}: {post_disaster_img_path}")
        continue

    # Load the pre-disaster image
    pre_disaster_img_path = os.path.join(pre_disaster_dir, pre_disaster_file)
    pre_disaster_img = cv2.imread(pre_disaster_img_path)

    # Load the post-disaster image
    post_disaster_img = cv2.imread(post_disaster_img_path)

    # Convert images to HSV to isolate the green color for buildings
    hsv_pre = cv2.cvtColor(pre_disaster_img, cv2.COLOR_BGR2HSV)
    hsv_post = cv2.cvtColor(post_disaster_img, cv2.COLOR_BGR2HSV)

    # Define the exact green color range (#00ff00 = rgb(0, 255, 0))
    lower_green = np.array([60, 255, 255])  # exact green
    upper_green = np.array([60, 255, 255])

    # Create a mask for green color (undamaged buildings) in pre-disaster image
    pre_disaster_mask = cv2.inRange(hsv_pre, lower_green, upper_green)

    # Extract the regions from the post-disaster image where the buildings are located in pre-disaster
    post_disaster_buildings = cv2.bitwise_and(post_disaster_img, post_disaster_img, mask=pre_disaster_mask)

    # Prepare the result image
    result_img = pre_disaster_img.copy()

    # Where there is a green mask (undamaged buildings), replace with the post-disaster colors
    result_img[pre_disaster_mask > 0] = post_disaster_buildings[pre_disaster_mask > 0]

    # Modify the pre-disaster filename for saving the result image
    post_filename = post_disaster_file.replace("_pre_", "_post_")  # Replace "pre" with "post"

    # Save the result image
    output_path = os.path.join(save_folder, post_filename)
    cv2.imwrite(output_path, result_img)
    print(f"Result image saved at {output_path}")