import cv2
import numpy as np
import os
import csv

# Class colors corresponding to damage levels
class_colors = {
    "background": (0, 0, 0),  # Black
    "no_damage": (0, 255, 0),  # Green
    "light_damage": (0, 255, 255),  # Yellow
    "major_damage": (0, 128, 255),  # Orange
    "fully_destroyed": (0, 0, 255)  # Red
}

# Initialize counters for each damage level
damage_areas = {
    "no_damage": 0,
    "light_damage": 0,
    "major_damage": 0,
    "fully_destroyed": 0
}

# Calculate the severity index (weighted average)
weights = {
    "no_damage": 0,
    "light_damage": 1,
    "major_damage": 2,
    "fully_destroyed": 3
}

# Define the directories
pre_disaster_dir = os.path.join('models', 'predict_37_weighted_dominantv2', 'localization')
post_disaster_dir = os.path.join('DEV', 'output', 'output_mask')

# Get list of pre-disaster image filenames
pre_disaster_files = [f for f in os.listdir(pre_disaster_dir) if f.endswith('_pre_disaster_prediction.png')]

# Prepare to save results to CSV
csv_file_path = os.path.join('DEV', 'output', 'severity_index.csv')
with open(csv_file_path, mode='w', newline='') as csvfile:
    fieldnames = ['Pre-Disaster Image', 'Post-Disaster Image', 'Total Building Area',
                  'No Damage Area', 'Light Damage Area', 'Major Damage Area',
                  'Fully Destroyed Area', 'Severity Index']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # Write the header to the CSV file
    writer.writeheader()

    # Process each pre-disaster image
    for pre_disaster_file in pre_disaster_files:
        # Extract the unique ID from the filename
        unique_id = pre_disaster_file.split('_')[2]  # This assumes the format is as specified

        # Construct the corresponding post-disaster filename
        post_disaster_file = f'test_damage_{unique_id}_post_disaster_prediction.png'
        post_disaster_path = os.path.join(post_disaster_dir, post_disaster_file)

        # Check if the corresponding post-disaster image exists
        if not os.path.exists(post_disaster_path):
            print(f"Post-disaster image not found for ID {unique_id}: {post_disaster_path}")
            continue

        # Load the pre-disaster image
        pre_disaster_img_path = os.path.join(pre_disaster_dir, pre_disaster_file)  # Pre-disaster path
        pre_disaster_img = cv2.imread(pre_disaster_img_path)

        # Load the post-disaster image (the overlaid image)
        result_img = cv2.imread(post_disaster_path)

        # Check if the images are loaded properly
        if pre_disaster_img is None:
            print(f"Error loading pre-disaster image: {pre_disaster_img_path}")
            continue
        if result_img is None:
            print(f"Error loading post-disaster image: {post_disaster_path}")
            continue

        # Convert images to numpy arrays for pixel-wise comparison
        pre_disaster_img = np.array(pre_disaster_img)
        result_img = np.array(result_img)

        # Get the total building area from the pre-disaster image (where there is green)
        pre_green_mask = (pre_disaster_img == class_colors["no_damage"]).all(axis=-1)
        total_building_area = np.sum(pre_green_mask)

        # Reset damage areas for each image pair
        for damage_type in damage_areas.keys():
            damage_areas[damage_type] = 0

        # Color thresholding for each damage type using exact matching
        for damage_type, color in class_colors.items():
            if damage_type != "background":  # Skip the background
                # Create a mask for each damage type using np.all for exact color matches
                mask = (result_img[:, :, 0] == color[0]) & (result_img[:, :, 1] == color[1]) & (result_img[:, :, 2] == color[2])
                damage_areas[damage_type] = np.sum(mask)

        # Calculate severity index
        severity_index = sum(weights[damage] * damage_areas[damage] for damage in damage_areas) / total_building_area if total_building_area > 0 else 0

        # Print out the results for each image pair
        print(f"Processing pre and post disaster files with ID: {unique_id}")
        # print(f"Total Building Area: {total_building_area}")
        # for damage_type, area in damage_areas.items():
        #     print(f"Area of {damage_type}: {area}")

        print(f"Severity Index: {severity_index}\n")

        # Save the results to the CSV file
        writer.writerow({
            'Pre-Disaster Image': pre_disaster_file,
            'Post-Disaster Image': post_disaster_file,
            'Total Building Area': total_building_area,
            'No Damage Area': damage_areas['no_damage'],
            'Light Damage Area': damage_areas['light_damage'],
            'Major Damage Area': damage_areas['major_damage'],
            'Fully Destroyed Area': damage_areas['fully_destroyed'],
            'Severity Index': severity_index
        })

print(f"Results saved to {csv_file_path}")
