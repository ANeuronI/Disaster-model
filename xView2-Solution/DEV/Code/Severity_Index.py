import cv2
import numpy as np
import os
import csv
import streamlit as st

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


def Severity_Index_Predict(pre_image_mask,New_post_image_mask ):
    
    # Load the pre-disaster and result image (the overlaid image)
    pre_disaster_img = cv2.imread(pre_image_mask)
    result_img = cv2.imread(New_post_image_mask) 
    
    pre_disaster_img = np.array(pre_disaster_img)
    result_img = np.array(result_img)
    
    pre_green_mask = (pre_disaster_img == class_colors["no_damage"]).all(axis=-1)
    total_building_area = np.sum(pre_green_mask)
    # Reset damage areas for each image pair
    for damage_type in damage_areas.keys():
        damage_areas[damage_type] = 0
    for damage_type, color in class_colors.items():
        if damage_type != "background":  
            mask = (result_img[:, :, 0] == color[0]) & (result_img[:, :, 1] == color[1]) & (result_img[:, :, 2] == color[2])
            damage_areas[damage_type] = np.sum(mask)
    
    severity_index = sum(weights[damage] * damage_areas[damage] for damage in damage_areas) / total_building_area if total_building_area > 0 else 0
    
    # print(f"Processing pre and post disaster files with ID: {unique_id}")
    
    # print(f"Total Building Area: {total_building_area}")
    # for damage_type, area in damage_areas.items():
    #         print(f"Area of {damage_type}: {area}")
    
    # print(f"Severity Index: {severity_index}\n")
    
    return severity_index, damage_areas, total_building_area


def display_damage_table(severity_index, damage_areas, total_building_area):
    
    st.markdown("<h4 style='text-align: center;'>Damaged Prediction:</h4>", unsafe_allow_html=True)
    st.write("""
    <table style="font-size: 20px; width: 80%; margin-left:auto; margin-right:auto; border-collapse: collapse;">
    <tr>
    <th style="border: 1px solid black; padding: 8px;">Damaged Type</th>
    <th style="border: 1px solid black; padding: 8px;">Area</th>
    </tr>
    <tr>
    <td style="border: 1px solid black; padding: 8px;">Total Building Pixel</td>
    <td style="border: 1px solid black; padding: 8px;">{:.2f}</td>
    </tr>
    <tr>
    <td style="border: 1px solid black; padding: 8px;">No Damage Pixel</td>
    <td style="border: 1px solid black; padding: 8px;">{:.2f}</td>
    </tr>
    <tr>
    <td style="border: 1px solid black; padding: 8px;">Light Damage Pixel</td>
    <td style="border: 1px solid black; padding: 8px;">{:.2f}</td>
    </tr>
    <tr>
    <td style="border: 1px solid black; padding: 8px;">Major Damage Pixel</td>
    <td style="border: 1px solid black; padding: 8px;">{:.2f}</td>
    </tr>
    <tr>
    <td style="border: 1px solid black; padding: 8px;">Fully Destroyed Pixel</td>
    <td style="border: 1px solid black; padding: 8px;">{:.2f}</td>
    </tr>
    </table>
    """.format(
        total_building_area,
        damage_areas['no_damage'],
        damage_areas['light_damage'],
        damage_areas['major_damage'],
        damage_areas['fully_destroyed']
    ), unsafe_allow_html=True)
    
    st.markdown("<h4 style='text-align: center;'>Severity Index</h4>", unsafe_allow_html=True)
    st.write("""
    <table style="font-size: 20px; width: 80%; margin-left:auto; margin-right:auto; border-collapse: collapse;">
    <tr>
    <td style="border: 1px solid black; padding: 8px;">Severity Index</td>
    <td style="border: 1px solid black; padding: 8px;">{:.5f}</td>
    </tr>
    </table>
    """.format(
        severity_index
    ), unsafe_allow_html=True)
    
