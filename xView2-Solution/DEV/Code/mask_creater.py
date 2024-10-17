import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

save_folder =  os.path.join('DEV', 'output', 'predicted_weight_dominant_v2', 'output_mask')
os.makedirs(save_folder, exist_ok=True)

save_folder_overlay =  os.path.join('DEV', 'output', 'predicted_weight_dominant_v2', 'Overlay')
os.makedirs(save_folder_overlay, exist_ok=True)

def mask_creater(pre_image_mask, post_image_mask):
    
    pre_disaster_img = cv2.imread(pre_image_mask)
    post_disaster_img = cv2.imread(post_image_mask)
    
    # Convert images to HSV to isolate the green color for buildings
    hsv_pre = cv2.cvtColor(pre_disaster_img, cv2.COLOR_BGR2HSV)
    hsv_post = cv2.cvtColor(post_disaster_img, cv2.COLOR_BGR2HSV)

    # Define the exact green color range (#00ff00 = rgb(0, 255, 0))
    lower_green = np.array([60, 255, 255])  # exact green
    upper_green = np.array([60, 255, 255])
    
    pre_disaster_mask = cv2.inRange(hsv_pre, lower_green, upper_green)
    post_disaster_buildings = cv2.bitwise_and(post_disaster_img, post_disaster_img, mask=pre_disaster_mask)
    
    # Prepare the result image
    result_img = pre_disaster_img.copy()
    
    # Where there is a green mask (undamaged buildings), replace with the post-disaster colors
    result_img[pre_disaster_mask > 0] = post_disaster_buildings[pre_disaster_mask > 0]

    # Modify the pre-disaster filename for saving the result image
    post_disaster_file = os.path.basename(post_image_mask)
    post_filename = post_disaster_file.replace("_pre_", "_post_") 

    # Save the result image
    output_path = os.path.join(save_folder, post_filename)
    cv2.imwrite(output_path, result_img)
    print(f"Result image saved at {output_path}")
    
    return output_path


def Overlay_mask(pre_disaster_image_path, New_post_image_mask):
    pre_disaster_rgb = cv2.imread(pre_disaster_image_path)
    Post_image_mask = cv2.imread(New_post_image_mask)
    
    alpha = 0.5  # Set transparency level (0.0 to 1.0, with 1.0 being fully opaque)
    
    result_img_resized = cv2.resize(Post_image_mask, (pre_disaster_rgb.shape[1], pre_disaster_rgb.shape[0]))
    
    # Perform alpha blending of the result_img and pre-disaster RGB image
    overlay = cv2.addWeighted(pre_disaster_rgb, 1 - alpha, result_img_resized, alpha, 0)
    
    # Modify the post-disaster filename for saving the result image
    post_disaster_file = os.path.basename(pre_disaster_image_path)
    post_filename = post_disaster_file.replace("_pre_", "_post_overlay_") 

    # Save the result image
    output_path = os.path.join(save_folder_overlay, post_filename)
    cv2.imwrite(output_path, overlay)
    print(f"Overlay image saved at {output_path}")
    
    return output_path