import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import torch
from PIL import Image
import pandas as pd
from Code.model_creater import load_modal,run_inference
from Code.dataset_creater import save_uploaded_image
from Code.mask_creater import mask_creater,Overlay_mask
from Code.Severity_Index import Severity_Index_Predict,display_damage_table

IMAGE_PATH = os.path.join('DEV', 'output', 'temp')
PRE_IMAGE_MASK = os.path.join('DEV', 'output', 'predicted_weight_dominant_v2', 'localization')
POST_IMAGE_MASK = os.path.join('DEV', 'output', 'predicted_weight_dominant_v2', 'damage')

st.set_page_config(page_title="DAG-NET", page_icon=":satellite:")

# Function to get pre or post-disaster files from the directory
def get_image_files(directory, keyword):
    return [f for f in os.listdir(directory) if keyword in f and (f.endswith('.png') or f.endswith('.jpg'))]


# Streamlit app interface
def main():
    st.markdown("<h2 style='text-align: center;'> Damage Analyzer Generative Network </h2>", unsafe_allow_html=True)
    
    st.markdown("<h4 style='text-align: center;'>Upload images to run Prediction</h4>", unsafe_allow_html=True)
    upload_container = st.container()
    
    with upload_container:
        # Create an expander for upload instructions
        with st.expander("Please upload pre-disaster and post-disaster images in JPG or PNG format."):
            
            col1, col2 = st.columns(2)
        
            with col1:
                pre_disaster_image = st.file_uploader("Upload pre-disaster image", type=['jpg', 'png'], accept_multiple_files=False)
            
            with col2:
                post_disaster_image = st.file_uploader("Upload post-disaster image", type=['jpg', 'png'], accept_multiple_files=False)          

        
    if pre_disaster_image and post_disaster_image:
        # st.image([pre_disaster_image, post_disaster_image], caption=["Pre-disaster", "Post-disaster"], use_column_width=True)
        st.markdown("<h4 style='text-align: center;'>Images Preview</h4>", unsafe_allow_html=True)
        pre_disaster_image_path = save_uploaded_image(pre_disaster_image,DIR=IMAGE_PATH)
        post_disaster_image_path = save_uploaded_image(post_disaster_image,DIR=IMAGE_PATH)
        
        Image_Viewer(pre_disaster_image_path, post_disaster_image_path)
        
        # Process images and run prediction when 'Predict' button is clicked
        model, df = load_modal()
        if st.button("Predict"):
            with st.spinner("Running prediction..."):

                ## Load the model
                metrics, df = run_inference(model=model,
                                            df=df,pre_disaster_image=pre_disaster_image_path,
                                            post_disaster_image=post_disaster_image_path)
                ## predicting the result
                st.success("Prediction completed!")
                
                # Prediction Viewer
                st.markdown("<h4 style='text-align: center;'>Pre and Post Image Masks</h4>", unsafe_allow_html=True)
                pre_image_mask, post_image_mask = Prediction_Viewer(pre_disaster_image_path,post_disaster_image_path)
                
                #making the mask 
                st.markdown("<h4 style='text-align: center;'>Overlaped Post Over Pre Image Mask</h4>", unsafe_allow_html=True)
                New_post_image_mask = mask_creater(pre_image_mask, post_image_mask)
                Image_Viewer(pre_image_mask,New_post_image_mask)
                
                # overlay images
                st.markdown("<h4 style='text-align: center;'>Overlay Images</h4>", unsafe_allow_html=True)
                Final_predicted_img = Overlay_mask(pre_disaster_image_path, New_post_image_mask)
                Image_Viewer(pre_disaster_image_path,Final_predicted_img)
                
                
                # calculation of model predictions
                severity_index, damage_areas, Total_building_area = Severity_Index_Predict(pre_image_mask,New_post_image_mask)
                
                st.markdown("<h4 style='text-align: center;'> Model Evaluation </h4>", unsafe_allow_html=True)
                with st.expander("Model Assessment : DAMS"):
                    
                    st.markdown("<h4 style='text-align: center;'> Disaster Assessment and Metric System </h4>", unsafe_allow_html=True)
                    # Display results
                    Matrix_viewer(metrics,df)
                    # severity_prediction 
                    display_damage_table(severity_index=severity_index, damage_areas=damage_areas, total_building_area=Total_building_area)
                
def Matrix_viewer(metrics, df):
    col1, col2, col3 = st.columns([2, 6, 2])
    with col2:
        # Center the header using HTML
        st.markdown("<h4 style='text-align: center;'>Prediction Metrics:</h4>", unsafe_allow_html=True)
        
        # Define the table with enhanced styling
        st.write("""
        <table style="font-size: 20px; width: 80%; margin-left:auto; margin-right:auto; border-collapse: collapse;">
        <tr>
        <th style="border: 1px solid black; padding: 8px; ">Metric</th>
        <th style="border: 1px solid black; padding: 8px; ">Mean</th>
        <th style="border: 1px solid black; padding: 8px; ">Std Dev</th>
        </tr>
        <tbody>
        <tr>
        <td style="border: 1px solid black; padding: 8px; ">Score</td>
        <td style="border: 1px solid black; padding: 8px; ">{:.4f}</td>
        <td style="border: 1px solid black; padding: 8px; ">{:.4f}</td>
        </tr>
        <tr>
        <td style="border: 1px solid black; padding: 8px; ">Localization</td>
        <td style="border: 1px solid black; padding: 8px; ">{:.4f}</td>
        <td style="border: 1px solid black; padding: 8px; ">{:.4f}</td>
        </tr>
        <tr>
        <td style="border: 1px solid black; padding: 8px; ">Damage</td>
        <td style="border: 1px solid black; padding: 8px; ">{:.4f}</td>
        <td style="border: 1px solid black; padding: 8px; ">{:.4f}</td>
        </tr>
        </tbody>
        </table>
        """.format(
            metrics['score'], 
            df["score"].std(), 
            metrics['localization'], 
            df["localization"].std(), 
            metrics['damage'], 
            df["damage"].std()
        ), unsafe_allow_html=True)

def Prediction_Viewer(pre_disaster_image_path,post_disaster_image_path): 
    
    pre_image_file =  os.path.basename(pre_disaster_image_path)
    unique_id = pre_image_file.split('_')[1] 
    pre_image_file = f'test_localization_{unique_id}_pre_disaster_prediction.png'
    pre_image_mask = os.path.join(PRE_IMAGE_MASK, pre_image_file )
    
    post_image_file =  os.path.basename(post_disaster_image_path)
    unique_id = post_image_file.split('_')[1] 
    post_image_file = f'test_damage_{unique_id}_pre_disaster_prediction.png'
    post_image_mask = os.path.join(POST_IMAGE_MASK, post_image_file)
    
    # if os.path.exists(pre_image_mask) and os.path.exists(post_image_mask):
    #     st.image([pre_image_mask, post_image_mask], caption=["Pre-disaster Mask", "Post-disaster Mask"], use_column_width=True)
    Image_Viewer(pre_image_mask, post_image_mask)
    
    return pre_image_mask, post_image_mask
      
def Image_Viewer(Pre_image_path, Post_image_path):
    if os.path.exists(Pre_image_path) and os.path.exists(Post_image_path):
        pre_image_file_name = os.path.basename(Pre_image_path)
        post_image_file_name = os.path.basename(Post_image_path)
        
        # Use an expander to group both images
        with st.expander("Expand Images", expanded=True):
            # Create two columns inside the expander
            col1, col2 = st.columns(2)
            
            # Display images in each column
            with col1:
                st.image(Pre_image_path, caption=pre_image_file_name, use_column_width=True)
            
            with col2:
                st.image(Post_image_path, caption=post_image_file_name, use_column_width=True)

if __name__ == "__main__":
    main()
