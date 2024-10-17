import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import torch
from pytorch_toolbelt.utils import fs
from xview.dataset import OUTPUT_MASK_KEY
from xview.inference import (
    model_from_checkpoint,
    ApplyWeights,
    Ensembler,
    ApplySoftmaxTo
)
import os
from Code.dataset_creater import get_test_dataset
from Code.inference_Creator import run_inference_on_dataset

import streamlit as st

def weighted_model(checkpoint_fname: str, weights, activation: str):
    model, info = model_from_checkpoint(fs.auto_file(checkpoint_fname, where="./DEV/Code/models"), activation_after=activation, report=False)
    model = ApplyWeights(model, weights)
    return model, info

WORKERS = 4
IMAGE_SIZE = 1024, 1024
BATCH_SIZE = 1
ACTIVATION_AFTER = 'ensemble'
FP16 = 'store_true'
POSTPROCESSING = 'dominant_v2'
OUTPUT_DIR = "./DEV/output/predicted_weight"

@st.cache_resource
def load_modal():

    print("Size      ", IMAGE_SIZE)
    print("Output dir", OUTPUT_DIR)
    print("Postproc  ", POSTPROCESSING)

    fold_0_models_dict = [

        (
            "Dec30_15_34_resnet34_unet_v2_512_fold0_fp16_pseudo_crops.pth",
            [0.51244243, 1.42747062, 1.23648384, 0.90290896, 0.88912514],
        ),

        (
            "Dec30_15_34_resnet101_fpncatv2_256_512_fold0_fp16_pseudo_crops.pth",
            [0.50847073, 1.15392272, 1.2059733, 1.1340391, 1.03196719],
        ),
    ]

    fold_1_models_dict = [
        (
            "Dec22_22_24_seresnext50_unet_v2_512_fold1_fp16_crops.pth",
            [0.54324459, 1.76890163, 1.20782899, 0.85128004, 0.83100698],
        ),
        (
            "Dec31_02_09_resnet34_unet_v2_512_fold1_fp16_pseudo_crops.pth",
            [0.48269921, 1.22874469, 1.38328066, 0.96695393, 0.91348539],
        ),
        (
            "Dec31_03_55_densenet201_fpncatv2_256_512_fold1_fp16_pseudo_crops.pth",
            [0.48804137, 1.14809462, 1.24851827, 1.11798428, 1.00790482]
        )
    ]

    fold_2_models_dict = [
        (
            "Dec17_19_12_inceptionv4_fpncatv2_256_512_fold2_fp16_crops.pth",
            [0.34641084, 1.63486251, 1.14186036, 0.86668715, 1.12193125],
        ),

        (
            "Dec27_14_08_densenet169_unet_v2_512_fold2_fp16_crops.pth",
            [0.55429115, 1.34944309, 1.1087044, 0.89542089, 1.17257541],
        ),
        (
            "Dec31_12_45_resnet34_unet_v2_512_fold2_fp16_pseudo_crops.pth",
            [0.65977938, 1.50252452, 0.97098732, 0.74048182, 1.08712367],
        )
    ]

    fold_3_models_dict = [
        (
            "Dec15_23_24_resnet34_unet_v2_512_fold3_crops.pth",
            [0.84090623, 1.02953555, 1.2526516, 0.9298182, 0.94053529],
        ),

        (
            "Dec21_11_50_seresnext50_unet_v2_512_fold3_fp16_crops.pth",
            [0.43108046, 1.30222898, 1.09660616, 0.94958969, 1.07063753],
        ),
        (
            "Dec31_18_17_efficientb4_fpncatv2_256_512_fold3_fp16_pseudo_crops.pth",
            [0.59338243, 1.17347438, 1.186104, 1.06860638, 1.03041829]
        )
    ]

    fold_4_models_dict = [
        (
            "Dec19_06_18_resnet34_unet_v2_512_fold4_fp16_crops.pth",
            [0.83915734, 1.02560309, 0.77639015, 1.17487775, 1.05632771],
        ),
        (
            "Dec27_14_37_resnet101_unet_v2_512_fold4_fp16_crops.pth",
            [0.57414314, 1.19599486, 1.05561912, 0.98815567, 1.2274592],
        ),
    ]

    infos = []
    models = []

    for models_dict in [
        fold_0_models_dict,
        fold_1_models_dict,
        fold_2_models_dict,
        fold_3_models_dict,
        fold_4_models_dict,
    ]:
        for checkpoint, weights in models_dict:
            model, info = weighted_model(checkpoint, weights, activation=ACTIVATION_AFTER)
            models.append(model)
            infos.append(info)

    model = Ensembler(models, outputs=[OUTPUT_MASK_KEY])

    df = pd.DataFrame.from_records(infos)

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)

    print(df)
    print("score        ", df["score"].mean(), df["score"].std())
    print("localization ", df["localization"].mean(), df["localization"].std())
    print("damage       ", df["damage"].mean(), df["damage"].std())

    if ACTIVATION_AFTER == "ensemble":
        model = ApplySoftmaxTo(model, OUTPUT_MASK_KEY)
        print("Applying activation after ensemble")
        
    return model, df

def run_inference(model, df, pre_disaster_image, post_disaster_image): 
    
    test_ds = get_test_dataset(pre_image=pre_disaster_image, post_image=post_disaster_image, image_size=IMAGE_SIZE)

    run_inference_on_dataset(
        model=model,
        dataset=test_ds,
        output_dir=OUTPUT_DIR,
        batch_size=BATCH_SIZE,
        workers=WORKERS,
        fp16=FP16,
        postprocessing=POSTPROCESSING,
        save_pseudolabels=False,
        cpu=False
    )
    
    # Return metrics as a dictionary
    metrics = {
        "score": df["score"].mean(),
        "localization": df["localization"].mean(),
        "damage": df["damage"].mean()
    }
    
    return metrics ,df
