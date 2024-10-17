import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.nn import functional as F
from pytorch_toolbelt.utils.torch_utils import to_numpy
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from xview.dataset import (
    INPUT_IMAGE_ID_KEY,
    OUTPUT_MASK_KEY,
    INPUT_IMAGE_KEY,
)
from xview.postprocessing import (
    make_predictions_dominant_v2,
    make_pseudolabeling_target,
)
from xview.utils.inference_image_output import colorize_mask

import numpy as np

@torch.no_grad()
def run_inference_on_dataset(
    model, dataset, output_dir, batch_size=1, workers=0, weights=None, fp16=False, cpu=False, postprocessing="naive", save_pseudolabels=True
):
    if not cpu:
        if fp16:
            model = model.half()
        model = model.cuda()
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            print("Using multi-GPU inference")

    model = model.eval()

    if weights is not None:
        print("Using weights", weights)
        weights = torch.tensor(weights).float().view(1, -1, 1, 1)

        if not cpu:
            if fp16:
                weights = weights.half()
            weights = weights.cuda()

    data_loader = DataLoader(dataset, batch_size=batch_size, pin_memory=not cpu, num_workers=workers)

    pseudolabeling_dir = os.path.join(output_dir + "_pseudolabeling")

    # os.makedirs(output_dir, exist_ok=True)
    # os.makedirs(pseudolabeling_dir, exist_ok=True)

    postprocessings = {}
    if postprocessing in {"dominant2", "dominantv2", "dominant_v2"}:
        postprocessings[postprocessing] = make_predictions_dominant_v2

    for batch in tqdm(data_loader):
        batch_dict = dict(batch)
        image_ids = batch_dict['image_id']
        # print(image_ids)
        
        image = batch[INPUT_IMAGE_KEY]
        if not cpu:
            if fp16:
                image = image.half()
            image = image.cuda(non_blocking=True)

        image_ids = batch[INPUT_IMAGE_ID_KEY]
        # print(image_ids)
        
        output = model(image)
        
        masks = output[OUTPUT_MASK_KEY]

        if weights is not None:
            masks *= weights

        if masks.size(2) != 1024 or masks.size(3) != 1024:
            masks = F.interpolate(masks, size=(1024, 1024), mode="bilinear", align_corners=False)
        masks = to_numpy(masks).astype(np.float32)

        for i, image_id in enumerate(image_ids):
            image_uuid = "_".join(image_id.split("_")[-3:])

            # Save pseudolabeling target
            if save_pseudolabels:
                pseudo_mask = make_pseudolabeling_target(masks[i])
                pseudo_mask = pseudo_mask.astype(np.uint8)
                pseudo_mask = colorize_mask(pseudo_mask)
                pseudo_mask.save(os.path.join(pseudolabeling_dir, f"test_post_{image_uuid}.png"))

            for postprocessing_name, postprocessing_fn in postprocessings.items():

                output_dir_for_postprocessing = os.path.join(output_dir + "_" + postprocessing_name)
                os.makedirs(output_dir_for_postprocessing, exist_ok=True)
                
                localization_dir = os.path.join(output_dir_for_postprocessing, "localization")
                os.makedirs(localization_dir, exist_ok=True)
                
                damage_dir = os.path.join(output_dir_for_postprocessing, "damage")
                os.makedirs(damage_dir, exist_ok=True)
                 
                localization_image, damage_image = postprocessing_fn(masks[i])

                localization_fname = os.path.join(
                    localization_dir, f"test_localization_{image_uuid}_prediction.png"
                )
                localization_image = colorize_mask(localization_image)
                localization_image.save(localization_fname)

                damage_fname = os.path.join(damage_dir, f"test_damage_{image_uuid}_prediction.png")
                damage_image = colorize_mask(damage_image)
                damage_image.save(damage_fname)

    del data_loader