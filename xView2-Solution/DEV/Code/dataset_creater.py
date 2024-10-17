import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import albumentations as A
from pytorch_toolbelt.utils import fs
from xview.dataset import ImageMaskDataset

def get_test_dataset(pre_image: str, post_image: str, image_size=(224, 224), use_edges=False, fast=False, align_post=False):
    """
    Create train and validation data loaders
    :param data_dir: Inria dataset directory
    :param fast: Fast training model. Use only one image per location for training and one image per location for validation
    :param image_size: Size of image crops during training & validation
    :param use_edges: If True, adds 'edge' target mask
    :param augmentation: Type of image augmentations to use
    :param train_mode:
    'random' - crops tiles from source images randomly.
    'tiles' - crop image in overlapping tiles (guaranteed to process entire dataset)
    :return: (train_loader, valid_loader)
    """
    resize = A.Resize(image_size[0], image_size[1])
    normalize = A.Normalize(
        mean=(0.485, 0.456, 0.406, 0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225, 0.229, 0.224, 0.225)
    )
    valid_transform = A.Compose([resize, normalize])
    
    # Create lists with the image paths
    test_images_pre = [pre_image]
    test_images_post = [post_image]

    if fast:
        test_images_pre = test_images_pre[:128]
        test_images_post = test_images_post[:128]

    validset = ImageMaskDataset(
        test_images_pre,
        test_images_post,
        None,
        use_edges=use_edges,
        spatial_transform=valid_transform,
        align_post=align_post,
    )

    return validset


def save_uploaded_image(uploaded_image,DIR):
    # Create a temporary directory if it doesn't exist
    if not os.path.exists(DIR):
        os.makedirs(DIR)

    # Save the uploaded image to a temporary file
    image_path = os.path.join(DIR, uploaded_image.name)
    with open(image_path, "wb") as f:
        f.write(uploaded_image.getbuffer())

    return image_path