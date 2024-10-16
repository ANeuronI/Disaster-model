import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np

from xview.dataset import read_mask

import matplotlib.pyplot as plt

from xview.postprocessing import make_predictions_floodfill, make_predictions_dominant_v2
from xview.utils.inference_image_output import make_rgb_image
import pytest


@pytest.mark.parametrize(["actual", "expected"], [
    ("hurricane-florence_00000115_post_disaster.npy", "hurricane-florence_00000115_post_disaster.png"),
    ("hurricane-florence_00000475_post_disaster.npy", "hurricane-florence_00000475_post_disaster.png"),
])
def test_watershed(actual, expected):
    dmg = np.load(actual)
    dmg_true = read_mask(expected)

    loc_cls, dmg_cls = make_predictions_dominant_v2(dmg)

    plt.figure()
    plt.imshow(make_rgb_image(dmg_true))
    plt.show()

    plt.figure()
    plt.imshow(make_rgb_image(np.argmax(dmg, axis=0)))
    plt.show()

    plt.figure()
    plt.imshow(make_rgb_image(loc_cls))
    plt.show()

    plt.figure()
    plt.imshow(make_rgb_image(dmg_cls))
    plt.show()


def test_watershed_with_image() -> None:
    dmg = read_mask("tests/test_damage_00121_prediction.png")
    loc = read_mask("tests/test_localization_00121_prediction.png")
    img = cv2.imread("tests/test_post_00121.png")

    assert img.shape[:2] == dmg.shape, "Image and mask shapes mismatch"
    assert img.shape[:2] == loc.shape, "Image and localization shapes mismatch"

    # Fix mask
    dmg[loc == 0] = 0

    seed = dmg.copy()
    seed[loc == 0] = 0

    try:
        markers = cv2.watershed(img, seed.astype(np.int32))
    except cv2.error as e:
        print(f"OpenCV Error: {e}")
        return

    markers[markers == 0] = 1

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(dmg)
    axs[1].imshow(loc)
    axs[2].imshow(markers)
    plt.show()


if __name__ == "__main__":
    try:
        test_watershed_with_image()
    except Exception as e:
        print(f"Error: {e}")