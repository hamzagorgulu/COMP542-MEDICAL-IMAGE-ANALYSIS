import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import sharpen_img, clip_outside, show, load, filtered_connected_components, adapt_threshold, evaluate


def segment(img):
    # preprocess
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sharpen = sharpen_img(gray, kernel_size=15)
    # thresholding for feature extraction
    thr = adapt_threshold(sharpen, kernel_size=17, constant=3)
    # postprocessing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    mask = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)
    # noise filtering
    mask = filtered_connected_components(
        mask, connectivity=8, area_threshold=40)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)
    return clip_outside(img, mask)

def main():
    img1_path = "fundus\\d4_h.jpg"
    gold1_path = "fundus\\d4_h_gold.png"
    img2_path = "fundus\\d7_dr.jpg"
    gold2_path = "fundus\\d7_dr_gold.png"
    img3_path = "fundus\\d11_g.jpg"
    gold3_path = "fundus\\d11_g_gold.png"

    img_paths = [img1_path, img2_path, img3_path]
    gold_paths = [gold1_path, gold2_path, gold3_path]
    imgs = []
    golds = []
    masks = []

    for img, gold in zip(img_paths, gold_paths):
        imgs.append(load(img, 1))
        golds.append(load(gold, 0))

    for img, gold in zip(imgs, golds):
        mask = segment(img)
        masks.append(mask)
        show([img, gold, mask])

    for mask, gold in zip(masks, golds):
        score = evaluate(mask, gold)
        print(score)


if __name__ == "__main__":
    main()
