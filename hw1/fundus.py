import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import sharpen_img, clip_outside, show, load, filtered_connected_components, adapt_threshold, evaluate


def segment(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    sharpen = sharpen_img(blur, kernel_size=15)
    thr = adapt_threshold(sharpen)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = filtered_connected_components(
        mask, connectivity=8, area_threshold=40)
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
        show(img, mask)

    for mask, gold in zip(masks, golds):
        score = evaluate(mask, gold)
        print(score)


if __name__ == "__main__":
    main()
