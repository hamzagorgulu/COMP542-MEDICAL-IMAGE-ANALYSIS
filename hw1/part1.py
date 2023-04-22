import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils_v2 import load, adapt_threshold, sharpen_img, evaluate, show, filtered_connected_components, clip_outside, load_txt, show_multiple, check_set


def obtain_foreground(img):
    #print(img.shape)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (11, 11), 0)
    sharpened = sharpen_img(blur, kernel_size=6)
    adapt_th = adapt_threshold(sharpened)

    kernel = np.ones((8, 8), np.uint8)
    mask = cv2.morphologyEx(adapt_th, cv2.MORPH_CLOSE, kernel, iterations=2)

    #mask = filtered_connected_components(mask, connectivity=500, area_threshold=400)
    #breakpoint()
    
    return mask

def main():
    im1_path = "data/im1.jpg"
    gold1_path = "data/im1_gold_mask.txt"
    im2_path = "data/im2.jpg"
    gold2_path = "data/im2_gold_mask.txt"
    im3_path = "data/im3.jpg"
    gold3_path = "data/im3_gold_mask.txt"

    im_paths = [im1_path, im2_path, im3_path]
    gold_paths = [gold1_path, gold2_path, gold3_path]
    ims = []
    golds = []
    masks = []

    for im_path, gold_path in zip(im_paths, gold_paths):
        ims.append(load(im_path, 1))
        golds.append(load_txt(gold_path))
    #breakpoint()

    for im, gold in zip(ims, golds):
        mask = obtain_foreground(im)
        masks.append(mask)
        show(im, mask)

    #im_vertical_concat = np.concatenate((ims[0], ims[1], ims[2]), axis=0)
    #mask_vertical_concat = np.concatenate((masks[0], masks[1], masks[2]), axis=0)
    #show(im_vertical_concat, mask_vertical_concat)
    
    for idx, mask in enumerate(masks):
        cv2.imwrite(f"data/mask_{idx}.jpg", mask)

    for mask, gold in zip(masks, golds):
        #breakpoint()
        score = evaluate(mask, gold)
        print(score)

if __name__ == "__main__":
    main()
