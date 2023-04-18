import cv2
import numpy as np
from utils import show, load, readtxt, get_cells_as_singel_objects, get_seeds, get_mean_of_cell, region_grow, filtered_connected_components, evaluate_cells, fuse_predictions_to_print


def segment(img, mask, seed, cell):
    segments = []
    for cell_idx in range(len(seed)):
        print(cell_idx)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blured = cv2.medianBlur(gray, 3)
        out = region_grow(blured, seed[cell_idx]
                          [0], seed[cell_idx][1], 0.38, mask)
        out[out < 1] = 0
        out = out.astype(np.uint8)
        cell[cell_idx] = cell[cell_idx].astype(np.uint8)
        if np.sum(out) > 800:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            out = cv2.morphologyEx(out, cv2.MORPH_ERODE, kernel, iterations=2)
            out = filtered_connected_components(out, 8, 1, seed[cell_idx])
            out = cv2.morphologyEx(out, cv2.MORPH_DILATE, kernel, iterations=2)
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (7, 7)), iterations=1)
        out = cv2.morphologyEx(out, cv2.MORPH_DILATE, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
        #show([blured, out, cell[cell_idx]])
        segments.append(out)
    map = fuse_predictions_to_print(segments)
    cell_map = fuse_predictions_to_print(cell)
    show([img, map, cell_map.astype(np.uint8)])
    return segments



def main():
    img1_path = "data\\im1.jpg"
    gold1_cells_path = "data\\im1_gold_cells.txt"
    gold1_mask_path = "data\\im1_gold_mask.txt"
    img2_path = "data\\im2.jpg"
    gold2_cells_path = "data\\im2_gold_cells.txt"
    gold2_mask_path = "data\\im2_gold_mask.txt"
    img3_path = "data\\im3.jpg"
    gold3_cells_path = "data\\im3_gold_cells.txt"
    gold3_mask_path = "data\\im3_gold_mask.txt"

    img_paths = [img1_path, img2_path, img3_path]
    gold_cell_paths = [gold1_cells_path, gold2_cells_path, gold3_cells_path]
    gold_mask_paths = [gold1_mask_path, gold2_mask_path, gold3_mask_path]
    imgs = []
    masks = []
    cells = []
    outputs = []
    seeds = []
    cell_objects = []

    for img, mask, cell in zip(img_paths, gold_mask_paths, gold_cell_paths):
        imgs.append(load(img, 1))
        masks.append(readtxt(mask))
        cells.append(readtxt(cell))

    # artificially create cell locations
    for img, cell in zip(imgs, cells):
        cell_object = get_cells_as_singel_objects(cell)
        cell_objects.append(cell_object)

        seed_objects = get_seeds(cell_object, img)
        seeds.append(seed_objects)
        for i in range(len(cell_object)):
            if cell_object[i][seed_objects[i][0], seed_objects[i][1]] != 1:
                print("Seed and cell does not match")

    # for debugging)
    for img, mask, seed, cell in zip(imgs, masks, seeds, cell_objects):
        output = segment(img, mask, seed, cell)
        outputs.append(output)
        
        

    for img, out, cell in zip(imgs, outputs, cell_objects):
        score = evaluate_cells(out, cell)
        print(score)
        map = fuse_predictions_to_print(out)
        show([map])
        


if __name__ == "__main__":
    main()
