import cv2
import numpy as np
from utils import show, load, readtxt, get_cells_as_singel_objects, get_seeds, get_mean_of_cell, region_grow

def segment(img, mask, seed, cell): #Need to initialize in middle of cell and mean of cell intensitiy does not work
    cell_idx = 5
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blured = cv2.GaussianBlur(gray, (5,5), 0)
    #mean_of_cell = get_mean_of_cell(blured, cell[cell_idx])
    out = region_grow(blured, seed[cell_idx][0], seed[cell_idx][1], 23)
    out[out < 1] = 0
    out = out.astype(np.uint8)
    cell[cell_idx] = cell[cell_idx].astype(np.uint8)
    show([blured, out, cell[cell_idx]])

    pass

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
    for cell in cells:
        cell_object = get_cells_as_singel_objects(cell)
        cell_objects.append(cell_object)
        seed_objects = get_seeds(cell_object)
        seeds.append(seed_objects)
        for i in range(len(cell_object)):
            if cell_object[i][seed_objects[i][0], seed_objects[i][1]] != 1:
                print("Seed and cell does not match")

    for img, mask, seed, cell in zip(imgs, masks, seeds, cell_objects): # for debugging)
        output = segment(img, mask, seed, cell)
        outputs.append(output)
        show(img, output)

    # for mask, gold in zip(masks, golds):
    #     score = evaluate(mask, gold)
    #     print(score)


if __name__ == "__main__":
    main()
