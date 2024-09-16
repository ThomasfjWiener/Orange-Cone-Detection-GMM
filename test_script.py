import cv2
from matplotlib import pyplot as plt
import os
import csv
import pandas as pd
import numpy as np
import cv2
from skimage import data, util
from skimage.measure import label, regionprops
from skimage import morphology

import inference

folder = "Test\_Set" # REPLACE THIS WITH PATH TO TEST FOLDER
folder = "ECE5242Proj1-test" # REPLACE THIS WITH PATH TO TEST FOLDER

##### upload lookup table for score ######
# lookup_table_builtin = np.load("lookup_table_builtin.npy") # keep commented out
# lookup_table_naive = np.load("lookup_table_naive_1img.npy") # keep commented out
lookup_table_12_comps = np.load("lookup_table_12_components.npy") # Final prediction model that should be run
# lookup_table_30_comps = np.load("lookup_table_30_components.npy") # 30-component prediction model

# lines_bn = []
# lines_naive = []
lines_final = []

for filename in os.listdir(folder):
    print(filename) # record the filename, since listdir is unordered
    # read one test image
    if filename.endswith(".png"):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read {img_path}")
            continue
        print(f"Loaded {filename}")

        # My computations here!
        # res_lst_bn = inference.run(img=img, lookup_table=lookup_table_builtin, th=10 ** -6.5, name= "Built-in GMM")
        # res_lst_naive = inference.run(img=img, lookup_table=lookup_table_naive, th=10 ** -9, name= "Naive GMM")
        res_lst_12comps = inference.run(img=img, lookup_table=lookup_table_12_comps, th=10 ** -6.5, name= "My 30 component GMM (150 epochs)", imgname=filename)
        # res_lst_30comps = inference.run(img=img, lookup_table=lookup_table_30_comps, th=10 ** -6.5, name= "My 30 component GMM (150 epochs)", imgname=filename)

    
    # for (x, y), d in res_lst_bn:
    #     # If there are multiple "cones" detected, then each is written on their own line
    #     lines_bn.append(f"Image Name = {filename}, Bottom-Center X = {x}, Bottom-Center Y = {y}, Distance = {d} ft")
    
    # for (x, y), d in res_lst_naive:
    #     # If there are multiple "cones" detected, then each is written on their own line
    #     lines_naive.append(f"Image Name = {filename}, Bottom-Center X = {x}, Bottom-Center Y = {y}, Distance = {d} ft")
        
    for (x, y), d in res_lst_12comps:
        # If there are multiple "cones" detected, then each is written on their own line
        lines_final.append(f"Image Name = {filename}, Bottom-Center X = {x}, Bottom-Center Y = {y}, Distance = {d} ft")

# with open('test_builtinGMM_output.txt', 'w') as file:
#     for line in lines_bn:
#         file.write(line + '\n')

# with open('test_naive_implementation_output.txt', 'w') as file:
#     for line in lines_naive:
#         file.write(line + '\n')

with open('test_12_components_GMM_output.txt', 'w') as file:
    for line in lines_final:
        file.write(line + '\n')