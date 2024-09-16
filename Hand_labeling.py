from roipoly import RoiPoly
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import os
import csv
import cv2
import numpy as np

images = []
filenames_list = []
images_rgbrgb = [] # for float stuff

HSV = True

for filename in os.listdir('ECE5242Proj1-train'):
    filenames_list.append(filename)
    if filename.endswith(".png"):
        img_path = os.path.join('ECE5242Proj1-train', filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read {img_path}")
        if HSV:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images_rgbrgb.append(img_rgb)
            img = img.astype(np.float32) / 255 # Convert to float32
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 

            ## old
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
        print(f"Loaded {filename}")
    # break # this line is just for testing

print("filenames: ", filenames_list)

segmented_colors_dict = {'Orange cone': []}
segmented_colors_dict_train = {'Orange cone': []}
segmented_colors_dict_val = {'Orange cone': []}

# train_images = images[:-5]
# val_images = images[-5:]

# get orange cone region for each image
for i, img in enumerate(images):

    # Show the image
    fig = plt.figure(figsize=(10, 8))
    # plt.imshow(img, interpolation='nearest', cmap="Greys") ## old
    plt.imshow(images_rgbrgb[i], interpolation='nearest', cmap="Greys") ## float code
    plt.colorbar()
    plt.title("left click: line segment         right click or double click: close region")
    plt.show(block=False)

    # Let user draw first ROI
    roi = RoiPoly(color='r', fig=fig)

    # mask = roi.get_mask(img[:, :, 0]) ## old
    mask = roi.get_mask(images_rgbrgb[i][:, :, 0]) ## float code


    cone_colors = img[mask]

    # print(f'image {i} cone values: {cone_colors}')

    if i < len(images) - 5:
    # if True: ## This line is just for testing
        segmented_colors_dict_train['Orange cone'].extend(cone_colors)
    else:
        ## separating last 5 images for testing myself
        segmented_colors_dict_val['Orange cone'].extend(cone_colors)

# Write out segmented colors dict to csv
if HSV:
    csv_paths = [('hand_labels_hsv_train_float.csv', segmented_colors_dict_train),
                  ('hand_labels_hsv_val_float.csv', segmented_colors_dict_val)] # training data with shifted hsv values
else:
    csv_paths = [('hand_labels_rbg_train.csv', segmented_colors_dict_train), ('hand_labels_rbg_val.csv', segmented_colors_dict_val)]
for csv_path, color_dict in csv_paths:
    with open(csv_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Label', 'H', 'S', 'V'] if HSV else ['Label', 'R', 'G', 'B'])
        for label, colors in color_dict.items():
            for color in colors:
                csvwriter.writerow([label, color[0], color[1], color[2]])

if HSV:
    # plot 3D plot of all the color points in segmented_colors_dict in 3D rgb space and hsv space
    from mpl_toolkits.mplot3d import Axes3D

    # First HSV plot 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for color_name, colors in segmented_colors_dict_train.items():
        # Convert each HSV color in the list to RGB
        colors_rgb = [cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_HSV2RGB)[0][0] for color in colors]

        hs = [color[0] for color in colors]  
        ss = [color[1] for color in colors]  
        vs = [color[2] for color in colors] 
        colors_normalized = [[r/255, g/255, b/255] for r, g, b in colors_rgb] # need to normalize for matplotlib apparently
        ax.scatter(hs, ss, vs, color='blue')


    ax.set_xlabel('Hue')
    ax.set_ylabel('Saturation')
    ax.set_zlabel('Value')
    plt.savefig('hsv_plot_float.png')
    # plt.show()

    # Then rgb plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for color_name, colors in segmented_colors_dict_train.items():
        # Convert each HSV color in the list to RGB
        colors_rgb = [cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_HSV2RGB)[0][0] for color in colors]

        xs = [color[0] for color in colors_rgb]
        ys = [color[1] for color in colors_rgb]
        zs = [color[2] for color in colors_rgb]
        colors_normalized = [[r/255, g/255, b/255] for r, g, b in colors_rgb] # need to normalize for matplotlib apparently
        ax.scatter(xs, ys, zs, color=colors_normalized)

    ax.set_xlabel('R')
    ax.set_ylabel('G')
    ax.set_zlabel('B')
    plt.savefig('rgb_plot.png')
    # plt.show()


else:
    # plot 3D plot of all the color points in segmented_colors_dict in 3D rgb space
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for color_name, colors in segmented_colors_dict_train.items():
        xs = [color[0] for color in colors]
        ys = [color[1] for color in colors]
        zs = [color[2] for color in colors]
        colors_normalized = [[r/255, g/255, b/255] for r, g, b in colors] # need to normalize for matplotlib apparently
        ax.scatter(xs, ys, zs, color=colors_normalized)

    ax.set_xlabel('R')
    ax.set_ylabel('G')
    ax.set_zlabel('B')
    plt.show()
    plt.savefig('rgb_plot.png')