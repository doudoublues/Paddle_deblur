import os
import random
import glob

image_dir = 'dataset/train/blur_image/*.png'
image_paths = glob.glob(image_dir)

f = open('train_new.txt', 'w')

for i in range(len(image_paths)):
    f.write(image_paths[i])
    f.write('\n')

# random.seed(2021)

# image_dir = "dataset/*/blur_image/*.png"
# image_paths = glob.glob(image_dir)

# random.shuffle(image_paths)

# length = len(image_paths)

# train_path = open('train_95.txt', 'w')
# val_path = open('val_95.txt', 'w')


# for i in range(length):
#     if i < int(0.95*length):
#         train_path.write(image_paths[i])
#         train_path.write('\n')
#     else:
#         val_path.write(image_paths[i])
#         val_path.write('\n')
