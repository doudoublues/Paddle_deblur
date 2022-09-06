import glob
import os
import cv2

train_files = open('val_95.txt', 'r').readlines()
blur_dir = 'dataset/val/blur_image'
gt_dir = 'dataset/val/gt_image'

if not os.path.exists(blur_dir):
    os.makedirs(blur_dir)

if not os.path.exists(gt_dir):
    os.makedirs(gt_dir)

count = 1
patch_size = 512

for file in train_files:
    file = file.strip()
    img = cv2.imread(file)
    print(file.replace('blur_image', 'gt_image'))
    gt = cv2.imread(file.replace('blur_image', 'gt_image'))
    h, w, c = img.shape
    for i in range(0, h - int(patch_size * 0.4) - 1, int(patch_size * 0.4)):
        for j in range(0, w - int(patch_size * 0.4) - 1, int(patch_size * 0.4)):
            blur_patch = img[i : i+512, j : j+512]
            gt_patch = gt[i : i+512, j : j+512]
            cv2.imwrite(os.path.join(blur_dir, 'val_img_{}.png'.format(count)), blur_patch)
            cv2.imwrite(os.path.join(gt_dir, 'val_img_{}.png'.format(count)), gt_patch)
            count += 1