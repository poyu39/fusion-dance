"""
Train & Test Split:
We use each Pokemon's pokedex ID as it's unique ID.
Additionally, we split based on these IDs so as to not contaminate our testing 
or validation sets. 
Thus there may be an uneven split since some Pokemon have female sprites.
It automatically creates the train/val/test folders and moves the images there.

Preprocessing:
This script is reponsible for preprocessing the original data folder.
Since the images are in RGBA, we first need to convert them into RGB images.
Now we need to decide what color the background should be. 
White or black are standard, but they may impact how the model sees certain 
Pokemon that have very dark or very light shades.
We follow Gonzalez et. al's approach and use both. 
In the case of training images, they also used 2 additional noisy backgrounds.
We repeat the same.
In terms of image augmentation, we perform horizontal flips as well as rotations in the 15 and 30 degree angles in two directions. We do not rotate the horizontally flipped image though.
"""

import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image
from tqdm import tqdm


def change_background_color(image, color):
    if color == "none":
        return image.convert("RGB")
    if color == "black":
        background = Image.new("RGBA", image.size, (0, 0, 0))
    elif color == "white":
        background = Image.new("RGBA", image.size, (255, 255, 255))
    else:
        # 使用高斯 noise 作為背景
        width, height = image.size
        noise = np.random.normal(128, 50, (height, width, 4))
        noise = np.clip(noise, 0, 255).astype(np.uint8)
        background = Image.fromarray(noise, mode="RGBA")
    
    return Image.alpha_composite(background, image).convert("RGB")

def process_single_image(args):
    """處理單張圖片的函數"""
    input_file, input_dir, output_dir, test, val, colors, rotations, use_noise, do_a_flip = args
    
    try:
        # Load file
        sprite = Image.open(os.path.join(input_dir, input_file)).convert("RGBA")

        # Find directory to save to
        input_file_name = input_file.split(".")[0]
        image_id = input_file_name.split("_")[0].split("-")[0]
        
        if image_id in test:
            save_dir = os.path.join(output_dir, "test")
            category = "test"
        elif image_id in val:
            save_dir = os.path.join(output_dir, "val")
            category = "val"
        else:
            save_dir = os.path.join(output_dir, "train")
            category = "train"
            if use_noise:
                colors = colors + ["noise1", "noise2"]

        # Augment data
        for color in colors:
            # Change background color & save base image
            new_sprite = change_background_color(sprite, color)
            output_file = f"{input_file_name}_{color}BG_0rotation.png"
            new_sprite.save(os.path.join(save_dir, output_file))

            # Rotation
            prev_angle = 0
            for angle in rotations:
                # Counter Clockwise Rotation
                new_angle = np.random.randint(prev_angle, angle + 1)
                rotated = sprite.rotate(new_angle, expand=False, fillcolor=(0, 0, 0, 0))
                rotated = change_background_color(rotated, color)
                output_file = f"{input_file_name}_{color}BG_{new_angle}rotation.png"
                rotated.save(os.path.join(save_dir, output_file))
                
                # Clockwise Rotation (negative angle)
                clockwise_angle = np.random.randint(prev_angle, angle + 1)
                rotated = sprite.rotate(-clockwise_angle, expand=False, fillcolor=(0, 0, 0, 0))
                rotated = change_background_color(rotated, color)
                output_file = f"{input_file_name}_{color}BG_{360 - clockwise_angle}rotation.png"
                rotated.save(os.path.join(save_dir, output_file))
                
                # Housekeeping
                prev_angle = angle

            # Horizontal Flip
            if do_a_flip:
                # 先翻轉原始 sprite,再應用背景
                flipped_sprite = sprite.transpose(Image.FLIP_LEFT_RIGHT)
                flipped_with_bg = change_background_color(flipped_sprite, color)
                output_file = f"{input_file_name}_{color}BG_flipped.png"
                flipped_with_bg.save(os.path.join(save_dir, output_file))
        
        return (category, image_id)
    
    except Exception as e:
        print(f"\n處理 {input_file} 時發生錯誤: {e}")
        return None

input_dir = sys.argv[1]
output_dir = sys.argv[2]
num_test = int(sys.argv[3])  # 49
num_valid = int(sys.argv[4]) # 100
rotations = [15, 30]
use_noise = True
do_a_flip = True
colors = ["white", "black"]

# Create Folders
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

for folder in ["train", "test", "val"]:
    if not os.path.exists(os.path.join(output_dir, folder)):
        os.mkdir(os.path.join(output_dir, folder))

# Do a Train-Val-Test Split
unique = []
for file in os.listdir(input_dir):
    file = file.split('.')[0].split("_")[0].split("-")[0]
    if file not in unique and file not in ["train", "test", "val"]:
        unique.append(file)

random.shuffle(unique)
test = set(unique[-num_test:])
unique = unique[:-num_test]
val = set(unique[-num_valid:])
train = set(unique[:-num_valid])
print(f"Train: {len(train)}\nVal: {len(val)}\nTest: {len(test)}")

all_train = []
all_val = []
all_test = []

# 準備所有任務參數
input_files = [f for f in os.listdir(input_dir) if not f.startswith('.')]
tasks = [
    (input_file, input_dir, output_dir, test, val, colors, rotations, use_noise, do_a_flip)
    for input_file in input_files
]

# 使用多線程處理
max_workers = os.cpu_count()  # 使用 CPU 核心數作為線程數
print(f"使用 {max_workers} 個線程進行處理...")

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {executor.submit(process_single_image, task): task[0] for task in tasks}
    
    for future in tqdm(as_completed(futures), total=len(futures), desc="處理圖片"):
        try:
            result = future.result()
            if result is not None:
                category, image_id = result
                if category == "test":
                    all_test.append(image_id)
                elif category == "val":
                    all_val.append(image_id)
                else:
                    all_train.append(image_id)
        except Exception as e:
            input_file = futures[future]
            print(f"\n執行任務 {input_file} 時發生錯誤: {e}")

print(f"Train: {len(all_train)}\nVal: {len(all_val)}\nTest: {len(all_test)}")
