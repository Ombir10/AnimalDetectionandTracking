import os
import random

# Path to your dataset
image_dir = r"C:\Users\raiom.LAPTOP-59QT21KS\Computer Vision\22067341_OmbirRai\dataset\voc_night\voc_night/JPEGImages"
images = [f.split('.')[0] for f in os.listdir(image_dir) if f.endswith('.jpg')]

# Shuffle and split
random.shuffle(images)
train_split = int(0.8 * len(images))
train = images[:train_split]
val = images[train_split:]

# Make output directory
os.makedirs("C:/Users/raiom.LAPTOP-59QT21KS/Computer Vision/22067341_OmbirRai/dataset/voc_night/voc_night/ImageSets/Main", exist_ok=True)

# Write to files
with open("C:/Users/raiom.LAPTOP-59QT21KS/Computer Vision/22067341_OmbirRai/dataset/voc_night/voc_night/ImageSets/Main/train.txt", "w") as f:
    f.write("\n".join(train) + "\n")


with open("C:/Users/raiom.LAPTOP-59QT21KS/Computer Vision/22067341_OmbirRai/dataset/voc_night/voc_night/ImageSets/Main/val.txt", "w") as f:
    f.write("\n".join(val) + "\n")

