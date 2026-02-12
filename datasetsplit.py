import os
import random
import shutil

# paths
source_dir = "dataset/Potato___Early_blight"
train_dir = "dataset/train"
test_dir = "dataset/test"

classes = ["Healthy", "Blight"]
split_ratio = 0.8   # 80% train, 20% test

# create folders
for cls in classes:
    os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(test_dir, cls), exist_ok=True)

# split images
for cls in classes:
    cls_path = os.path.join(source_dir, cls)
    images = os.listdir(cls_path)
    random.shuffle(images)

    split_index = int(len(images) * split_ratio)

    train_images = images[:split_index]
    test_images = images[split_index:]

    for img in train_images:
        shutil.copy(os.path.join(cls_path, img),
                    os.path.join(train_dir, cls, img))

    for img in test_images:
        shutil.copy(os.path.join(cls_path, img),
                    os.path.join(test_dir, cls, img))

print("Train-Test split completed ✅")
