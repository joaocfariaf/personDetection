import os
import shutil
import random

patches = os.path.join("..", "patches")
positive_patches = os.path.join(patches, "positive")
negative_patches = os.path.join(patches, "negative")
n_positive = 29050
n_negative = 39700

data = os.path.join("..", "data")
train_dir = os.path.join(data, "train")
val_dir = os.path.join(data, "validation")
test_dir = os.path.join(data, "test")
data_folders = [train_dir, val_dir, test_dir]

positive = [os.path.join(train_dir, "positive"), os.path.join(val_dir, "positive"), os.path.join(test_dir, "positive")]
negative = [os.path.join(train_dir, "negative"), os.path.join(val_dir, "negative"), os.path.join(test_dir, "negative")]

dir_list = [data, *data_folders, *positive, *negative]
if os.path.exists(data):
    shutil.rmtree(data)
for directory in dir_list:
    os.mkdir(directory)

split = [.7, .2, .1]

pos_list = random.sample(os.listdir(positive_patches), k=n_positive)
pos = [pos_list[0:int(split[0] * n_positive)],
       pos_list[int(split[0] * n_positive):int(split[0] * n_positive)+int(split[1] * n_positive)],
       pos_list[int(split[0] * n_positive)+int(split[1] * n_positive): n_positive]]
neg_list = random.sample(os.listdir(negative_patches), k=n_negative)
neg = [neg_list[0:int(split[0] * n_negative)],
       neg_list[int(split[0] * n_negative):int(split[0] * n_negative)+int(split[1] * n_negative)],
       neg_list[int(split[0] * n_negative)+int(split[1] * n_negative):n_negative]]

for i in range(3):
    for file in pos[i]:
        shutil.copy(os.path.join(positive_patches, file), positive[i])
    for file in neg[i]:
        shutil.copy(os.path.join(negative_patches, file), negative[i])
