import numpy as np
from model import random_rotate

def save_landmarks(row, label, f):
    row = np.around(row)
    for pt in row:
        f.write("{},".format(pt))
    f.write(str(label) + ",\n")

def generate_augmented_data(input_path, output_path, per_item_augment=2):
    data   = np.genfromtxt(input_path, delimiter=',')[:, :-2]
    labels = np.genfromtxt(input_path, delimiter=",")[:, -2:-1]

    output = open(output_path, "a")
    for i in range(data.shape[0]):
        save_landmarks(data[i, :], labels[i, 0], output)
        for _ in range(per_item_augment):
            new_data = random_rotate(data[i, :])
            save_landmarks(new_data, labels[i, 0], output)
    return