import numpy as np
from math import pi as PI

def save_landmarks(row, label, f):
    row = np.around(row)
    for pt in row:
        f.write("{},".format(pt))
    f.write(str(label) + ",\n")

# Axis param, 0 = Z, 1 = Y, 2 = X
def rotate(lm, theta, axis=0):
    lm = np.array(lm)
    result = np.copy(lm)
    c, s = np.cos(theta), np.sin(theta)
    R = None
    if axis == 0:
        R = np.array([(c, -s, 0), 
                      (s,  c, 0),
                      (0,  0, 1)])    
    if axis == 1:
        R = np.array([( c, 0, s), 
                      ( 0, 1, 0),
                      (-s, 0, c)])   
    if axis == 2:
        R = np.array([(1, 0,  0), 
                      (0, c, -s),
                      (0, s,  c)])

    if len(lm.shape) == 1:
        array = np.array([[lm[i], lm[i+1], 0] for i in range(0, len(lm), 2)])
        return array.dot(R)[:,:-1].flatten()
    for j in range(lm.shape[0]):
        array = lm[j].reshape((-1, 2))
        array = np.concatenate((array, np.zeros_like(array[:,:-1])), axis=1)
        result[j] = array.dot(R)[:,:-1].flatten()
    return result

def random_rotate(data):
    angle = np.random.rand(1)[0]*2*PI - PI
    data = rotate(data, angle, axis=0)

    angle = np.random.rand(1)[0]*PI/2 - PI / 4
    data = rotate(data, angle, axis=1)

    angle = np.random.rand(1)[0]*PI/2 - PI / 4
    data = rotate(data, angle, axis=2)

    return data

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