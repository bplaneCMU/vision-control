import copy, time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import optimizer
from torch.utils.data import Dataset, DataLoader, dataloader, dataset, random_split
from torchvision import transforms, utils
import matplotlib.pyplot as plt
from sys import argv
import os
from math import pi as PI
os.environ['KMP_DUPLICATE_LIB_OK']='True'


"""
    Gesture Classification Network Model
"""
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.ModuleList()
        self.front = nn.Linear(42, 256)

        # Normal Hourglass
        for _ in range(2):
            self.layers.append(nn.BatchNorm1d(256))
            self.layers.append(nn.Linear(256, 128))
            self.layers.append(nn.Linear(128, 64))
            self.layers.append(nn.Linear(64, 64))
            self.layers.append(nn.Linear(64, 64))
            self.layers.append(nn.Linear(64, 128))
            self.layers.append(nn.Linear(128, 256))

        self.back = nn.Linear(256, 5)

    def forward(self, x):
        x = F.normalize(x)
        x = self.front(F.relu(x.float()))

        xt = {}
        for f in self.layers:
            if type(f) != type(nn.Linear(1, 1)):
                x = f(x)
            elif f.in_features > f.out_features:
                x = F.relu(f(x))
                xt[f.out_features] = x
            elif f.in_features < f.out_features:
                x = F.relu(f(x + xt[f.in_features]))
            else:
                x = F.relu(f(x))
        
        x = self.back(x)
        x = F.softmax(x, dim=1).float()
        return x

"""
    Dataset/Dataloader Setup
"""
class Landmarks(Dataset):

    def __init__(self, csv_file):
        self.data = np.genfromtxt(csv_file, delimiter=',')[:, :-2]
        self.labels = np.genfromtxt(csv_file, delimiter=",")[:, -2:-1]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        return random_rotate(self.data[index, :]), LABEL_TRANSFORM2[self.labels.astype(int)[index, 0]]

def random_rotate(data):
    angle = np.random.rand(1)[0]*2*PI - PI
    data = rotate(data, angle, axis=0)

    angle = np.random.rand(1)[0]*PI/2 - PI / 4
    data = rotate(data, angle, axis=1)

    angle = np.random.rand(1)[0]*PI/2 - PI / 4
    data = rotate(data, angle, axis=2)

    return data

LABEL_TRANSFORM = np.array([
    0, 0,
    1, 1,
    2, 2,
    3, 3,
    4, 4,
    5, 5,
    6, 6,
    7, 7,
    8, 8,
    9, 9,
    10, 10,
    11, 11,
    12, 12,
])

LABEL_TRANSFORM2 = np.array([
    0, 0, # Left Click
    1, 1,
    1, 1,
    2, 2, # Scroll
    1, 1,
    3, 3, # Move Mouse
    1, 1,
    1, 1,
    1, 1,
    1, 1,
    4, 4, # Right Click
    1, 1,
    1, 1,
])

LM = Landmarks("./data/augmented.csv")
LM_train, LM_test = random_split(LM, [len(LM)  - (len(LM)) // 4, len(LM) // 4])

dl_train = DataLoader(dataset=LM_train, batch_size=32, shuffle=True)
dl_test = DataLoader(dataset=LM_test, batch_size=32, shuffle=True)

dataloaders = {
    "train": dl_train,
    "val": dl_test,
}

dataset_sizes = {
    "train": len(LM_train),
    "val": len(LM_test),
}

"""
    Main Functional Script to train and test
"""
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

def train_model(model, criterion, optimizer, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    trackpath = "./data/loss_{:.2f}.csv".format(time.time())
    trackfile = open(trackpath, "a")

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)

                    loss = criterion(outputs.float(), labels.long())

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]


            print('{} Loss: {:.4f} Acc: {:.4f} [{}/{}]]'.format(
                phase, epoch_loss, epoch_acc, running_corrects, dataset_sizes[phase]))

            if phase == 'val':
                trackfile.write("{},{},\n".format(epoch, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

if __name__ == "__main__":
    model = None
    optimizer = None
    if len(argv) == 1:
        model_save_path = "./data/model_{:.4f}".format(time.time())
        model = Net()
        optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01, lr_decay=0)
    else:
        try:
            model = Net()
            model_save_path = "./data/" + argv[1]
            model.load_state_dict(torch.load(model_save_path))
            optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01, lr_decay=0)
            optimizer.load_state_dict(torch.load(model_save_path + ".dict"))
        except:
            print("Could not find model file: ./data/" + argv[1])
            quit()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    model = train_model(model, criterion, optimizer, num_epochs=50)
    torch.save(model.state_dict(), model_save_path)
    torch.save(optimizer.state_dict(), model_save_path + ".dict")

