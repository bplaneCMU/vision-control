from model import Net
import torch
from functools import wraps
from time import time


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r args:[%r, %r] took: %2.4f sec' % (f.__name__, args, kw, te-ts))
        return result
    return wrap

class Evaluator():

    def __init__(self, model_path=""):
        self.model = Net()
        if model_path != "":
            self.model.load_state_dict(torch.load(model_path))
        
        self.model.eval()


    def load_model_parameters(self, model_path):
        self.model.load_state_dict(torch.load(model_path))

    # @timing
    def eval(self, x):
        if type(x) == list:
            x = torch.tensor([x])
        if type(x) != torch.Tensor:
            x = torch.tensor(x)
        self.model.eval()
        confidence = self.model(x.float())
        _, pred = torch.max(confidence, 1)
        
        return pred.numpy(), float(confidence[0][pred])