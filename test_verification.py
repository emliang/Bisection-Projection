from utils.verification_utils import *
from default_args import *
from utils.training_utils import unsupervised_training_meip
from utils.nn_utils import *
from training_all import load_instance
args = config()

def __main__():
    data, result_save_dir, model_save_dir = load_instance(args)
    prob = args['probType']
    ecc = True 
    model = torch.load(model_save_dir + f'/verification_model_{ecc}.pth', map_location=data.device)
    nn_specification(model.net, prob, data)

if __name__ == '__main__':
    __main__()