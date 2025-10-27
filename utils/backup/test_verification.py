from utils.verification_utils import *
from default_args import *
from utils.training_utils import unsupervised_training_meip
from utils.nn_utils import *
from training_all import load_instance
args = config()

def __main__():
    data, result_save_dir, model_save_dir = load_instance(args)
    prob = args['probType']
    paras = args['ipnn_para']
    n_layer = paras['n_layer']
    ecc = paras['minimum_ecc']
    c_dim = data.xdim
    n_dim = len(data.partial_vars_idx)
    #### IPNN: input to multiple IPs
    try:
        model = torch.load(model_save_dir + f'/verification_model_{True}.pth', map_location=data.device)
    except:
        model = ResNet(c_dim, n_dim, data.intrin_dim, n_layer, act=None).to(data.device)
        c_tensor = torch.rand([paras['c_samples'], c_dim]).to(data.device)
        c_tensor = c_tensor * (data.input_U - data.input_L) + data.input_L
        model, penalty_list, eccentricity_list = unsupervised_training_meip(model, data, n_dim, c_tensor, paras)
        torch.save(model, model_save_dir + f'/verification_model_{ecc}.pth')

    nn_specification(model.net, prob, data)



if __name__ == '__main__':
    __main__()