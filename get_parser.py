import argparse
import copy
import itertools
import torch.nn as nn

def get_parser():

    parser = argparse.ArgumentParser(description='Train DSMIL on 20x patch features learned by SimCLR')

    # Initialization phase
    parser.add_argument('--seed', default=5, type=int, help='Additional nonlinear operation [0]')
    parser.add_argument('--model', default='dsmil', type=str, help='MIL model [dsmil]')
    parser.add_argument('--project', default="gridsearch2_s1", type=str, help='project name')
    parser.add_argument('--parallel_job', default=True, type=bool)

    # Optimization phase
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--lr', default=0.0002, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--weight_decay', default=5e-3, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--non_linearity', default=1, type=float, help='Additional nonlinear operation [0]')
    parser.add_argument('--dropout_node', default=0, type=float, help='Bag classifier dropout rate [0]')
    parser.add_argument('--dropout_patch', default=0, type=float, help='Patch dropout rate [0]')

    # Data
    parser.add_argument('--trial', type=str, default='pfi_45_180')
    parser.add_argument('--res', type=str, default='x5', help='[x5, x5s_2, x5s_3, x10, x20]')
    parser.add_argument('--feats_size', default=384, type=int, help='Dimension of the feature size')
    parser.add_argument('--epoch_min', default=10, type=int, help='After this number of epoch, the model checkpoint will be saved')
    # parser.add_argument('--slide_dir', type=str, default='/mnt/beegfs/work/H2020DeciderFicarra/DECIDER/WSI_24_11_2022')
    # parser.add_argument('--coords_path', type=str, default='/mnt/beegfs/work/H2020DeciderFicarra/DECIDER/clamdeciderintervalx20_20221124/patches')

    # Phases
    # Training phase
    parser.add_argument('--training', type=bool, default=True)
    parser.add_argument('--num_epochs', default=100, type=int, help='Number of total training epochs [40|200]')
    parser.add_argument('--num_classes', default=1, type=int, help='Number of output classes [2]')
    parser.add_argument('--average', default=True)

    # Attention pipeline
    parser.add_argument('--save_attention_map', default=True, type=bool)
    args = parser.parse_args()

    return args


def args_parallel_jobs(args, params:dict):
    temp = copy.copy(args)
    args = []
    key_list = list(params.keys())
    for c in itertools.product(*list(params.values())):
        for i, k in enumerate(key_list):
            setattr(temp, k, c[i])
        temp2 = copy.copy(temp)
        args.append(temp2)
    return args


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)