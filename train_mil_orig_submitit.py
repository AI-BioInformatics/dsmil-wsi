# import matplotlib.pyplot as plt
# import sklearn.metrics
import openslide
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
# import torchvision.transforms.functional as VF
# from torchvision import transforms
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from skimage import exposure, io, img_as_ubyte, transform
import sys, argparse, os, copy, itertools, glob, datetime
import pandas as pd
import numpy as np
import copy
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support, confusion_matrix, accuracy_score
# from collections import OrderedDict
from models import classifiers as mil
# from sklearn.model_selection import GroupShuffleSplit
import wandb
import submitit
from dataloader import WSI
from datetime import datetime
import random

# from attention_map import test_att

def get_bag_feats(csv_file_df, args):
    '''
    @param csv_file_df: csv file contains the features column (paths of the features, 0) and the labels columns (1)
    @param args: Argument parser -> 'num_classes' for the generation of the encoding of the labels
    @return
        label: numpy array of the label encoding of the current WSI feature
        feats: numpy array of the WSI feature
    '''
    feats_csv_path = csv_file_df.iloc[0]
    df = pd.read_csv(feats_csv_path)
    # df = pd.read_csv(glob.glob(os.path.join(feats_csv_path, '*.csv'))[0])
    feats = shuffle(df).reset_index(drop=True)
    feats = feats.to_numpy()
    label = np.zeros(args.num_classes)
    if args.num_classes == 1:
        label[0] = csv_file_df.iloc[1]
    else:
        if int(csv_file_df.iloc[1]) <= (len(label) - 1):

            label[int(csv_file_df.iloc[1])] = 1
    return label, feats

def dropout_patches(feats, p):
    idx = np.random.choice(np.arange(feats.shape[0]), int(feats.shape[0]*(1-p)), replace=False)
    sampled_feats = np.take(feats, idx, axis=0)
    pad_idx = np.random.choice(np.arange(sampled_feats.shape[0]), int(feats.shape[0]*p), replace=False)
    pad_feats = np.take(sampled_feats, pad_idx, axis=0)
    sampled_feats = np.concatenate((sampled_feats, pad_feats), axis=0)
    return sampled_feats

def save_masks(model,testloader, args):
    model.eval()
    results=[]
    test_probabilities=[]
    test_predictions=[]
    test_labels=[]
    names=[]
    idxs=[]
    threshold = []
    for _,data in enumerate(testloader):
        embeddings, label, name, x, y = data
        embeddings = embeddings.cuda().squeeze(0)

        bag_feats = embeddings.cuda().squeeze(0)

        ins_prediction, bag_prediction, A, _ = model(bag_feats)

        bag_label=label.float().squeeze().cpu().numpy()
        id= name[0].split("_")[0]
        idxs.append(id)
        names.append(name)
        test_labels.append(int(bag_label))
        preds=torch.sigmoid(bag_prediction)[0].detach().cpu().numpy()[0]
        test_probabilities.append(preds)
        test_predictions.append(int(preds>args.thresholds_optimal))
        threshold.append(args.thresholds_optimal)

        attentionmap(A= A.cpu().detach().numpy(),x_coords=x.cpu().detach().numpy(),y_coords=y.cpu().detach().numpy(),name=name[0])

        attention = pd.DataFrame({'x':np.array(x).reshape(x.shape[1]), 'y': np.array(y).reshape(x.shape[1]), 'A': np.array(A.cpu().detach()).reshape(x.shape[1])})
        os.makedirs('/mnt/beegfs/work/H2020DeciderFicarra/DECIDER/attention_maps/'+args.trial+'/'+args.res+'/', exist_ok=True)
        attention.to_csv('/mnt/beegfs/work/H2020DeciderFicarra/DECIDER/attention_maps/'+args.trial+'/'+args.res+'/'+name[0]+'.csv')

        # wandb.log('A': wandb.Table())
    df=pd.DataFrame({"id":idxs,"slide_name":names,"test_labels":test_labels,"predictions":test_predictions,"probability":test_probabilities, 'thre_opt': threshold})
    wandb.log({"table":wandb.Table(columns=list(df.columns),data=df)})



def attentionmap(A,x_coords,y_coords,name):
    x_coords=x_coords.squeeze()
    y_coords=y_coords.squeeze()
    colors = [np.array([255,0,0]) for i in range(1)]
    colored_tiles = np.matmul(A[:, None], colors[0][None, :])
    colored_tiles = exposure.rescale_intensity(colored_tiles, out_range=(0, 1))
    # plot_attention_map(colored_tiles, x_coords, y_coords, name)
    color_map = np.zeros((int(np.amax(x_coords, 0)/2048+1), int(np.amax(y_coords, 0)/2048+1), 3))
    for i in range(x_coords.shape[0]):
        color_map[int(x_coords[i]/2048),int( y_coords[i]/2048)] = colored_tiles[i]
    color_map = transform.resize(color_map, (color_map.shape[0], color_map.shape[1]), order=0)
    #color_map= transform.rotate(color_map,270)
    #color_map= np.flipud(color_map)
    # wandb.log({'color_map': wandb.Table(columns=color_map)})
    wandb.log({name:wandb.Image(img_as_ubyte(color_map))})
    #io.imsave(os.path.join(name+'mappa.png'), img_as_ubyte(color_map))

def plot_attention_map(color_map, x_coords, y_coords, name):
    cmap = "coolwarm"
    if isinstance(cmap, str):
        newcmp = plt.get_cmap(cmap)
    slide_path = os.path.join('mnt/beegfs/work/H2020DeciderFicarra/DECIDER/WSI_24_11_2022/'+name+'.mrxs')
    wsi = openslide.open_slide(slide_path)

    probs = [1]*len(color_map)
    coords = [(int(el[0]), int(el[1])) for el in zip(x_coords, y_coords)]
    vis_level = wsi.get_best_level_for_downsample(8)
    downsample = wsi.level_downsamples[vis_level]
    scale = [1/downsample, 1/downsample]
    region_size = wsi.level_dimensions[vis_level]
    top_left = (0,0)
    patch_size = (256, 256)
    patch_size = np.ceil(np.array(patch_size) * np.array(scale)).astype(int)
    coords = np.ceil(coords * np.array(scale)).astype(int)

    heatmap = np.full(np.flip(region_size), 0).astype(int)

    for idx, prob in zip(range(len(coords)), probs):
        score = color_map[idx]
        coord = coords[idx]
        if prob == '0.0':
            continue
        heatmap[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]] = score

    img = np.array(wsi.read_region(top_left, vis_level))

    return None

def init_seed(args):
    import torch
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def multi_label_roc(labels, predictions, num_classes, pos_label=1):
    fprs = []
    tprs = []
    thresholds = []
    thresholds_optimal = []
    aucs = []
    # if len(predictions.shape) == 1:
    #     predictions = predictions[:, None]
    # for c in range(0, num_classes):
    #     label = labels[:, c]
    #     prediction = predictions[:, c]
    #     fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
    #     fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
    #     c_auc = roc_auc_score(label, prediction)
    #     aucs.append(c_auc)
    #     thresholds.append(threshold)
    #     thresholds_optimal.append(threshold_optimal)

    fpr, tpr, threshold = roc_curve(labels, predictions, pos_label=1)
    fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
    c_auc = roc_auc_score(labels, predictions)
    aucs.append(c_auc)
    thresholds.append(threshold)
    thresholds_optimal.append(threshold_optimal)

    return aucs, thresholds, thresholds_optimal

def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]

def train(train_df, milnet, criterion, optimizer, args):
    """
    @param train_df: pandas DataFrame that contains the list of the feature paths
    @param milnet: milnet model
    @param criterion: Loss
    @param optimizer: Optimizer
    @param args: Argument Parser -> feat_size
                                 -> num_classes
    @return:
        current loss
    """
    milnet.train()
    total_loss = 0
    Tensor = torch.cuda.FloatTensor
    for i, slide in enumerate(train_df):
        optimizer.zero_grad()
        bag_feats, bag_label, name, x, y = slide

        bag_label = bag_label.float().cuda()
        # print(bag_feats)
        # print(bag_feats.shape)
        # print(type(bag_feats))
        # print(np.array(bag_feats))
        bag_feats = dropout_patches(np.array(bag_feats).reshape(bag_feats.shape[1], bag_feats.shape[2]), args.dropout_patch)
        bag_feats = Variable(Tensor([bag_feats]))
        bag_feats = bag_feats.view(-1, args.feats_size)
        # bag_feats = bag_feats.cuda().squeeze(0)

        ins_prediction, bag_prediction, A, _ = milnet(bag_feats)

        max_prediction, _ = torch.max(ins_prediction, 0)
        bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
        max_loss = criterion(max_prediction.view(1, -1), bag_label.view(1, -1))
        loss = 0.5 * bag_loss + 0.5 * max_loss
        # loss = loss.mean()
        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss.item()
        wandb.log({'train_loss_step': loss.item()})

        #print('Training bag [%d/%d] bag loss: %.4f' % (i, len(train_df), loss.item()))
        # sys.stdout.write('\r Training bag [%d/%d] bag loss: %.4f' % (i, len(train_df), loss.item()))
    return total_loss / len(train_df)


def test(test_df, milnet, criterion, args):
    milnet.eval()
    total_loss = 0
    test_labels = []
    test_predictions = []
    with torch.no_grad():
        for i, slide in enumerate(test_df):
            bag_feats, bag_label, name, _, _= slide
            bag_label = bag_label.float().cuda()
            bag_feats = bag_feats.cuda().squeeze(0)

            ins_prediction, bag_prediction, _, _ = milnet(bag_feats)
            max_prediction, _ = torch.max(ins_prediction, 0)
            bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
            max_loss = criterion(max_prediction.view(1, -1), bag_label.view(1, -1))
            loss = 0.5 * bag_loss + 0.5 * max_loss
            # loss = loss.mean()
            total_loss = total_loss + loss.item()
            # print('Val bag [%d/%d] bag loss: %.4f' % (i, len(test_df), loss.item()))
            # wandb.log({'test_loss_step': loss.item()})
            # sys.stdout.write('\r Val bag [%d/%d] bag loss: %.4f' % (i, len(test_df), loss.item()))

            test_labels.extend([bag_label.float().squeeze().cpu().numpy()])
            if args.average:
                pred = [(0.5 * torch.sigmoid(max_prediction) + 0.5 * torch.sigmoid(bag_prediction)).squeeze().cpu().numpy()]
                test_predictions.extend(pred)
            else:
                pred = [(0.0*torch.sigmoid(max_prediction) + 1.0*torch.sigmoid(bag_prediction)).squeeze().cpu().numpy()]
                test_predictions.extend(pred)

    #         temp = {'slides': name[0],
    #                 'pred': pred[0],
    #                 'label': bag_label.squeeze().cpu().numpy()}
    #         analisi.append(temp)
    # analisi = pd.DataFrame(analisi)

    test_labels = np.array(test_labels)
    test_predictions = np.array(test_predictions)

    auc_value, _, thresholds_optimal = multi_label_roc(test_labels, test_predictions, args.num_classes, pos_label=1)
    # analisi['thres_opt'] = thresholds_optimal
    temp_test_predictions = copy.deepcopy(test_predictions)
    if args.num_classes == 1:
        class_prediction_bag = copy.deepcopy(test_predictions)
        class_prediction_bag[test_predictions >= thresholds_optimal[0]] = 1
        class_prediction_bag[test_predictions < thresholds_optimal[0]] = 0
        test_predictions = class_prediction_bag
        test_labels = np.squeeze(test_labels)
    else:
        for i in range(args.num_classes):
            class_prediction_bag = copy.deepcopy(test_predictions[:, i])
            class_prediction_bag[test_predictions[:, i] >= thresholds_optimal[i]] = 1
            class_prediction_bag[test_predictions[:, i] < thresholds_optimal[i]] = 0
            test_predictions[:, i] = class_prediction_bag
    # analisi['class_pred'] = test_predictions
    bag_score = 0
    for i in range(0, len(test_df)):
        bag_score = np.array_equal(test_labels[i], test_predictions[i]) + bag_score
    avg_score = bag_score / len(test_df)
    return total_loss / len(test_df), avg_score, test_labels, test_predictions, auc_value, thresholds_optimal
        # , analisi

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)

def main():
    parser = argparse.ArgumentParser(description='Train DSMIL on 20x patch features learned by SimCLR')

    # Optimization phase
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--lr', default=0.0002, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--weight_decay', default=5e-3, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--non_linearity', default=1, type=float, help='Additional nonlinear operation [0]')
    parser.add_argument('--dropout_node', default=0, type=float, help='Bag classifier dropout rate [0]')

    parser.add_argument('--model', default='dsmil', type=str, help='MIL model [dsmil]')
    parser.add_argument('--init', default='random', type=str)
    parser.add_argument('--dropout_patch', default=0, type=float, help='Patch dropout rate [0]')
    parser.add_argument('--project', default="pfi_45_180_resx5_hyper", type=str, help='project name')
    parser.add_argument('--seed', default=5, type=int, help='Additional nonlinear operation [0]')

    parser.add_argument('--export_scores', type=int, default=1)
    parser.add_argument('--results_path', type=str, default='/mnt/beegfs/work/H2020DeciderFicarra/DECIDER/results')

    # Data
    # parser.add_argument('--tissue_type', type=str, default='All', help='[All, Sep]')
    # parser.add_argument('--dim', type=str, default='without_dim', help='[with_dim, without_dim]')
    parser.add_argument('--res', type=str, default='x5', help='[with_dim, without_dim]')
    parser.add_argument('--single_tissue', type=str, default=None, help="['Ome','Adn', None]")
    parser.add_argument('--trial', type=str, default='pfi_45_180')
    parser.add_argument('--feats_size', default=384, type=int, help='Dimension of the feature size')
    # parser.add_argument('--slide_dir', type=str, default='/mnt/beegfs/work/H2020DeciderFicarra/DECIDER/WSI_24_11_2022')
    # parser.add_argument('--coords_path', type=str, default='/mnt/beegfs/work/H2020DeciderFicarra/DECIDER/clamdeciderintervalx20_20221124/patches')

    # Phases
    # Training phase
    parser.add_argument('--training', type=bool, default=True)
    parser.add_argument('--num_epochs', default=200, type=int, help='Number of total training epochs [40|200]')
    parser.add_argument('--num_classes', default=1, type=int, help='Number of output classes [2]')
    parser.add_argument('--dropout', default=False)
    parser.add_argument('--average', default=True)

    # Attention pipeline
    parser.add_argument('--attention', type=bool, default=False)
    parser.add_argument('--checkpoint', type=str, default='/mnt/beegfs/work/H2020DeciderFicarra/DECIDER/weights/All/without_dim/1a_5/split_0_best_model.pth')
    parser.add_argument('--thres', type=float, default=0.6819368600845337)
    parser.add_argument('--split', type=str, default=0)
    parser.add_argument('--eval_trial', type=str, default='1a_5')
    parser.add_argument('--class_name', type=list, default=['short', 'long'])
    parser.add_argument('--map_path', type=str, default='/mnt/beegfs/work/H2020DeciderFicarra/DECIDER/results/output')
    parser.add_argument('--data_path', type=str, default='/mnt/beegfs/work/H2020DeciderFicarra/DECIDER/features/trials_1/without_dim/All_pos')

    args = parser.parse_args()
    # cases=["All_sep_without_dim_pfi_filt0_pfi_thr210","All_sep_without_dim_pfi_filt30_pfi_thr210","All_without_dim_pfi_filt0_pfi_thr210","All_without_dim_pfi_filt30_pfi_thr210",
    # "All_with_dim_pfi_filt0_pfi_thr210","All_sep_with_dim_pfi_filt30_pfi_thr210","All_sep_with_dim_pfi_filt0_pfi_thr210","All_with_dim_pfi_filt30_pfi_thr210"]

    params = {'res': ['x5', 'x5s_2', 'x5s_3']*3}

    temp = copy.copy(args)
    args = []
    for res in params['res']:
        temp.res = res
        temp2 = copy.copy(temp)
        args.append(temp2)

    log_folder = 'log_test'
    executor = submitit.AutoExecutor(folder=log_folder)
    executor.update_parameters(
            mem_gb=10,
            slurm_gpus_per_task=1,
            nodes=1,
            # exclude="aimagelab-srv-10,rezzonico,gervasoni",
            timeout_min=100,  # max is 60 * 72
            # Below are cluster dependent parameters
            slurm_partition="prod",
            slurm_signal_delay_s=120,
            slurm_array_parallelism=3)
    executor.map_array(process_experiment,args)
    # process_experiment(args)


def process_experiment(args):

    # ================================================================================================================
    # Prepare dataset
    # Preparation of splitting csv files
    init_seed(args)

    torch.manual_seed(args.seed)
    root = "/mnt/beegfs/work/H2020DeciderFicarra/decider/feats/"+args.trial+'/'+args.res

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_dataset = WSI(root, "train")
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=1,
                                  shuffle=True,
                                  num_workers=0,
                                  worker_init_fn=seed_worker,
                                  generator=g)

    test_dataset = WSI(root, "test")
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=1,
                                 shuffle=True,
                                 num_workers=0,
                                 worker_init_fn=seed_worker,
                                 generator=g)

    i_classifier = mil.FCLayer(in_size=args.feats_size, out_size=args.num_classes).cuda()
    b_classifier = mil.BClassifier(input_size=args.feats_size, output_class=args.num_classes, dropout_v=args.dropout_node).cuda()
    milnet = mil.MILNet(i_classifier, b_classifier).cuda()
    # milnet.apply(init_weights)
    state_dict_weights = torch.load('/homes/nbartolini/PyCharm_projects/dasmil-extension/init.pth')
    try:
        milnet.load_state_dict(state_dict_weights, strict=False)
    except:
        del state_dict_weights['b_classifier.v.1.weight']
        del state_dict_weights['b_classifier.v.1.bias']
        milnet.load_state_dict(state_dict_weights, strict=False)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(milnet.parameters(), lr=args.lr, betas=(0.5, 0.9),
                                 weight_decay=args.weight_decay)
    # optimizer = torch.optim.SGD(milnet.parameters(), lr=0.001, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, 0.000005)
    if args.training:
        # ===========================================================================================================
        # Training Phase
        # Prepare models
        # if args.init == 'random':
        #     milnet.apply(init_weights)
        # elif args.init == 'pre_trained':
        #     if args.model == 'dsmil':
        #         state_dict_weights = torch.load('/homes/nbartolini/PyCharm_projects/dino/init.pth')
        #         try:
        #             milnet.load_state_dict(state_dict_weights, strict=False)
        #         except:
        #             del state_dict_weights['b_classifier.v.1.weight']
        #             del state_dict_weights['b_classifier.v.1.bias']
        #             milnet.load_state_dict(state_dict_weights, strict=False)


        run = wandb.init(project=args.project, reinit=True, name=args.trial + '_res' +args.res+'_lr'+str(args.lr)+'_epoc'+str(args.num_epochs)+'_dropout'+str(args.dropout_node), settings=wandb.Settings(start_method="fork"))
        wandb.config.update(args)

        # wandb.log({"Train - Short": train_dataset.num_class(cl=0),
        #            "Train - Long": train_dataset.num_class(cl=1),
        #            "Train - Slides": len(train_dataset),
        #            "Train - Patients": train_dataset.num_pat()
        #            })
        # wandb.log({"Test - Short": test_dataset.num_class(cl=0),
        #            "Test - Long": test_dataset.num_class(cl=1),
        #            "Test - Slides": len(test_dataset),
        #            "Test - Patients": test_dataset.num_pat()
        #                })
        score_max = 0
        best_test_predictions = 0
        best_test_labels = 0
        now = datetime.now()
        print(f'Start session: {now}')
        for epoch in range(args.num_epochs):

            train_loss_bag = train(train_dataloader, milnet, criterion, optimizer, args)  # iterate all bags
            test_loss_bag, avg_score, test_labels, test_predictions, auc_value, thresholds_optimal = test(test_dataloader, milnet, criterion, args)
            print('\r Epoch [%d/%d] train loss: %.4f test loss: %.4f, average score: %.4f, AUC: %.4f' % (epoch, args.num_epochs, train_loss_bag, test_loss_bag, avg_score, auc_value[0]))
            scheduler.step()
            wandb.log({"epoch": epoch,
                       "train_loss_bag": train_loss_bag,
                       "test_loss_bag": test_loss_bag,
                       "test_avg_score": avg_score,
                       "test_auc_score": auc_value[0],
                       "lr": scheduler.get_last_lr()[0]
                       })

            os.makedirs(os.path.join('/mnt/beegfs/work/H2020DeciderFicarra/DECIDER/weights', args.trial, args.res), exist_ok=True)
            torch.save(milnet.state_dict(), os.path.join('/mnt/beegfs/work/H2020DeciderFicarra/DECIDER/weights', args.trial, args.res, 'epoc' + str(epoch) + '_model.pth'))

            current_score = (float(auc_value[0]) + float(avg_score))/2
            if epoch>30:
                if current_score >= score_max:
                    # save model
                    torch.save(milnet.state_dict(), os.path.join(wandb.run.dir, "model.pt"))
                    wandb.save(os.path.join(wandb.run.dir, "model.pt"))

                    args.thresholds_optimal = thresholds_optimal
                    avg_score_max = avg_score
                    auc_score_max = auc_value[0]
                    score_max = (float(auc_value[0]) + float(avg_score))/2
                    wandb.log({'avg_score_max': avg_score_max,
                               'auc_score_max': auc_score_max})
                    best_test_labels = test_labels
                    best_test_predictions = test_predictions

                    save_masks(milnet, test_dataloader, args)
        wandb.sklearn.plot_confusion_matrix(best_test_labels, best_test_predictions, ['short', 'long'])


if __name__ == '__main__':
    main()
