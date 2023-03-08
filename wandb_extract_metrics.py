import copy
import json

import numpy as np
import pandas as pd
import wandb
import argparse
import glob


def predict_info(predict_dict, prob=[0.2, 0.8]):
    '''

    @param df:
    @param predict_dict:
    @param prob:
    @return: dictionary for report
    '''
    dict = {'length': len(predict_dict),
            'n_high_security': len(predict_dict[np.logical_and(predict_dict['probability'] < prob[0],
                                                               predict_dict['probability'] < prob[1])][
                'slide_name'].values),
            'slides_high_security': list(predict_dict[np.logical_and(predict_dict['probability'] < prob[0],
                                                                     predict_dict['probability'] < prob[1])][
                'slide_name'].values)}

    return dict


def get_info(df):
    '''

    @param df:
    @return:
    '''
    report = {}
    report['threshold_optimal'] = df['thre_opt'][0][0]
    true_label = df['test_labels']
    pred_label = (df['predictions'] > df['thre_opt'][0][0]).astype(int)
    false_short = df[true_label - pred_label == -1]
    false_long = df[df['test_labels'] - (df['predictions'] > df['thre_opt'][0][0]).astype(int) == 1]

    true_short = df[np.logical_and(true_label == 0,
                                   pred_label == 0)]
    true_long = df[np.logical_and(true_label == 1,
                                  pred_label == 1)]
    #TODO: multiple_predict_info method
    report['true_short'] = predict_info(true_short)
    report['true_long'] = predict_info(true_long)
    report['false_short'] = predict_info(false_short)
    report['false_long'] = predict_info(false_long)

    return report


def baseline(args):
    '''

    @param args:
    @return:
    '''
    run_base = args.run_base.split('-')
    return run_base

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--api', default="nbartolini/gridsearch_seed_var")
    args = parser.parse_args()
    return args

def main():
    args = get_parser()
    process(args)


def get_cross_info(false_all):
    '''

    @param false_all:
    @return:
    '''
    new_list = []
    for list_slides in false_all:
        new_list.append(list(np.reshape(list_slides, (len(list_slides)))))
    intersection = set.intersection(*map(set,new_list))
    print(list(intersection))

def single_repo(report, pred:bool=True):
    if pred:
        short = 'true_short'
        long = 'true_long'
    else:
        short = 'false_short'
        long = 'false_long'
    repo = {'threshold_optimal': report['threshold_optimal'],
            'short length': report[short]['length'],
            'short n_high_security': report[short]['n_high_security'],
            'short slides_high_security': report[short]['slides_high_security'],
            'long length': report[long]['length'],
            'long n_high_security': report[long]['n_high_security'],
            'long slides_high_security': report[long]['slides_high_security'], }
    return repo

def process(args):
    api = wandb.Api()

    runs = api.runs(args.api)
    root = '/homes/nbartolini/PyCharm_projects/dsmil-wsi/wandb/*/files/'
    false_all = []
    true_all = []
    for run in runs:
        path = glob.glob(root + run.summary['table']['path'])[0]
        with open(path, 'r') as f:
            table = json.load(f)
        df = pd.DataFrame(table['data'], columns=table['columns'])
        report = get_info(df)
        #TODO: multiple_repo method
        false_single_repo = single_repo(report, pred=False)
        true_single_repo = single_repo(report, pred=True)
        false_all.append(false_single_repo)
        true_all.append(true_single_repo)
    false_all = pd.DataFrame(false_all)
    true_all = pd.DataFrame(true_all)
    false_all.to_csv('report_false_prediction.csv')
    true_all.to_csv('report_true_prediction.csv')

if __name__ == '__main__':
    main()
