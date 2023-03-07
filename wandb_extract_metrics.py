import copy
import json

import numpy as np
import pandas as pd
import wandb
import argparse
import glob


def false_predict_info(false_predict_dict, prob=[0.2, 0.8]):
    '''

    @param df:
    @param false_predict_dict:
    @param prob:
    @return: dictionary for report
    '''
    dict = {'length': len(false_predict_dict),
            'n_high_security': len(false_predict_dict[np.logical_and(false_predict_dict['probability'] < prob[0],
                                                                      false_predict_dict['probability'] < prob[1])][
                'slide_name'].values),
            'slides_high_security': list(false_predict_dict[np.logical_and(false_predict_dict['probability'] < prob[0],
                                                                      false_predict_dict['probability'] < prob[1])][
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

    report['false_short'] = false_predict_info(false_short)
    report['false_long'] = false_predict_info(false_long)

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

def process(args):
    api = wandb.Api()

    runs = api.runs(args.api)
    root = '/homes/nbartolini/PyCharm_projects/dsmil-wsi/wandb/*/files/'
    false_all = []
    for run in runs:
        path = glob.glob(root + run.summary['table']['path'])[0]
        with open(path, 'r') as f:
            table = json.load(f)
        df = pd.DataFrame(table['data'], columns=table['columns'])
        report = get_info(df)
        print(report)
        temp = {'threshold_optimal':report['threshold_optimal'],
                'false_short length': report['false_short']['length'],
                'false_short n_high_security': report['false_short']['n_high_security'],
                'false_short slides_high_security': report['false_short']['slides_high_security'],
                'false_long length': report['false_long']['length'],
                'false_long n_high_security': report['false_long']['n_high_security'],
                'false_long slides_high_security': report['false_long']['slides_high_security'],}
        false_all.append(temp)
    false_all = pd.DataFrame(false_all)
    print(false_all)
    false_all.to_csv('report_false_prediction.csv')

if __name__ == '__main__':
    main()
