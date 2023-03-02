import gc
import os

import openslide
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from PIL import Image
import cv2
import argparse
import pandas as pd
from skimage import exposure, io, img_as_ubyte, transform


parser = argparse.ArgumentParser(description='features reduction and clusterization')

parser.add_argument('--output_dir', type=str, default='/mnt/beegfs/work/H2020DeciderFicarra/DECIDER/attention_maps/pfi_45_180/x5/')
parser.add_argument('--scores_path', type=str, default='/mnt/beegfs/work/H2020DeciderFicarra/DECIDER/attention_maps/pfi_45_180/x5/H160_iOme_PE_IIA_HE_rscn_0.csv' )
parser.add_argument('--slide_dir', type=str, default='/mnt/beegfs/work/H2020DeciderFicarra/DECIDER/WSI_24_11_2022')
parser.add_argument('--coords_path', type=str, )
parser.add_argument('--wsi_list', type=str, default='/mnt/beegfs/work/H2020DeciderFicarra/DECIDER/features/trials_1/without_dim/Sep')

parser.add_argument('--exp_name', type=str, default="heatmap_computation")


args = parser.parse_args()

def compute_heatmap(scores_path, wsi_list, slide_dir, out_dir, run_number="", res='x5'):
    """
    @type: scores_path -> attention map path of the current WSI
    @type: slide_dir -> slide path of the current WSI
    @rtype: output_dir -> directory for the storing images

    """
    print("creating heatmaps\n")

    df = pd.read_csv(scores_path, index_col=0)
    coords = df.loc[:, ['x', 'y']]
    A = df.loc[:, 'A']

    probs = [1] * len(A)
    colors = [np.array([255,0,0]) for i in range(1)]
    colored_tiles = np.matmul(np.array(A[:, None]), colors[0][None, :])
    colored_tiles = exposure.rescale_intensity(colored_tiles, out_range=(0, 1))


    encoder = {'x5': (512, 512),
               'x10': (256, 256),
               'x20': (128, 128)}
    patch_size = encoder[res]

    # tiles = scores_dict.keys()
    # valid_wsi = set(["_".join(tile.split(".")[0].split("_")[:-2]) for tile in tiles])
    sample = 'H160_iOme_PE_IIA_HE_rscn'
    slide_path = os.path.join(slide_dir, sample + ".mrxs")
    wsi = openslide.open_slide(slide_path)
    # if categorical:
    #     newcmp = colors.ListedColormap(['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:brown', 'tab:pink', 'tab:olive', 'tab:purple'])
    #
    # #instead if they are continous like probablity score you use this map
    # else:
    cmap = "coolwarm"
    if isinstance(cmap, str):
        newcmp = plt.get_cmap(cmap)

    vis_level = wsi.get_best_level_for_downsample(4)

    # here we scale to the desired level of downsampling
    downsample = wsi.level_downsamples[vis_level]
    scale = [1 / downsample, 1 / downsample]

    region_size = wsi.level_dimensions[vis_level]
    top_left = (0, 0)

    # patch_size = np.ceil(np.array(patch_size) * np.array(scale)).astype(int)
    coords = np.ceil(coords * np.array(scale)).astype(int)

    heatmap = np.full(np.flip(region_size), 0).astype(float)

    for idx, prob in zip(range(len(coords)), probs):
        score = colored_tiles[idx]
        coord = coords.iloc[idx].values
        if prob == '0.0':
            continue

        heatmap[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0]] += score[0]
    # tmp = wsi.get_thumbnail(region_size)
    # tmp.save(os.path.join(out_dir, sample + "_raw.jpeg"), "JPEG")

    img = np.array(wsi.read_region(top_left, vis_level, region_size).convert("L"))
    img = np.stack((img,) * 3, axis=-1)

    for idx, prob in zip(range(len(coords)), probs):
        coord = coords.iloc[idx].values
        if prob == 0.0:
            continue

        raw_block = heatmap[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0]]

        img_block = img[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0]].copy()

        color_block = (newcmp(raw_block) * 255)[:, :, :3].astype(np.uint8)

        alpha = 0.3
        img_block = cv2.addWeighted(color_block, alpha, img_block, 1 - alpha, 0, img_block)

        img[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0]] = img_block.copy()
        # not sure if this is necessary
    del heatmap
    gc.collect()

    img = Image.fromarray(img)
    w, h = img.size
    img.save(os.path.join(out_dir, f"{sample}_heatmap_clusters{run_number}.jpeg"), "JPEG")
    print('Heatmap saved')

            # for sample in wsi_list:
    #     if sample not in valid_wsi:
    #         print(sample)
    #         continue
    #     coords_str = [tile.split(".")[0].split("_")[-2:] for tile in tiles if tile.startswith(sample)]
    #     coords = [(int(elem[0]), int(elem[1])) for elem in coords_str]


        # scores = list(scores_dict.values())
        #
        # slide_path = os.path.join(slide_dir, sample + ".mrxs")
        # wsi = openslide.open_slide(slide_path)
        #
        # patch_size = patch_size
        # vis_level = wsi.get_best_level_for_downsample(8)
        #
        # #if scores are categorical like cluster membership you have a fixed list of colors
        # if categorical:
        #     newcmp = colors.ListedColormap(['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:brown', 'tab:pink', 'tab:olive', 'tab:purple'])
        #
        # #instead if they are continous like probablity score you use this map
        # else:
        #     cmap = "coolwarm"
        #     if isinstance(cmap, str):
        #         newcmp = plt.get_cmap(cmap)

        # if mapStrat == "one":
        #     # cmap = "Spectral"
        #     # cmap = "YlOrRd"
        #     cmap = "coolwarm"
        #     if isinstance(cmap, str):
        #         cmap = plt.get_cmap(cmap)
        #
        # else:
        #     cmap_pos = "Greens"
        #     if isinstance(cmap_pos, str):
        #         cmap_pos = plt.get_cmap(cmap_pos)
        #     cmap_neg = "Reds"
        #     if isinstance(cmap_neg, str):
        #         cmap_neg = plt.get_cmap(cmap_neg)

        #Downsample level: 8 is the preferred one, you can see the result but is not so memory consuming,
        # with smaller value the job could go out of memory
        # vis_level = wsi.get_best_level_for_downsample(32)
        #
        # #here we scale to the desired level of downsampling
        # downsample = wsi.level_downsamples[vis_level]
        # scale = [1 / downsample, 1 / downsample]
        #
        # region_size = wsi.level_dimensions[vis_level]
        # top_left = (0, 0)
        #
        # patch_size = np.ceil(np.array(patch_size) * np.array(scale)).astype(int)
        # coords = np.ceil(coords * np.array(scale)).astype(int)
        #
        # heatmap = np.full(np.flip(region_size), 0).astype(int)
        #
        # for idx, prob in zip(range(len(coords)), probs):
        #     score = scores[idx]
        #     coord = coords[idx]

            # here you can decide to jump the patch if probs is too low
        #     if prob == '0.0':
        #         continue
        #
        #     heatmap[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0]] = score
        #
        # tmp = wsi.get_thumbnail(region_size)
        # tmp.save(os.path.join(out_dir, sample + "_raw.jpeg"), "JPEG")
        #
        # img = np.array(wsi.read_region(top_left, vis_level, region_size).convert("L"))
        # img = np.stack((img,) * 3, axis=-1)
        #
        # for idx, prob in zip(range(len(coords)), probs):
        #
        #     coord = coords[idx]
        #
        #     if prob == 0.0:
        #         continue
        #
        #     raw_block = heatmap[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0]]
        #
        #     img_block = img[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0]].copy()
        #
        #     color_block = (newcmp(raw_block) * 255)[:, :, :3].astype(np.uint8)
        #
        #     alpha = 0.3
        #     img_block = cv2.addWeighted(color_block, alpha, img_block, 1 - alpha, 0, img_block)
        #
        #     img[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0]] = img_block.copy()
        #
        # #not sure if this is necessary
        # del heatmap
        # gc.collect()
        #
        # img = Image.fromarray(img)
        # w, h = img.size
        # img.save(os.path.join(out_dir, f"{sample}_heatmap_clusters{run_number}.jpeg"), "JPEG")
        # print('Heatmap saved')


if __name__ == '__main__':

    #in my format wsi_list is a csv with a colum 'slide_id' with all wsi names
    # wsi_list = pd.read_csv(args.wsi_list)['slide_id'].tolist()
    wsi_list = os.listdir(args.wsi_list)

    compute_heatmap(args.scores_path, wsi_list, slide_dir=args.slide_dir, out_dir=args.output_dir)

