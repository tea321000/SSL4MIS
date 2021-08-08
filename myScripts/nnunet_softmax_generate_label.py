import numpy as np
import pickle as pk
import os
import argparse
import json
from collections import OrderedDict
import SimpleITK as sitk


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--preprocessed', help='nnunet preprocessed data dir', required=True, type=source_path)
    parser.add_argument('-the', '--threshold', default = 0.5, help='mask threshold', required=True, type=float)
    # parser.add_argument('-o', '--output', help='output 2d slice directory', required=True, type=output_path)
    # parser.add_argument('-m', '--mode', help='2d or 3d mode', required=True, type=str)
    return parser.parse_args()


def output_path(path):
    if os.path.isdir(path):
        return path
    else:
        os.mkdir(path)
        if os.path.isdir(path):
            return path
        else:
            raise argparse.ArgumentTypeError(f"nnunet preprocessed data dir:{path} is not a valid path")


def source_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"output 2d slice directory:{path} is not a valid path")


if __name__ == '__main__':
    parse_args = parse_arguments()
    for file in os.listdir(parse_args.preprocessed):
        if file.endswith(".npz"):
            name, ext = os.path.splitext(file)
            nii_arr = np.load(os.path.join(parse_args.preprocessed, file))
            print(file+":")
            # print(np.amax(nii_arr['softmax']), np.percentile(nii_arr['softmax'],66.5),np.amin(nii_arr['softmax']))
            # print(nii_arr['softmax'].shape)
            # print('-------------')
            print(np.argmax(nii_arr['softmax'],axis=0).shape)
            max_idx = np.argmax(nii_arr['softmax'],axis=0)
            print("max_idx", max_idx.shape)
            foreground_mask = np.where(np.amax(nii_arr['softmax'], axis=0)>=parse_args.threshold, max_idx, 0)
            print("foreground_mask", foreground_mask.shape)
            new_label_arr = foreground_mask*max_idx
            
            new_label = sitk.GetImageFromArray(new_label_arr)

            org_label = sitk.ReadImage(os.path.join(parse_args.preprocessed,name+'.nii.gz'))
            new_label.CopyInformation(org_label)
            sitk.WriteImage(new_label, os.path.join(parse_args.preprocessed,name+'_new.nii.gz'))



            # if parse_args.mode == '2d':
            #     for
            # with open(os.path.join(parse_args.preprocessed, name + '.pkl'), "r") as input_pickle, open(os.path.join(parse_args.output, name + '.json'), "w") as output_json:
            #     pk_dict = pk.load(input_pickle)
            #     output_json.write(json.dumps(pk_dict))
