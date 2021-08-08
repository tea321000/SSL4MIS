import numpy as np
import pickle as pk
import os
import argparse
import math


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-pk', '--pickle', help='path to splits_final.pkl', required=True, type=source_path)
    parser.add_argument('-per', '--percent', default=0.5, help='percentage of semi-supervised samples', required=True, type=float)
    return parser.parse_args()

def source_path(path):
    if os.path.isfile(path):
        return path
    else:
        raise argparse.ArgumentTypeError("splits_final.pkl directory:{} is not a valid path".format(path))

if __name__ == '__main__':
    parse_args = parse_arguments()
    with open((parse_args.pickle), "rb") as input_pickle:
        pk_dict = pk.load(input_pickle)
        for fold in range(len(pk_dict)):
            # print(len(pk_dict[fold]['train']))
            pk_dict[fold]['unsupervised'] = pk_dict[fold]['train'][math.floor(len(pk_dict[fold]['train'])*parse_args.percent):]
            pk_dict[fold]['train'] = pk_dict[fold]['train'][:math.floor(len(pk_dict[fold]['train'])*parse_args.percent)]
        path = os.path.split(parse_args.pickle)[0]
        with open(os.path.join(path, 'splits_final_'+str(parse_args.percent)+'.pkl'), "wb") as output_pickle:
            pk.dump(pk_dict, output_pickle)
