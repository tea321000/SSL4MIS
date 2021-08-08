import numpy as np
import pickle as pk
import os
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--preprocessed', help='nnunet preprocessed data dir', required=True, type=source_path)
    return parser.parse_args()

def source_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"output 2d slice directory:{path} is not a valid path")

if __name__ == '__main__':
    parse_args = parse_arguments()
    for file in os.listdir(parse_args.preprocessed):
        if file.endswith(".pkl"):
            with open(os.path.join(parse_args.preprocessed, file), "rb") as input_pickle:
                pk_dict = pk.load(input_pickle)
                print(file, ":")
                print(pk_dict)
                print("--------")