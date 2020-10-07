from __future__ import print_function
import os
import sys
import torch
import evaluation
from model import get_model

def parse_agrs():
    parser = argparse.ArgumentParser()
    parser.add_argument('testCollection', type=str, help='test collection')
    parser.add_argument('--logger_name',default='runs',help= 'path')
    parser.add_argument('--checkpoint_name', default='model_best.pth.tar', type=str, help='name of checkpoint (default: model_best.pth.tar)')
    args = parser.parse_args()
    return args

def main():
    opt = parse_args()
    testCollection = opt.testCollection
    print(json.dumps(vars(opt),indent))
    path = "cd /usr/local/extstore01/gengyi/VisualSearch/msrvtt10ktest/results/msrvtt10ktrain/model_best.pth.tar/pred_errors_matrix.pth.tar"
   model = torch.load 
