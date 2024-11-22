import os
import json
import torch
import wandb
import argparse
import numpy as np
import utils, modeloss, engine

import gc
torch.cuda.empty_cache()
gc.collect()

parser = argparse.ArgumentParser(description="Learning membrane Z as a function of protein location on membrane surface.")
parser.add_argument('-i', '--infile', type=str, help='Input filename for the training.')
parser.add_argument('--wandb', type=str, default='False', help="Whether to log/exp in wandb. Default is False")
parser.add_argument('--hyprm', type=str, help="JSON dictionary to read hyperparameters.")
parser.add_argument('-o', '--outfile', type=str, help='Output filename for the inference.')

args = parser.parse_args()
WANDB = args.wandb
HYPRM = args.hyprm
INFILE = args.infile
OUTFILE = args.outfile

run = True

def iterDict(d,i=0,j=-1):
    if j!=-1:
        keys = list(d.keys())[i:j]
    else:
        keys = list(d.keys())[i:]
    newd = {key: d[key] for key in keys}
    return newd

with open(HYPRM,'r') as f:
    HYPERPARAMS = json.load(f)
    PROJECT = HYPERPARAMS["PROJECT"]
    DEVICE = HYPERPARAMS["DEVICE"]
    ARCHITECTURE = HYPERPARAMS["ARCHITECTURE"]
    EPOCHS = HYPERPARAMS["EPOCHS"]
    TRAIN_SIZE = HYPERPARAMS["TRAIN_SIZE"]
    BATCH_SIZE = HYPERPARAMS["BATCH_SIZE"]
    OUTPUT_CHANNELS = HYPERPARAMS["OUTPUT_CHANNELS"]
    LEARNING_RATE = HYPERPARAMS["LEARNING_RATE"]
    LOSS = HYPERPARAMS["LOSS"]
    MODELS_DIR = HYPERPARAMS["MODELS_DIR"]
    MODEL_NAME = HYPERPARAMS["MODEL_NAME"]


if run:
    device = torch.device(DEVICE)

    #model = modeloss.CustomResNetRN11().to(device)
    model = modeloss.GraphAtt2Mesh(node_feature_dim=9+625, conv_output_size=(25, 25)).to(device)
    #model = modeloss.Graph2Mesh(node_feature_dim=9+625, conv_output_size=(25, 25)).to(device)
    #model_path = "models/GraphAtt2Mesh/Membrane2_minZ_GAT_fullMSE_sfl1_2300_r2_0.81.pth"
    #model_path = "models/GraphConv2/Membrane2_minZ_GCN_fullMSE_touch_80_1000.pth"
    #model_path = "models/GraphAtt2Mesh/Membrane2_minZ_GAT_fullMSE_touch_1054.pth"
    #model_path = "models/GraphAtt2Mesh/Membrane2_minZ_GAT_fullMSE_touch_1054_r2_0.81.pth"
    model_path = "models/GraphAtt2Mesh/Membrane2_minZ_GAT_fullMSE_touch_0.8_set1_999.pth"
    #model_path = "models/GraphAtt2Mesh/Membrane2_minZ_GAT_fullMSE_touch_0.8_999.pth"
    #model_path = "models/GraphConv2/Membrane2_minZ_GCN_fullMSE_touch_1205.pth"
    
    #model_path = "models/GraphAtt2Mesh/Membrane1_minZ_GAT_fullLoss_3000.pth"
    model.load_state_dict(torch.load(model_path))

    data = utils.load_graphs_pred(data_path=INFILE)

    test_dloader = utils.build_dataloader(TestData=data, batch=BATCH_SIZE)

    y_pred = np.empty((0,25,25))

    for X in test_dloader:
        X = X.to(device)
        y_infr = engine.infer(
                model = model,
                dataX = X,
                device = device)
        y_pred = np.vstack((y_pred,y_infr))

    print(y_pred.shape)
    np.save(OUTFILE, y_pred)
    print(f"[INFO] the inferred data is saved to {OUTFILE}\n")

