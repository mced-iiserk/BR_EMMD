import os
import json
import torch
import wandb
import argparse
import utils, modeloss, engine

import gc
torch.cuda.empty_cache()
gc.collect()

parser = argparse.ArgumentParser(description="Learning membrane Z as a function of protein location on membrane surface.")
parser.add_argument('-i', '--infile', type=str, help='Input filename for the training.')
parser.add_argument('--idx', type=str, default=None, help='Input indices for shuffling')
parser.add_argument('--wandb', type=str, default='False', help="Whether to log/exp in wandb. Default is False")
parser.add_argument('--hyprm', type=str, help="JSON dictionary to read hyperparameters.")
parser.add_argument('--weights', type=bool, default=False, help="when True, it uses weighted loss function")

args = parser.parse_args()
WANDB = args.wandb
HYPRM = args.hyprm
INFILE = args.infile
WEIGHT = args.weights
INDEX = args.idx

run = False

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
    #HIDDEN_LAYERS = HYPERPARAMS["HIDDEN_LAYERS"]
    #HIDDEN_UNITS = HYPERPARAMS["HIDDEN_UNITS"]
    LEARNING_RATE = HYPERPARAMS["LEARNING_RATE"]
    LOSS = HYPERPARAMS["LOSS"]
    MODELS_DIR = HYPERPARAMS["MODELS_DIR"]
    MODEL_NAME = HYPERPARAMS["MODEL_NAME"]
    SAVEFREQ = HYPERPARAMS["SAVING_FREQ"]
    EPOCH_INI = HYPERPARAMS["EPOCH_INI"]
    #WEIGHT = HYPERPARAMS["WEIGHT"]
    #ALPHA = HYPERPARAMS["ALPHA"]
    #GAMMA = HYPERPARAMS["GAMMA"]


if WANDB.lower() == 'log':
    wandb.init(
        # set the wandb project where this run will be logged
        project=HYPERPARAMS["PROJECT"],

        # track hyperparameters and run metadata
        config=iterDict(HYPERPARAMS,i=1),
    )
    run = True

elif WANDB.lower() == 'exp':
    print('starting experiment:\n')
    sweep_config = {
        'method': 'random'
        }

    metric = {
        'name': 'loss',
        'goal': 'minimize'   
        }

    sweep_config['metric'] = metric

    parameters_dict = iterDict(HYPERPARAMS,i=1)

    sweep_config['parameters'] = parameters_dict

    sweep_id = wandb.sweep(sweep_config, project=HYPERPARAMS["PROJECT"])

    wandb.agent(sweep_id, engine.train_wandb, count=10)

else:
    run = True
    wandb = None

if run:
    device = torch.device(DEVICE)

    #model = modeloss.GraphAtt2Mesh(node_feature_dim=9+625, conv_output_size=(25, 25)).to(device)
    model = modeloss.Graph2Mesh(node_feature_dim=9+625, conv_output_size=(25, 25)).to(device)
    #model_path = "models/GraphAtt2Mesh/Membrane2_minZ_GAT_fullMSE_sfl1_2300.pth"
    #model_path = "models/GraphAtt2Mesh/Membrane2_minZ_GAT_fullMSE_touch_0.8_311.pth"
    #model.load_state_dict(torch.load(model_path))

    print('\nmodel built')

    if INDEX is not None:
        train_data, test_data = utils.load_graphs(data_path=INFILE, train_size=TRAIN_SIZE, shuffle_index=INDEX, weight=WEIGHT)
    else:
        train_data, test_data = utils.load_graphs(data_path=INFILE, train_size=TRAIN_SIZE, weight=WEIGHT)

    print('\ndata prepared\n')

    #print(train_data)

    train_dloader, test_dloader = utils.build_dataloader(TrainData=train_data, TestData=test_data, batch=BATCH_SIZE)

    LOSS_fn = {
            'MSE': modeloss.MSELoss(),
            'FMSE': modeloss.FocalMSELoss(),
            }

    loss_fn = LOSS_fn[LOSS]
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)#, weight_decay=1e-4)

    results = engine.train(
            model = model,
            train_dataloader = train_dloader,
            test_dataloader = test_dloader,
            loss_fn = loss_fn,
            optimizer = optimizer,
            epochs = EPOCHS,
            device = device,
            wandb = wandb,
            target_dir = MODELS_DIR,
            model_name = MODEL_NAME,
            weight = WEIGHT,
            saving_freq = SAVEFREQ,
            epoch_ini = EPOCH_INI,
            )

    #utils.save_model(
    #        model = model,
    #        target_dir = MODELS_DIR,
    #        model_name = MODEL_NAME
    #        )
