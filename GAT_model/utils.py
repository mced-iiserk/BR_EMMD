# Setup data, Dataloader, Model loader, Model saver

import os
import json
import netCDF4 as nc
from pathlib import Path
import numpy as np
from glob import glob
import torch
#from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Dataset
from torch_geometric.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from torch_geometric.data import Data

class build_dataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray = None, w: np.ndarray = None) -> None:
        try:
            self.X = torch.from_numpy(X.astype(np.float32))
        except AttributeError:
            self.X = X
        if y is not None:
            self.y = torch.from_numpy(y.astype(np.float32))
            self.target = True
        else:
            self.target = False
        self.weight = False
        if w is not None:
            self.w = torch.from_numpy(w.astype(np.float32))
            self.weight = True
        self.len = len(self.X)

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, index: int) -> tuple:
        if self.weight:
            return self.X[index], self.y[index], self.w[index]
        elif self.target:
            return self.X[index], self.y[index]
        else:
            return self.X[index]

def minmaxscale(X):
    Xmin = X.min(axis=0).min(axis=0)
    Xmax = X.max(axis=0).max(axis=0)
    X = (X-Xmin)/(Xmax-Xmin)
    return X

def normalize(X):
    Xmean = X.mean()
    Xstdv = X.std()
    X = (X-Xmean)/Xstdv
    return X

def load_graphs(data_path: str, train_size: float, shuffle: bool = True, shuffle_index = None, weight: bool = False):
    data = nc.Dataset(data_path, 'r')
    node_features = np.array(data.variables['features'][:,:,:])
    #node_features = minmaxscale(node_features[:,:,:])#[:,:,np.newaxis])
    #node_features2 = node_features[:,:,9:]
    #node_features = np.concatenate((node_features1,node_features2),axis=2)
    #node_features_to_scale = node_features[:,:,:9]
    #node_features_min = node_features_to_scale.min(axis=0).min(axis=0)
    #node_features_max = node_features_to_scale.max(axis=0).max(axis=0)
    #node_features_to_scale = (node_features_to_scale-node_features_min)/(node_features_max-node_features_min)
    #node_features[:,:,:9] = node_features_to_scale
    inputs = []
    targets = np.array(data.variables['targets'][:,:,:])     # for mesh
    #targets = normalize(np.array(data.variables['targets'][:,:,:]))
    if weight:
        weights = np.array(data.variables['weights'][:,:,:])
    #targets_min = targets.min(axis=0)
    #targets_max = targets.max(axis=0)
    #targets = (targets-targets_min)/(targets_max-targets_min)
    instances = node_features.shape[0]
    training_instances = round(train_size*instances)
    if shuffle:
        if shuffle_index is not None:
            ids = np.load(shuffle_index)
        else:
            ids = np.arange(0,instances,1)
            np.random.shuffle(ids)
    train_ids = ids[:training_instances]
    valid_ids = ids[training_instances:]
    for i in range(instances):
        features = torch.tensor(node_features[i,:,:])
        edge_idx = torch.tensor(np.array(data.variables['edge_idx'][i,:,:]).T)
        inputs.append(Data(x=features, edge_index=edge_idx))
    if weight:
        train_data = build_dataset([inputs[ids] for ids in train_ids], targets[train_ids], weights[train_ids])
        valid_data = build_dataset([inputs[ids] for ids in valid_ids], targets[valid_ids], weights[valid_ids])
    else:
        train_data = build_dataset([inputs[ids] for ids in train_ids], targets[train_ids])
        valid_data = build_dataset([inputs[ids] for ids in valid_ids], targets[valid_ids])
    return train_data, valid_data

def load_graphs_pred(data_path: str):
    data = nc.Dataset(data_path, 'r')
    node_features = np.array(data.variables['features'][:,:,:])
    inputs = []
    instances = node_features.shape[0]
    for i in range(instances):
        features = torch.tensor(node_features[i,:,:])
        edge_idx = torch.tensor(np.array(data.variables['edge_idx'][i,:,:]).T)
        inputs.append(Data(x=features, edge_index=edge_idx))
    
    data = build_dataset(inputs)
    return data

def load_dataset(data_path: str, train_size: float, shuffle: bool = True, shuffle_index = None):
    #inputs = np.load(data_path)['inputs']
    #targets = np.load(data_path)['targets']
    data = nc.Dataset(data_path, 'r')
    inputs = np.array(data.variables['features'][:,:,:,:])
    targets = np.array(data.variables['targets'][:,:,:])
    #try:
    #    weights = np.load(data_path)['weights']
    #except KeyError:
    #    weights = None
    weights = None
    instances = inputs.shape[0]
    training_instances = round(train_size*instances)
    if training_instances//2:
        training_instances -= 1
    if shuffle:
        if shuffle_index is not None:
            ids = np.load(shuffle_index)
        else:
            ids = np.arange(0,instances,1)
            np.random.shuffle(ids)
    train_ids = ids[:training_instances]
    valid_ids = ids[training_instances:]
    if weights is not None:
        train_data = build_dataset(inputs[train_ids], targets[train_ids], weights[train_ids])
        valid_data = build_dataset(inputs[valid_ids], targets[valid_ids], weights[valid_ids])
    else:
        train_data = build_dataset(inputs[train_ids], targets[train_ids])
        valid_data = build_dataset(inputs[valid_ids], targets[valid_ids])
    return train_data, valid_data

def build_dataloader(TrainData=None, TestData=None, batch=16):
    """
    Builds training and testing dataloader.
    """
    if TrainData is not None:
        train_dataloader = DataLoader(
                TrainData,
                batch_size=batch,
                shuffle=True,
                pin_memory=True,
                #num_workers = 16,
                )
    test_dataloader = DataLoader(
            TestData,
            batch_size=batch,
            shuffle=False,
            pin_memory=True,
            #num_workers = 16,
            )
    if TrainData is not None:
        return train_dataloader, test_dataloader
    else:
        return test_dataloader

def save_model(
        model: torch.nn.Module,
        target_dir: str,
        model_name: str
        ):

    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith('.pth') or model_name.endswith('pt'), "model_name should end with '.pth' or '.pt'"
    model_save_path = target_dir_path / model_name

    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)
