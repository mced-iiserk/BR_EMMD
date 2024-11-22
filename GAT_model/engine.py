from typing import Dict, List, Tuple
import numpy as np
import torch
import glob
from tqdm.auto import tqdm
import modeloss
import wandb
import utils
from torcheval.metrics import MeanSquaredError, R2Score
from torch.optim.lr_scheduler import ReduceLROnPlateau
#from torch.optim.lr_scheduler import CosineAnnealingLR

#scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

def train_step(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        weight: bool = False
        ) -> float:
    # turn on train mode
    model.train()
    mse_metric = MeanSquaredError(device=device)
    r2s_metric = R2Score(device=device)

    # setup train loss and acc values
    train_loss = 0

    if weight:
        for batch, (X,y,w) in enumerate(dataloader):
            X,y,w = X.to(device), y.to(device), w.to(device)
            y_pred = model(X)

            loss = loss_fn(y_pred,y,w)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mask = w>0

            #mse_metric.update(torch.flatten(y_pred[mask==1]),torch.flatten(y[mask==1]))
            #r2s_metric.update(torch.flatten(y_pred[mask==1]),torch.flatten(y[mask==1]))
            mse_metric.update(torch.flatten(y_pred[mask]),torch.flatten(y[mask]))
            r2s_metric.update(torch.flatten(y_pred[mask]),torch.flatten(y[mask]))
    
    else:
        for batch, (X,y) in enumerate(dataloader):
            X,y = X.to(device), y.to(device)
            y_pred = model(X)

            loss = loss_fn(y_pred,y)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mse_metric.update(torch.flatten(y_pred),torch.flatten(y))
            r2s_metric.update(torch.flatten(y_pred),torch.flatten(y))

    train_loss = train_loss / len(dataloader)

    return train_loss, mse_metric.compute(), r2s_metric.compute()

def eval_step(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
	device: torch.device,
        weight: bool = False
	) -> float:
    # turn on train mode
    model.eval()
    mse_metric = MeanSquaredError(device=device)
    r2s_metric = R2Score(device=device)

    # setup train loss and acc values
    test_loss = 0

    with torch.inference_mode():
        if weight:
            for batch, (X,y,w) in enumerate(dataloader):#, total=len(dataloader)):
                X,y,w = X.to(device), y.to(device), w.to(device)
                y_pred = model(X)

                loss = loss_fn(y_pred,y,w)
                test_loss += loss.item()

                mask = w>0

                mse_metric.update(torch.flatten(y_pred[mask]),torch.flatten(y[mask]))
                r2s_metric.update(torch.flatten(y_pred[mask]),torch.flatten(y[mask]))
        else:
            for batch, (X,y) in enumerate(dataloader):#, total=len(dataloader)):
                X,y = X.to(device), y.to(device)
                y_pred = model(X)

                loss = loss_fn(y_pred,y)
                test_loss += loss.item()

                mse_metric.update(torch.flatten(y_pred),torch.flatten(y))
                r2s_metric.update(torch.flatten(y_pred),torch.flatten(y))

    test_loss = test_loss / len(dataloader)

    return test_loss, mse_metric.compute(), r2s_metric.compute()



def infer(
        model: torch.nn.Module,
        dataX: np.ndarray,
        device: torch.device
        ):
    model.eval()
    #dataT = torch.from_numpy(dataX.astype(np.float32)).to(device)
    with torch.inference_mode():
        y_pred = model(dataX)

    y_pred = y_pred.cpu().numpy()

    return y_pred



def train(
        model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        epochs: int,
        target_dir: str,
        model_name: str,
        device: torch.device,
        weight: bool = False,
        wandb = None,
        saving_freq: int = 1,
        epoch_ini: int = 0,
        ) -> Dict[str, List[float]]:

    #scheduler = CosineAnnealingLR(optimizer, T_max=10)

    #scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=5, verbose=True)

    results = {'train_loss': [], 'test_loss':[], 'train_mse': [], 'test_mse': [], 'train_r2': [], 'test_r2': []}

    for epoch in tqdm(range(epochs)):
        train_loss,train_mse,train_r2 = train_step(
                model = model,
                dataloader = train_dataloader,
                loss_fn = loss_fn,
                weight = weight,
                optimizer = optimizer,
                device = device
                )
        test_loss,test_mse,test_r2 = eval_step(
                model = model,
                dataloader = test_dataloader,
                loss_fn = loss_fn,
                weight = weight,
                device = device
                )
        #scheduler.step(test_loss)
        #scheduler.step()
        if (epoch+epoch_ini)%saving_freq == 0:
            MODEL_NAME = f"{model_name}_{epoch+epoch_ini}.pth"
            utils.save_model(
                model = model,
                target_dir = target_dir,
                model_name = MODEL_NAME,
                )
        if type(wandb) != type(None):
            wandb.log({"train_loss":train_loss, "test_loss":test_loss, "train_mse":train_mse, "test_mse":test_mse, "train_r2":train_r2, "test_r2":test_r2})
        
        print(f"Epoch: {epoch+1+epoch_ini} | train - mse: {train_mse:.4f}, r2: {train_r2:.2f} | test - mse: {test_mse:.4f}, r2: {test_r2:.2f}")

        results['train_loss'].append(train_loss)
        results['test_loss'].append(test_loss)
        results['train_mse'].append(train_mse)
        results['test_mse'].append(test_mse)
        results['train_r2'].append(train_r2)
        results['test_r2'].append(test_r2)
        
        """
        if epoch>=10 and round(float(torch.mean(torch.tensor(results['test_r2'][-10:])).cpu()),2)==0.85:
            MODEL_NAME = f"{model_name}_{epoch+epoch_ini}_r2_0.81.pth"
            utils.save_model(
                model = model,
                target_dir = target_dir,
                model_name = MODEL_NAME,
                )
            break
        """

    MODEL_NAME = f"{model_name}_{epoch+epoch_ini}.pth"
    utils.save_model(
            model = model,
            target_dir = target_dir,
            model_name = MODEL_NAME,
            )
    return results

def train_wandb(config=None) -> Dict[str, List[float]]:
    with wandb.init(config=config):
        config = wandb.config

        INFILE = config.INFILE
        device = torch.device(config.DEVICE)
        train_data, test_data = utils.load_dataset(src_files=INFILE, train_size=config.TRAIN_SIZE, weight=config.WEIGHT)
        train_dloader, test_dloader = utils.build_dataloader(TrainData=train_data, TestData=test_data, batch=config.BATCH_SIZE)

        print(config)

        LOSS_fn = {
            'MSE': modeloss.MSELoss(),
            'FMSE': modeloss.FocalMSELoss(),
            }

        model = modeloss.pretrained_ResNet50(output_channels=config.OUTPUT_CHANNELS).to(device)
        
        train_dataloader = train_dloader
        test_dataloader = test_dloader
        loss_fn = LOSS_fn[config.LOSS]
        lr = config.LEARNING_RATE
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        #device = device
        
        results = {'train_loss': [], 'test_loss':[], 'train_mse': [], 'test_mse': [], 'train_r2': [], 'test_r2': []}

        for epoch in tqdm(range(config.EPOCHS)):
            train_loss,train_mse,train_r2 = train_step(
                    model = model,
                    dataloader = train_dataloader,
                    loss_fn = loss_fn,
                    optimizer = optimizer,
                    device = device,
                    weight = config.WEIGHT
                    )
            test_loss,test_mse,test_r2 = eval_step(
                    model = model,
                    dataloader = test_dataloader,
                    loss_fn = loss_fn,
                    device = device,
                    weight = config.WEIGHT
                    )
            if wandb:
                wandb.log({"train_loss":train_loss, "test_loss":test_loss, "train_mse":train_mse, "test_mse":test_mse, "train_r2":train_r2, "test_r2":test_r2})

            print(f"Epoch: {epoch+1} | train - mse: {train_mse:.4f}, r2: {train_r2:.2f} | test - mse: {test_mse:.4f}, r2: {test_r2:.2f}")

            results['train_loss'].append(train_loss)
            results['test_loss'].append(test_loss)
            results['train_mse'].append(train_mse)
            results['test_mse'].append(test_mse)
            results['train_r2'].append(train_r2)
            results['test_r2'].append(test_r2)

    return results
