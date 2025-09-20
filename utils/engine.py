import torch
import torch.nn.functional as F

import numpy as np
import pdb

# from .metrics import get_metrics

def train_one_epoch(model, dataloader, loss_fn, optimizer, scheduler, task_cfg, device):
    model.train()
    total_loss = []
    
    for batch_idx, data in enumerate(dataloader, start=1):
        optimizer.zero_grad()
        
        if task_cfg['object'] == 'train_vae':
            x, label = data
            x = x.to(device)           
            label = label.to(device)
            
            x_prime, z, mu, log_var = model(x)
            loss, _, __ = loss_fn(x_prime, x, mu, log_var)

        else:
            raise Exception("Check your task_cfg['object'] configuration")
        
        total_loss.append(loss.item())
        
        loss.backward()
        optimizer.step()
        
        print(f"\rTraining: {100*batch_idx/len(dataloader):.2f}%, Loss: {sum(total_loss)/len(total_loss):.6f}, LR: {scheduler.get_last_lr()[0]:.6f}", end="")
    print()
    
    return sum(total_loss)/len(total_loss)

@torch.no_grad()
def validate(model, dataloader, loss_fn, scheduler, task_cfg, device):
    model.eval()
    total_loss = []
    
    for batch_idx, (x, target) in enumerate(dataloader, start=1):
        if task_cfg['object'] == 'train_vae':
            x, label = data
            x = x.to(device)           
            label = label.to(device)

        else:
            raise Exception("Check your task_cfg['object'] configuration")
        
        total_loss.append(loss.item())
        
        print(f"\rValidate: {100*batch_idx/len(dataloader):.2f}%, Loss: {sum(total_loss)/len(total_loss):.6f}", end="")
    print()
    
    scheduler.step(sum(total_loss)/len(total_loss))
    
    return sum(total_loss)/len(total_loss)

@torch.no_grad()
def evaluate(model, dataloader, loss_fn, task_cfg, device):
    model.eval()
    
    total_loss = []
    
    for batch_idx, (x, target) in enumerate(dataloader, start=1):
        if task_cfg['object'] == 'train_vae':
            x, label = data
            x = x.to(device)           
            label = label.to(device)

            '''
            outs = F.sigmoid(logits)
            outs = torch.where(outs >= task_cfg['threshold'], 1, 0)

            total_outputs.extend(outs.tolist())
            total_targets.extend(target.tolist())
            '''
            
            total_loss.append(loss.item())
        else:
            raise Exception("Check your task_cfg['object'] configuration")
        
        print(f"\rEvaluate: {100*batch_idx/len(dataloader):.2f}%, Loss: {sum(total_loss)/len(total_loss):.6f}", end="")
    print()
    
    # result = get_metrics(np.array(total_outputs), np.array(total_targets))
    result = {
        'loss': sum(total_loss)/len(total_loss)
    }
    
    return result