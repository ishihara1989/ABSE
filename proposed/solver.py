from pathlib import Path

import torch

from dataset import *
from model import VC

def n_params(model):
    n = 0
    for p in model.parameters():
        if p.requires_grad:
            n += p.numel()
    return n

class Solver(object):
    def __init__(self, config):
        self.config = config

    def train(self, iteration=None):
        config = self.config
        nonpara_dataset = JvsTwoNonparallelMcep(**config['data']['nonparallel'])
        nonpara_dataloader = torch.utils.data.DataLoader(nonpara_dataset, **config['dataloader']['nonparallel'],
            pin_memory=False,
            drop_last=True,
            collate_fn=two_nonparallel_collate)
        nonpara_iterator = infinit_iterator(nonpara_dataloader)
        para_dataset = JvsTwoParallelMcep(**config['data']['parallel'])
        para_dataloader = torch.utils.data.DataLoader(para_dataset, **config['dataloader']['parallel'],
            pin_memory=False,
            drop_last=True,
            collate_fn=two_parallel_collate)
        para_iterator = infinit_iterator(para_dataloader)
        model = VC(config['model'])
        print(f'#params: {n_params(model):,}')
        model = model.cuda()
        optimizer = torch.optim.AdamW(model.parameters(), **config['optimizer'])

        Path(config['training']['checkpoints']).mkdir(exist_ok=True, parents=True)
        if iteration is None:
            iteration = 1
        for batch in para_iterator:
            nonpara_batch = next(nonpara_iterator)
            optimizer.zero_grad()
            if True:
                batch = [b.cuda() for b in batch]
                nonpara_batch = [b.cuda() for b in nonpara_batch]
            loss, others = model(*batch, *nonpara_batch)
            others = [f'{a.data:.5f}' for a in others]

            loss.backward()
            g = torch.nn.utils.clip_grad_norm_(model.parameters(), **config['grad_norm'])
            optimizer.step()
            print(f'{iteration}: {loss.data:.5f}, {others}, {g:.5f}', flush=True)
            
            if (iteration)%config['training']['save_every'] == 0:
                print('save {}'.format('latest'))
                torch.save({
                    'model': model.state_dict(),
                    'config': config
                }, f'{config["training"]["checkpoints"]}/checkpoint_latest.pt')
            if (iteration)%config['training']['checkpoint_every'] == 0:
                print('save {}'.format(iteration))
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config,
                    'iteration': iteration
                }, f'{config["training"]["checkpoints"]}/checkpoint_{iteration}.pt')
            if (iteration) >= config['training']['max_itr']:
                print('save {}'.format('final'))
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config,
                    'iteration': iteration
                }, f'{config["training"]["checkpoints"]}/checkpoint_final.pt')
                return
            
            iteration += 1