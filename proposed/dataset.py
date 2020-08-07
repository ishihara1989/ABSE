from pathlib import Path
import random

import numpy as np
import torch
import torch.utils.data


def infinit_iterator(dataloader):
    while True:
        for batch in dataloader:
            yield batch

def two_parallel_collate(batch):
    batch = [tp for tp in batch if tp is not None]
    if len(batch) == 0: return None
    maxs = [0, 0, 0, 0]
    for tp in batch:
        for i in range(4):
            maxs[i] = max(maxs[i], tp[i].shape[-1])
    shapes = [[len(batch), *batch[0][i].shape] for i in range(4)]
    for i in range(4):
        shapes[i][-1] = maxs[i]
    ret = [np.zeros(s, dtype=np.float32) for s in shapes]
    for b, tp in enumerate(batch):
        for i in range(4):
            ret[i][b, ..., :tp[i].shape[-1]] = tp[i][...]
    return [torch.as_tensor(r) for r in ret]

def two_nonparallel_collate(batch):
    batch = [tp for tp in batch if tp is not None]
    if len(batch) == 0: return None
    x1 = [torch.as_tensor(b[0]).unsqueeze(0) for b in batch]
    x2 = [torch.as_tensor(b[1]).unsqueeze(0) for b in batch]
    with torch.no_grad():
        return torch.cat(x1, dim=0), torch.cat(x2, dim=0)

class JvsTwoNonparallelMcep(torch.utils.data.Dataset):
    def __init__(self, root, t_size, sp_min=1, sp_max=90, ut_min=1, ut_max=90):
        self.root = Path(root)
        self.sp_min = sp_min
        self.sp_max = sp_max
        self.ut_min = ut_min
        self.ut_max = ut_max
        self.t_size = t_size
        
    @property
    def n_speakers(self):
        return 1+self.sp_max-self.sp_min
    
    @property
    def n_utterances(self):
        return 1+self.ut_max-self.ut_min
    
    def unfold_index(self, k, N):
        # index to combination
        i = k // N
        j = k % N
        if i >= j:
            return i+1, j
        else:
            return N-i-1, N-j-1
        
    def __len__(self):
        return self.n_speakers * self.n_utterances * (self.n_utterances-1) // 2
    
    def indeces(self, index):
        J = self.n_utterances * (self.n_utterances-1) // 2
        si = index // J
        uk = index % J
        ui, uj = self.unfold_index(uk, self.n_utterances)
        return si, ui, uj
    
    def map_path(self, sp, ut):
        i = sp + self.sp_min
        j = ut + self.ut_min
        return self.root / f'jvs{i:03}' / 'parallel100/wav24kHz16bit' / f'VOICEACTRESS100_{j:03}.mcep.npy'
    
    def __getitem__(self, index):
        si, ui, uj = self.indeces(index)
        x1 = self.map_path(si, ui)
        x2 = self.map_path(si, uj)
        paths = [x1, x2]
        if not all([p.exists() for p in paths]):
            return None

        return [self.load_mcep(p) for p in paths]
    
    def load_mcep(self, path):
        mcep = np.load(path)
        if mcep.shape[-1] >= self.t_size:
            max_start = mcep.shape[-1] - self.t_size
            start = random.randint(0, max_start)
            mcep = mcep[:, start:start+self.t_size]
        else:
            offset = random.randint(0, self.t_size - mcep.shape[-1])
            mcep = np.pad(mcep, [(0, 0), (offset, self.t_size - mcep.shape[-1]-offset)], 'constant')
        return mcep

class JvsTwoParallelMcep(torch.utils.data.Dataset):
    def __init__(self, root, sp_min=1, sp_max=90, ut_min=1, ut_max=90, verbose=False):
        self.root = Path(root)
        self.sp_min = sp_min
        self.sp_max = sp_max
        self.ut_min = ut_min
        self.ut_max = ut_max
        self.verbose = verbose

    @property
    def n_speakers(self):
        return 1 - self.sp_min + self.sp_max

    @property
    def n_utterances(self):
        return 1 - self.ut_min + self.ut_max

    def __len__(self):
        return self.n_speakers * (self.n_speakers-1) * self.n_utterances * (self.n_utterances-1) // 4

    def indeces(self, index):
        J = self.n_utterances * (self.n_utterances-1) // 2
        sk = index // J
        uk = index % J
        si, sj = self.unfold_index(sk, self.n_speakers)
        ui, uj = self.unfold_index(uk, self.n_utterances)
        return si, sj, ui, uj

    def unfold_index(self, k, N):
        # index to combination
        i = k // N
        j = k % N
        if i >= j:
            return i+1, j
        else:
            return N-i-1, N-j-1

    def map_path(self, sp, ut):
        i = sp + self.sp_min
        j = ut + self.ut_min
        return self.root / f'jvs{i:03}' / 'parallel100/wav24kHz16bit' / f'VOICEACTRESS100_{j:03}.mcep.npy'

    def __getitem__(self, index):
        si, sj, ui, uj = self.indeces(index)
        if self.verbose:
            print(self.sp_min+si, self.sp_min+sj, self.ut_min+ui, self.ut_min+uj)
        x1 = self.map_path(si, ui)
        x2 = self.map_path(si, uj)
        y1 = self.map_path(sj, ui)
        y2 = self.map_path(sj, uj)
        paths = [x1, x2, y1, y2]
        if not all([p.exists() for p in paths]):
            return None

        return [np.load(p) for p in paths]
