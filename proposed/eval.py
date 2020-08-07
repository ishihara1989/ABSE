from argparse import ArgumentParser
import yaml
from pathlib import Path

import numpy as np
import numba

def calc_mcd(d, l):
    return 10.0/np.log(10)*np.sqrt(2*d/l)

def mcd(x, y):
    D = np.sum((x[:, :, None] - y[:, None, :])**2, axis=0)
    return dp(D)
    

@numba.jit(nopython=True)
def dp(D):
    I = D.shape[0]
    J = D.shape[1]
    L = np.zeros((I, J))
    R = np.zeros((I, J))
    for i in range(I):
        for j in range(J):
            if i==0:
                if j==0:
                    R[i, j] = D[i, j]
                    L[i, j] = 1
                else:
                    R[i, j] = R[i, j-1] + D[i, j]
                    L[i, j] = L[i, j-1] + 1
            else:
                if j==0:
                    R[i, j] = R[i-1, j] + D[i, j]
                    L[i, j] = L[i-1, j] + 1
                else:
                    rs = np.array([R[i-1, j-1], R[i, j-1], R[i-1, j]])
                    ls = np.array([L[i-1, j-1], L[i, j-1], L[i-1, j]])
                    mn = rs[0]
                    idx = 0
                    for n in range(1, 3):
                        if rs[n] < mn:
                            mn = rs[n]
                            idx = n  
                    R[i, j] = rs[idx] + D[i, j]
                    L[i, j] = ls[idx] + 1
    return R[-1, -1],  L[-1, -1]

def eval_mcd(root, result_dir, ut_min=91, ut_max=100, sp_min=91, sp_max=100):
    root = Path(root)
    result_dir = Path(result_dir)
    
    for s in range(sp_min, sp_max+1):
        sp = f'jvs{s:03}'
        sp_root = result_dir / sp  # tgt
        
        for s2 in range(sp_min, sp_max+1):
            sp2 = f'jvs{s2:03}'
            if sp == sp2:
                continue
            conv_root = sp_root / sp2  # sp2->sp
            
            for u in range(ut_min, ut_max+1):
                tgt_mcep = root / sp / 'parallel100/wav24kHz16bit' / f'VOICEACTRESS100_{u:03}.mcep.npy'
                conv_mcep = conv_root / f'VOICEACTRESS100_{u:03}.mcep.npy'
                assert conv_mcep.is_file()
                assert tgt_mcep.is_file()
                x = np.load(conv_mcep)
                y = np.load(tgt_mcep)
                d, l = mcd(x, y)
                print(sp2, sp, f'VOICEACTRESS100_{u:03}', d, l, calc_mcd(d, l), flush=True)

def main():
    parser = ArgumentParser()
    parser.add_argument('-config', '-c', default='config.eval.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print(yaml.dump(config, default_flow_style=False))
    eval_mcd(**config)

if __name__ == "__main__":
    main()
