from argparse import ArgumentParser
import yaml
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile
import pyworld
import pysptk
import torch

from model import AE

def extract_from(model, ref_mcep, ref_f0, sp_dict_path):
    assert(ref_mcep.is_file())
    assert(ref_f0.is_file())
    mcep = np.load(ref_mcep)
    f0 = np.load(ref_f0)
    f0_valid = np.log(f0[f0>0])
    f0_mean = np.mean(f0_valid)
    f0_std = np.std(f0_valid)
    k, v = model.extract(mcep)
    sp_dict = {
        "f0_mean": f0_mean,
        "f0_std": f0_std,
        "k": k,
        "v": v
    }
    torch.save(sp_dict, sp_dict_path)
    return sp_dict

def prep_content(model, src_mcep, src_dict_path):
    assert(src_mcep.is_file())
    mcep = np.load(src_mcep)
    c, q = model.content(mcep)
    src_dict = {
        "c": c,
        "q": q
    }
    torch.save(src_dict, src_dict_path)
    return src_dict

def convert_f0(f0, sp_dict):
    f0_idx = f0 > 0
    f0_valid = np.log(f0[f0_idx])
    f0_mean = np.mean(f0_valid)
    f0_std = np.std(f0_valid)
    f0_converted = (f0_valid - f0_mean) / f0_std * sp_dict['f0_std'] + sp_dict['f0_mean']
    f0[f0_idx] = np.exp(f0_converted)
    return f0

def reconstruct_ap(ap192):
    ap0 = np.ones_like(ap192)
    ap0[ap192 < 1.0] = 1e-3 
    ap512 = np.ones_like(ap192)
    ap = np.zeros((ap192.shape[0], 513), dtype=np.float64)
    i = np.arange(513)
    ip = np.array([0,192,512])
    for t in range(ap.shape[0]):
        jp = np.array([np.log(ap0[t]),np.log(ap192[t]), 0.0])
        ap[t, :] = np.exp(np.interp(i, ip, jp))
    return ap
    

def make_conversion(root, result_dir, checkpoint, ut_min=91, ut_max=100, sp_min=91, sp_max=100):
    alpha = 0.42
    n_fft = 1024
    root = Path(root)
    result_dir = Path(result_dir)
    dicts = torch.load(checkpoint, map_location='cpu')
    model = AE(dicts['config']['model'], train=False)
    model.load_state_dict(dicts['model'])
    model = model.eval()
    for s in range(sp_min, sp_max+1):
        sp = f'jvs{s:03}'
        sp_root = result_dir / sp
        sp_root.mkdir(parents=True, exist_ok=True)
        sp_dict_path = sp_root / 'sp_dict.pt'
        
        if not sp_dict_path.is_file():
            nonparas = list((root / sp / 'nonpara30/wav24kHz16bit').glob('BASIC5000_*.mcep.npy'))
            index = max(enumerate(nonparas), key=lambda p: p[1].stat().st_size)[0]
            ref_mcep = nonparas[index]
            ref_f0 = ref_mcep.parent / ref_mcep.stem.replace('.mcep', '.f0.npy')
            sp_dict = extract_from(model, ref_mcep, ref_f0, sp_dict_path)
        else:
            sp_dict = torch.load(sp_dict_path)
        for s2 in range(sp_min, sp_max+1):
            sp2 = f'jvs{s2:03}'
            sp2_root = result_dir / sp2
            sp2_root.mkdir(parents=True, exist_ok=True)
            
            target_root = sp_root / sp2
            target_root.mkdir(parents=True, exist_ok=True)
            for u in range(ut_min, ut_max+1):
                src_mcep = root / sp2 / 'parallel100/wav24kHz16bit' / f'VOICEACTRESS100_{u:03}.mcep.npy'
                src_f0 = root / sp2 / 'parallel100/wav24kHz16bit' / f'VOICEACTRESS100_{u:03}.f0.npy'
                src_c0 = root / sp2 / 'parallel100/wav24kHz16bit' / f'VOICEACTRESS100_{u:03}.c0.npy'
                src_ap = root / sp2 / 'parallel100/wav24kHz16bit' / f'VOICEACTRESS100_{u:03}.ap.npy'
                src_dict_path = sp2_root / f'VOICEACTRESS100_{u:03}.pt'
                if not src_dict_path.is_file():
                    src_dict = prep_content(model, src_mcep, src_dict_path)
                else:
                    src_dict = torch.load(src_dict_path)
                    
                converted_mcep = model.reconstruct_mcep(src_dict['c'], src_dict['q'], sp_dict['k'], sp_dict['v']).squeeze().numpy()
                tgt_mcep = target_root / f'VOICEACTRESS100_{u:03}.mcep.npy'
                np.save(tgt_mcep, converted_mcep)
                
                f0 = np.load(src_f0).astype(np.float64)
                f0 = convert_f0(f0, sp_dict)
                ap = np.load(src_ap).astype(np.float64)
                ap = reconstruct_ap(ap)
                c0 = np.load(src_c0).astype(np.float64)
                assert (c0.shape[0] <= converted_mcep.shape[-1]), f'{s}->{s2}/{u}, {c0.shape[0]} <= {converted_mcep.shape[-1]}'
                mcep = np.hstack([c0[:, None], converted_mcep[:, :c0.shape[0]].T]).astype(np.float64)
                sp = pysptk.mc2sp(np.ascontiguousarray(mcep), alpha, n_fft)
                wav = pyworld.synthesize(f0, sp, ap, 16000)
                tgt_wav = target_root / f'VOICEACTRESS100_{u:03}.wav'
                wavfile.write(tgt_wav, 16000, (wav*32768).astype(np.int16))
                print(tgt_wav, flush=True)

def main():
    parser = ArgumentParser()
    parser.add_argument('-config', '-c', default='config.test.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print(yaml.dump(config, default_flow_style=False))
    make_conversion(**config)

if __name__ == "__main__":
    main()
