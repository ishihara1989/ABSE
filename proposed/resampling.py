import sys
import pathlib

import librosa
import numpy as np
import scipy.io.wavfile as wavfile

def resample_dir(srcroot, tgtroot, sr=16000):
    src = pathlib.Path(srcroot)
    tgt = pathlib.Path(tgtroot)
    if not pathlib.Path(src).exists():
        raise ValueError('src not exists: {}'.format(src))

    for p in src.glob('**/*.wav'):
        print(p)
        x, _ = librosa.load(p, sr=sr)
        wav = (32768*x).astype(np.int16)

        tgt_dir = tgt / p.parent.relative_to(src)
        tgt_path = (tgt_dir / p.name).with_suffix('.wav')
        tgt_dir.mkdir(parents=True, exist_ok=True)

        wavfile.write(tgt_path, sr, wav)
        print(tgt_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f'usage: {sys.argv[0]} srcdir tgtdir')
        exit(1)

    if len(sys.argv) >= 4:
        try:
            sr = int(sys.argv[3])
        except:
            print(f'failed to parse as int: {sys.argv[3]}')
            exit(1)
    else:
        sr = 16000
    resample_dir(sys.argv[1], sys.argv[2], sr=sr)