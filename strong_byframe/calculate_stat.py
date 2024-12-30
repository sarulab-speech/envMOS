from tqdm import tqdm
import numpy as np
import logging

from byol_a.common import load_yaml_config
from byol_a.dataset import WaveInLMSOutDataset

def calc_norm_stats(cfg, n_stats=100000):
    """Calculates statistics of log-mel spectrogram features in a data source for normalization.

    Args:
        cfg: Configuration settings.
        data_src: Data source class object.
        n_stats: Maximum number of files to calculate statistics.
    """
    cfg = load_yaml_config('config.yaml')
    ### 使用するwaveファイル名前が書かれたファイル。
    ### 全waveファイルリスト
    path = "/work/ge43/e43020/master_project/data/wav"
    with open('/work/ge43/e43020/master_project/UTMOS_BYOL-A/byola/txts/train_wavs.txt', 'r') as f:
        data_src = [path + line.rstrip() for line in f]

    stats_data = data_src
    n_stats = min(n_stats, len(stats_data))
    logging.info(f'Calculating mean/std using random {n_stats} samples from training population {len(stats_data)} samples...')
    sample_idxes = np.random.choice(range(len(stats_data)), size=n_stats, replace=False)
    ds = WaveInLMSOutDataset(cfg, stats_data, labels=None, tfms=None)
    X = [ds[i] for i in tqdm(sample_idxes)]
    X = np.hstack(X)
    norm_stats = np.array([X.mean(), X.std()])
    logging.info(f'  ==> mean/std: {norm_stats}, {norm_stats.shape} <- {X.shape}')
    return norm_stats


if __name__ == '__main__':
    calc_norm_stats(cfg=load_yaml_config('config.yaml'))