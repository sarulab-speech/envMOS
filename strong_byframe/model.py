import torch
import torch.nn as nn
from text.symbols import symbols
import fairseq
import os
import hydra
from byol_a2.common import load_yaml_config
from byol_a2.augmentations import PrecomputedNorm
from byol_a2.models import AudioNTT2022, load_pretrained_weights
import nnAudio.features
import torchaudio
import torch
import os

def load_ssl_model(cfg_path, cp_path):
    cfg = load_yaml_config(cfg_path)
    to_melspec = nnAudio.features.MelSpectrogram(
            sr=cfg.sample_rate,
            n_fft=cfg.n_fft,
            win_length=cfg.win_length,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
            fmin=cfg.f_min,
            fmax=cfg.f_max,
            center=True,
            power=2,
            verbose=False,
        )
    model = AudioNTT2022(n_mels=cfg.n_mels, d=cfg.feature_d)
    load_pretrained_weights(model, cp_path)
    out_dim = 3072
    return SSL_model(model, out_dim, to_melspec)

class SSL_model(nn.Module):
    def __init__(self, model, out_dim, to_melspec) -> None:
        super().__init__()
        self.model = model
        self.out_dim = out_dim
        self.to_melspec = to_melspec
    
    def forward(self,batch):
        wavnames = batch['wavname'] 
        lis = []
        for i in range(len(wavnames)):
            wavname = wavnames[i]
            feat = torch.load(f'/work/ge43/e43020/master_project/UTMOS_BYOL-A/envMOS/strong/data/byola_frame_rep/{wavname.split(".")[0]}.pt', map_location="cuda:0")
            lis.append(feat[0])
        x = torch.stack(lis, dim=0)
        # batch x time x 1024
        return {"ssl-feature":x}
    def get_output_dim(self):
        return self.out_dim
    
class PhonemeEncoder(nn.Module):
    def __init__(self, out_dim) -> None:
        super().__init__()
        self.out_dim = out_dim

    def forward(self,batch):
        wavnames = batch['wavname'] 
        lis = []
        for i in range(len(wavnames)):
            wavname = wavnames[i]
            feat = torch.load(f'/work/ge43/e43020/master_project/UTMOS_BYOL-A/envMOS/strong/data/RoBERTa/{wavname.split("/")[-1].split(".")[0]}.pt', map_location="cuda:0")
            lis.append(feat[0])
        x = torch.stack(lis, dim=0)
        # batch x 1024
        return {"phoneme-feature": x}

    def get_output_dim(self):
        return self.out_dim
### listener id を加えて、LSTMから特徴量抽出
class LDConditioner(nn.Module):
    def __init__(self,input_dim, judge_dim, num_judges, listener_emb):
        super().__init__()
        self.input_dim = input_dim
        self.listener_emb = listener_emb
        if listener_emb:
            self.judge_dim = judge_dim
            self.num_judges = num_judges
            self.judge_embedding = nn.Embedding(num_judges, self.judge_dim)
            self.decoder_rnn = nn.LSTM(
                input_size = self.input_dim + self.judge_dim, 
                hidden_size = 512,
                num_layers = 1,
                batch_first = True,
                bidirectional = True
            )
        else: 
            self.decoder_rnn = nn.LSTM(
                input_size = self.input_dim, 
                hidden_size = 512,
                num_layers = 1,
                batch_first = True,
                bidirectional = True
            )
        self.out_dim = self.decoder_rnn.hidden_size*2
    def get_output_dim(self):
        return self.out_dim
    def forward(self, x, batch):
        # 12 x frame x ???
        concatenated_feature = torch.cat((x['ssl-feature'], x['phoneme-feature'].unsqueeze(1).expand(-1,x['ssl-feature'].size(1) ,-1)),dim=2)
        if self.listener_emb:
            judge_ids = batch['judge_id']
            concatenated_feature = torch.cat((concatenated_feature, self.judge_embedding(judge_ids).unsqueeze(1).expand(-1, concatenated_feature.size(1), -1)),dim=2)
        decoder_output, (h, c) = self.decoder_rnn(concatenated_feature)
        # decoder_output は batch x frames x 1024
        # 第一と最終フレームの出力のみ得る。
        lis = []
        for i in range(len(batch['wavname'])):
            output = (decoder_output[i][0] + decoder_output[i][-1]) / 2
            lis.append(output.unsqueeze(0))
        x = torch.stack(lis, dim=0)
        # batch x 1024
        return x
    
### RELUで予測値算出
class Projection(nn.Module):
    ### 予測値算出
    def __init__(self, input_dim, hidden_dim, activation, range_clipping=False):
        super(Projection, self).__init__()
        self.range_clipping = range_clipping
        output_dim = 1
        if range_clipping:
            self.proj = nn.Tanh()
        ### batch x flame x feature
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation,
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim),
        )
        self.output_dim = output_dim
    
    def forward(self, x, batch):
        output = self.net(x)
        ### 1024がinput_dim, hiddenが2048. 最後に2048から1.
        ### では、batch x flame x 1 が出てくるのか。
        return output
    def get_output_dim(self):
        return self.output_dim