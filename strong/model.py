import torch
import torch.nn as nn
from WavLM import WavLM, WavLMConfig
from text.symbols import symbols
import fairseq
import os
import hydra

class SSL_model(nn.Module):
    def __init__(self, out_dim, path) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.path = path

    def forward(self,batch):
        wavnames = batch['wavname'] 
        print(wavnames)
        lis = []
        for i in range(len(wavnames)):
            wavname = wavnames[i]
            feat = torch.load(self.path + f'{wavname.split(".")[0]}.pt', map_location="cuda:0")
            lis.append(feat)
        x = torch.stack(lis, dim=0)
        return {"ssl-feature":x}
    def get_output_dim(self):
        return self.out_dim


class PhonemeEncoder(nn.Module):
    def __init__(self, hidden_dim, out_dim,n_lstm_layers, llm_dim) -> None:
        super().__init__()
        self.encoder = nn.LSTM(llm_dim, hidden_dim,
                               num_layers=n_lstm_layers, dropout=0.1, batch_first=True, bidirectional=True)
        self.linear = nn.Sequential(
                ### input_dim は hidden_dim + hidden_dim*self.with_reference
                ### input 256, outputは256
                nn.Linear(hidden_dim, out_dim),
                nn.ReLU()
                )
        self.out_dim = out_dim

    def forward(self,batch):
        wavnames = batch['wavname'] 
        lis = []
        lens = []
        for i in range(len(wavnames)):
            wavname = wavnames[i]
            feat = torch.load(f'/work/ge43/e43020/master_project/UTMOS_BYOL-A/envMOS/strong/data/RoBERTa/{wavname.split("/")[-1].split(".")[0]}.pt', map_location="cuda:0")
            lis.append(feat)
            lens.append(len(feat))

        caption_batch = torch.nn.utils.rnn.pad_sequence(lis, batch_first=True)
        seq = caption_batch
        ### 256次元に。　batch x len(phonemes) x 256
        _, (ht, _) = self.encoder(seq)
        ### batch x sequence length (paddingされてるので、max(len(phonemes))) x 
        ### output, (h_n, c_n) が出力なので、 h_n だけを拾っている。
        feature = ht[-1] + ht[0]
        
        ### 3層 がbidirectionalなので6 x 出力次元
        ### なぜ第一層と最終層なんだ？ なぜoutputではないんだ？
        ### batch x 256 (hidden size)
        feature = self.linear(feature)
        return {"phoneme-feature": feature}
    def get_output_dim(self):
        return self.out_dim

### listener id を加えて、LSTMから特徴量抽出
class LDConditioner(nn.Module):
    def __init__(self,input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.out_dim = 3072+256

    def get_output_dim(self):
        return self.out_dim

    def forward(self, x, batch):
        ### ssl特徴量の 768に、+600とか横に足される。
        ### batch? x フレーム x ssl or phoneme or ...
        concatenated_feature = torch.cat((x['ssl-feature'], x['phoneme-feature'].unsqueeze(1).expand(-1,x['ssl-feature'].size(1) ,-1)),dim=2)
        return concatenated_feature

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
        ### 1024がinput_dim 
        ### hiddenが2048. 最後に2048から1.
        ### では、batch x flame x 1 が出てくるのか。
        # range clipping
        return output
    def get_output_dim(self):
        return self.output_dim
