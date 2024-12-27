import torch
import torch.nn as nn
from WavLM import WavLM, WavLMConfig
from text.symbols import symbols
import fairseq
import os
import hydra

class SSL_model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self,batch):
        wavnames = batch['wavname'] 
        print(wavnames)
        lis = []
        for i in range(len(wavnames)):
            wavname = wavnames[i]
            feat = torch.load(f'/work/ge43/e43020/master_project/UTMOS_BYOL-A/envMOS/strong/data/byola{wavname.split(".")[0]}.pt', map_location="cuda:0")
            lis.append(feat)
        x = torch.stack(lis, dim=0)
        return {"ssl-feature":x}
    def get_output_dim(self):
        return 3072


class PhonemeEncoder(nn.Module):
    '''
    '''
    def __init__(self, hidden_dim, emb_dim, out_dim,n_lstm_layers) -> None:
        super().__init__()
        # self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.encoder = nn.LSTM(1024, hidden_dim,
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
        print(caption_batch.shape, "captionbatch")
        ### 256次元に。　batch x len(phonemes) x 256
        _, (ht, _) = self.encoder(seq)
        ### batch x sequence length (paddingされてるので、max(len(phonemes))) x 
        ### output, (h_n, c_n) が出力なので、 h_n だけを拾っている。
        feature = ht[-1] + ht[0]
        print(ht[-1].shape, "-1")
        print(ht[0].shape)
        
        ### 3層 がbidirectionalなので6 x 出力次元
        ### なぜ第一層と最終層なんだ？ なぜoutputではないんだ？
        ### batch x 256 (hidden size)
        feature = self.linear(feature)
        return {"phoneme-feature": feature}
    def get_output_dim(self):
        return self.out_dim

### listener id を加えて、LSTMから特徴量抽出
class LDConditioner(nn.Module):
    '''
    Conditions ssl output by listener embedding
    '''
    def __init__(self,input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.out_dim = 3072+256

    def get_output_dim(self):
        return self.out_dim


    def forward(self, x, batch):
        if 'phoneme-feature' in x.keys():
            print("caption............")
            ### unsqueeze で、第一次元に 要素数1の次元を挿入. expandで、第一次元、第３次元の大きさ変えずにexpandで、第２次元だけコピー。
            ### 第3次元で結合。batchかつ、フレーム毎か。
            ### ssl特徴量の 768に、+600とか横に足される。
            ### batch? x フレーム x ssl or phoneme or ...
            ### 今回は第一次元で結合で良さげ。
            ### 第1, 3は次元が一致？？？？
            ### ssl-feature は 768次元。　phonemeは256次元。
            concatenated_feature = torch.cat((x['ssl-feature'], x['phoneme-feature'].unsqueeze(1).expand(-1,x['ssl-feature'].size(1) ,-1)),dim=2)
        else:
            concatenated_feature = x['ssl-feature']
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
        if self.range_clipping:
            return self.proj(output) * 2.0 + 3
        else:
            return output
    def get_output_dim(self):
        return self.output_dim
