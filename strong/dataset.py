from typing import Any, Dict, List
import augment
import torch
import torchaudio
import os
import random
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import hydra
from phonemizer import phonemize
import pandas as pd
from data_augment import ChainRunner, random_pitch_shift, random_time_warp
import text.symbols as symbols
import numpy as np
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence


class DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

    def setup(self, stage):
        ocwd = hydra.utils.get_original_cwd()
        join = os.path.join
        data_sources = self.cfg.dataset.data_sources
        self.wavdir = {} 
        ### 辞書ができる。{"name" : main, "train_mos_list_path": data/phase1-main/DATA/sets/TRAINSET}みたいな。
        for datasource in data_sources:
            ##  wavdir["main"] = cwd/data/phase1-main/DATA/wav/
            self.wavdir[datasource.name] = join(ocwd, datasource['wav_dir'])  
        # 一つのdatasource は、 "name": main から "outfile": answer-main.csv まで。
        # の辞書
        self.datasources = data_sources
        ### train_mos_list_path から、*.wavとMOS値入手
        # for 文を使っているが、train_mos_list_path: data/phase1-main/DATA/sets/TRAINSET　のみに対して反応。
        # main, odd, external のそれぞれのtrain_path
        train_paths = [join(ocwd,data_source.train_mos_list_path) for data_source in data_sources if hasattr(data_source,'train_mos_list_path')]
        val_paths = [join(ocwd,data_source.val_mos_list_path) for data_source in data_sources if hasattr(data_source,'val_mos_list_path')]
        ### data_source.nameは、main, oddとか。
        domains_train = [data_source.name for data_source in data_sources if hasattr(data_source,'train_mos_list_path')]
        domains_val = [data_source.name for data_source in data_sources if hasattr(data_source,'val_mos_list_path')]

        self.mos_df = {}
        ### TRAINSETから、扱いやすい形のcsvへ加工
        ### ,filename,rating,listener_name,domain,listener_id,domain_id 
        self.mos_df['train'] = self.get_mos_df(train_paths,domains_train,self.cfg.dataset.only_mean)
        self.mos_df['val'] = self.get_mos_df(val_paths,domains_val,only_mean=True,id_reference=self.mos_df['train'])
        ###  ,filename,rating,listener_name,domain,listener_id,domain_id
        ### 9,sys0eb39-utt4745b38.wav,3.0,oJgQdRV65lnW,main,9,0
        self.mos_df['train'].to_csv("listener_embedding_lookup.csv")
       

    def get_mos_df(self, paths:List[str], domains:List[str],only_mean=False,id_reference=None):
        assert len(domains) == len(paths)
        dfs = [] 
        ## main, odd, external 全て含まれてたら全部合わせた dfsができる。
        for path, domain in zip(paths,domains):
            df = pd.read_csv(path,names=["csv_name", "filename", "rating", "model", "caption", "temp_flag"])
            listener_df = pd.DataFrame()
            listener_df['filename'] = df['filename']
            listener_df['rating'] = df['rating']
            # ZPGlxO3OmLRp
            # listener_df['listener_name'] = df['listener_info'].str.split('_').str[2]
            listener_df['listener_name'] = df['csv_name']
            listener_df['domain'] = domain


            # 音ファイル単位でのmean
            mean_df = pd.DataFrame(listener_df.groupby('filename',as_index=False)['rating'].mean())
            mean_df['listener_name']= f"MEAN_LISTENER_{domain}"
            mean_df['domain'] = domain

            if only_mean:
                dfs.append(mean_df)
            else:
                ### 複数のappend みたいな感じ。 
                dfs.extend([listener_df,mean_df])
            
        return_df = pd.concat(dfs,ignore_index=True)
        if id_reference is None:
            ### 数値化 [a,b,c,a] → [0, 1, 2, 0]に。
            return_df['listener_id'] = return_df["listener_name"].factorize()[0]
            return_df['domain_id'] = return_df['domain'].factorize()[0]
        else:
            listener_id = []
            domain_id = []
            for idx, row in return_df.iterrows():
                # id_referenceの["listener_name"] と rowの["listener_name"]が一致する 行を抜き出す。そのlistener_idを取得。
                # for ループで、main, odd, external が順に見られるが、MEAN_LISTENER_main が同じになることはない。mainの部分が違うので。
                # これを各行について行う。
                listener_id.append( id_reference[id_reference['listener_name'] == row['listener_name']]['listener_id'].iloc[0])
                domain_id.append(id_reference[id_reference['domain'] == row['domain']]['domain_id'].iloc[0])
            return_df['listener_id'] = listener_id
            return_df['domain_id'] = domain_id
        return return_df

    def get_ds(self, phase):
        ds = MyDataset(
            self.wavdir,
            self.mos_df[phase],
            phase=phase,
            cfg=self.cfg,
        )
        return ds
    def get_loader(self, phase, batchsize):
        ### データゲット。
        ds = self.get_ds(phase)
        ### batchsizeとか指定してdataloaderへ。
        dl = DataLoader(
            ds,
            batchsize,
            shuffle=True if phase == "train" else False,
            num_workers=8,
            collate_fn=ds.collate_fn,
        )
        return dl

    def train_dataloader(self):
        return self.get_loader(phase="train", batchsize=self.cfg.train.train_batch_size)

    def val_dataloader(self):
        return self.get_loader(phase="val", batchsize=self.cfg.train.val_batch_size)

    def test_dataloader(self):
        return self.get_loader(phase="val", batchsize=self.cfg.train.test_batch_size)

class TestDataModule(DataModule):
    '''
        DataModule used for CV and test
        This is only used for inference so it has no train data
        Args:
            i_cv: number of fold
            set_name: test, val
    '''
    def __init__(self, cfg, i_cv, set_name):
        super().__init__(cfg)

        self.i_cv = i_cv
        self.set_name = set_name

    def setup(self, stage):
        ocwd = hydra.utils.get_original_cwd()
        join = os.path.join
        data_sources = self.cfg.dataset.data_sources
        self.wavdir = {} 
        for datasource in data_sources:
            self.wavdir[datasource.name] = join(ocwd, datasource['wav_dir'])  
        train_paths = [join(ocwd,data_source.train_mos_list_path) for data_source in data_sources if hasattr(data_source,'train_mos_list_path')]
        domains_train = [data_source.name for data_source in data_sources if hasattr(data_source,'train_mos_list_path')]
        ### 今が valモードか、testモードかでpathの場合分け。
        if self.set_name == 'val': 
            mos_list_path = 'val_mos_list_path'
        elif self.set_name == 'test':
            mos_list_path = 'test_mos_list_path'
        elif self.set_name == 'test_post':
            mos_list_path = 'test_post_mos_list_path'
        ### testかval。 test.scp or DEVSETへのpath. それが、main, odd, externalの３つ分。
        val_paths = [join(ocwd,getattr(data_source, mos_list_path)) for data_source in data_sources if hasattr(data_source, mos_list_path)]
        domains_val = [data_source.name for data_source in data_sources if hasattr(data_source, mos_list_path)]

        self.mos_df = {}
        self.mos_df['train'] = self.get_mos_df(train_paths,domains_train,self.cfg.dataset.only_mean)
        
        if self.set_name == 'val':
            ### なんでonly_meanがtrue?
            self.mos_df['val'] = self.get_mos_df(val_paths,domains_val,only_mean=True,id_reference=self.mos_df['train'])
        elif self.set_name == 'test':
            ### only_mean == False かな。trainと同じ話者からはidを参考にできるようにしているのか。
            self.mos_df['val'] = self.get_test_df(val_paths,domains_val,id_reference=self.mos_df['train'])
        elif self.set_name == 'test_post':
            self.mos_df['val'] = self.get_mos_df(val_paths,domains_val,only_mean=True,id_reference=self.mos_df['train'])


    def get_mos_df(self, paths:List[str], domains:List[str],only_mean=False,id_reference=None):
        return_df = super().get_mos_df(paths, domains, only_mean, id_reference)
        return_df['i_cv'] = self.i_cv
        ### testとか。
        return_df['set_name'] = self.set_name
        return return_df
    
    def get_test_df(self, paths, domains, id_reference):
        assert len(domains) == len(paths)
        dfs = [] 
        for path, domain in zip(paths,domains):
            ### open test.scp
            df_raw = pd.read_csv(path,names=["csv_name", "filename", "rating", "model", "caption", "temp_flag"])
            df = pd.DataFrame()
            df['filename'] = df_raw["filename"]
            df['rating'] = df_raw["rating"]

            ### mean_listerに評価させている？
            df['listener_name'] = f'MEAN_LISTENER_{domain}'
            df['domain'] = domain
            dfs.append(df)
        return_df = pd.concat(dfs,ignore_index=True)
        listener_id = []
        domain_id = []
        for idx, row in return_df.iterrows():
            ### mean_listerに評価させている。
            listener_id.append(id_reference[id_reference['listener_name'] == row['listener_name']]['listener_id'].iloc[0])
            domain_id.append(id_reference[id_reference['domain'] == row['domain']]['domain_id'].iloc[0])
        return_df['listener_id'] = listener_id
        return_df['domain_id'] = domain_id
        return_df['i_cv'] = self.i_cv
        return_df['set_name'] = self.set_name
        return return_df

class CVDataModule(DataModule):
    def __init__(self, cfg, k_cv, i_cv,fold_target='main'):
        super().__init__(cfg)

        self.k_cv = k_cv
        self.i_cv = i_cv
        self.seed_cv = 0
        self.fold_target_datset = fold_target


    def setup(self, stage):
        super().setup(stage)
        
        target_id = {}
        for idx, datasource in enumerate(self.datasources):
            ### {"main": idx}...
            target_id[datasource['name']] = idx
        ### 例えば、main だけを抜き出す
        target_df = self.mos_df["train"][self.mos_df["train"]['domain'] == self.fold_target_datset]
        ### odd, external
        not_target_df = self.mos_df["train"][self.mos_df["train"]['domain'] != self.fold_target_datset]
        ### frac=1 は100%
        shuffled_train_df = target_df.sample(frac=1,random_state=self.seed_cv)
        ### k個に分割 [0_df, ... , k_df]のリスト
        chuncked_train_df = np.array_split(shuffled_train_df,self.k_cv)
        ### i_cv番目の分割について、"listener_name"に"MEAN_LISTENER"を含むもののみ抜き出す
        self.mos_df['val'] = chuncked_train_df[self.i_cv][chuncked_train_df[self.i_cv]['listener_name'].str.contains("MEAN_LISTENER")]
        ### i_cv番目のdfをリストから抜き出す。
        chuncked_train_df.pop(self.i_cv)
        self.mos_df['val'] = self.mos_df['val'].reset_index()
        chuncked_train_df.append(not_target_df)
        self.mos_df['train'] = pd.concat(chuncked_train_df,ignore_index=True).reset_index()
        print("-"*20)
        print(len(self.mos_df['train']))
        print("-"*20)
        ### self.mos_df["train"] と self.mos_df['val']) の間で重複する "filename" を持つ行を削除。同じwavは使わないということか
        ### クロスバリデーションってそうだっけ? testだからそりゃそうか。
        self.mos_df["train"] = self.mos_df["train"][~self.mos_df["train"]["filename"].isin(self.mos_df['val']["filename"])]
        print(len(self.mos_df['train']))
        print("-"*20)
        #### train, val のデータを読み込んだやつから作ったのか。
        
    
    def get_mos_df(self, paths:List[str], domains:List[str],only_mean=False,id_reference=None):
        return_df = super().get_mos_df(paths, domains, only_mean, id_reference)
        return_df['i_cv'] = self.i_cv
        return_df['set_name'] = 'fold'
        return return_df
        

class MyDataset(Dataset):
    def __init__(self, wavdir, mos_df,phase, cfg,padding_mode='repetitive'):

        self.wavdir = wavdir if type(wavdir) != str else list(wavdir)
        self.additional_datas = []
        self.cfg = cfg
        self.padding_mode = padding_mode
        self.mos_df = mos_df

        # calc mean score by utterance
        raw_ratings = defaultdict(list)
        sys_ratings = defaultdict(list)
        utt_ratings = defaultdict(list)
        ### 行ごとに
        for _, row in mos_df.iterrows():
            wavname = row["filename"]
            ### sys0eb39-utt45dc367.wav
            ### utt → utt45dc367
            ### sys → sys0eb39
            # utt_ratings[wavname.split("-")[1].split(".")[0]].append(row["rating"])
            # 辞書に "111000": [4, 7] とか。
            raw_ratings[wavname].append(row["rating"])
            utt_ratings[wavname.split("/")[-1].split(".")[0]].append(row["rating"])
            # tangoとか
            sys_ratings[wavname.split("/")[1]].append(row["rating"])
        self.raw_avg_score_table = {}
        self.utt_avg_score_table = {}
        self.sys_avg_score_table = {}
        ### 発話単位、システム単位で平均
        for key in raw_ratings:
            self.raw_avg_score_table[key] = sum(raw_ratings[key])/len(raw_ratings[key])
        for key in utt_ratings:
            self.utt_avg_score_table[key] = sum(utt_ratings[key])/len(utt_ratings[key])
        for key in sys_ratings:
            self.sys_avg_score_table[key] = sum(sys_ratings[key])/len(sys_ratings[key])

        ### dataset.PhonemeData, dataset.NormalizeScore, dataset.AugmentWav, dataset.SliceWav が additional_data
        for i in range(len(self.cfg.dataset.additional_datas)):
            self.additional_datas.append(
                hydra.utils.instantiate(
                    self.cfg.dataset.additional_datas[i],
                    cfg=self.cfg,
                    phase=phase,
                    _recursive_=False,
                )
            )

    def __getitem__(self, idx):
        ### 行
        selected_row = self.mos_df.iloc[idx]
        wavname = selected_row['filename']
        score = selected_row['rating']
        domain_id = selected_row['domain_id']
        listener_id = selected_row['listener_id']
        ### i_cvはクロスバリデーションの時にどうやって使うんだ
        i_cv = selected_row['i_cv'] if 'i_cv' in selected_row else -1
        ### test, val, fold　など
        set_name = selected_row['set_name'] if 'set_name' in selected_row else ''
        ### 今読み込んでいるwavについて、その発話、システムでの平均取得
        # utt_avg_score = self.utt_avg_score_table[wavname.split("-")[1].split(".")[0]]
        # sys_avg_score = self.sys_avg_score_table[wavname.split("-")[0]]
        raw_avg_score = self.raw_avg_score_table[wavname]
        utt_avg_score = self.utt_avg_score_table[wavname.split("/")[-1].split(".")[0]]
        sys_avg_score = self.sys_avg_score_table[wavname.split("/")[1]]
        data = {
            'score': score,
            'wavname': wavname,
            'domain': domain_id,
            'judge_id': listener_id,
            'i_cv': i_cv,
            'set_name': set_name,
            'raw_avg_score': raw_avg_score,
            'utt_avg_score': utt_avg_score,
            'sys_avg_score': sys_avg_score
        }
        for additional_data_instances in self.additional_datas:
            ### dataに対して、additional_data_instancesの各処理を施す。
            ### 同じkeyを持つものは上書き。ないものは追加。
            data.update(additional_data_instances(data))
        return data

    def __len__(self):
        return len(self.mos_df)

    def collate_fn(self, batch):  # zero padding
        ### batch内の全てを開く。
        scores = [b['score'] for b in batch]
        wavnames = [b['wavname'] for b in batch]
        domains = [b['domain'] for b in batch]
        judge_id = [b['judge_id'] for b in batch]
        i_cvs = [b['i_cv'] for b in batch]
        set_names = [b['set_name'] for b in batch]
        raw_avg_scores = [b['raw_avg_score'] for b in batch]
        utt_avg_scores = [b['utt_avg_score'] for b in batch]
        sys_avg_scores = [b['sys_avg_score'] for b in batch]
        scores = torch.stack([torch.tensor(x,dtype=torch.float) for x in list(scores)], dim=0)
        ### 中身をtorch.float の tensorに変換。その後、リストにして、stackでテンソルに。
        ### 全体を通してやりたいことは、中身をtensorに変換かな。
        raw_avg_scores = torch.stack([torch.tensor(x,dtype=torch.float) for x in list(raw_avg_scores)], dim=0)
        utt_avg_scores = torch.stack([torch.tensor(x,dtype=torch.float) for x in list(utt_avg_scores)], dim=0)
        sys_avg_scores = torch.stack([torch.tensor(x,dtype=torch.float) for x in list(sys_avg_scores)], dim=0)
        domains = torch.stack([torch.tensor(x) for x in list(domains)], dim=0)
        judge_id = torch.stack([torch.tensor(x) for x in list(judge_id)], dim=0)
        collated_batch = {
            'score': scores,
            'raw_avg_score': raw_avg_scores,
            'utt_avg_score': utt_avg_scores,
            'sys_avg_score': sys_avg_scores,
            'wavname': wavnames,
            'domains': domains,
            'judge_id': judge_id,
            'domain': domains,
            'i_cv': i_cvs,
            'set_name': set_names
        } # judge id, domain, averaged score
        for additional_data_instance in self.additional_datas:
            ### このcollate_n は、AdditionalDataBase()内の関数。
            ### wav augumentationとかは、__get__itemのところでできているので問題なし。
            ### ここのcollate_fnは、NormalizeScore, AugumentWavとかは、空の辞書を返す。
            additonal_collated_batch = additional_data_instance.collate_fn(batch)
            collated_batch.update(additonal_collated_batch)
        return collated_batch

class AdditionalDataBase():
    def __init__(self, cfg=None) -> None:
        self.cfg = cfg
    ### 呼び出された時に、process_dataされた値を返す。ここでは、wav augumentationとか。
    def __call__(self, data: Dict[str, Any]):
        return self.process_data(data)

    def process_data(self, data: Dict[str, Any]):
        raise NotImplementedError

    def collate_fn(self, batch):
        return dict()

### 標準化
class NormalizeScore(AdditionalDataBase):
    def __init__(self, org_max, org_min,normalize_to_max,normalize_to_min,phase,cfg=None) -> None:
        super().__init__()
        self.org_max = org_max
        self.org_min = org_min
        self.normalize_to_max = normalize_to_max
        self.normalize_to_min = normalize_to_min
    def process_data(self, data: Dict[str, Any]):
        score = data['score']
        score = (score - (self.org_max + self.org_min)/2.0) / 5
        return {'score': score}

### データ拡張か。でも、音ファイルだけなのか。一回に入力する音ファイルを２つにするだけで、音、MOS値とかの組みが入るわけではないのか？
### 毎回ある変化を加えたwavを入力しているのか。じゃあ元と全く同じwavは入れてないんだ。 trainデータを何周かするから、ってことか？
