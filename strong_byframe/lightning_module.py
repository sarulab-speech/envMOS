import pytorch_lightning as pl
import torch
import torch.nn as nn
import os
import fairseq
import numpy as np
import pandas as pd
import scipy.stats
import hydra
from transformers import AdamW, get_linear_schedule_with_warmup
from model import SSL_model, PhonemeEncoder, LDConditioner, Projection
import wandb


class UTMOSLightningModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.construct_model()
        self.prepare_domain_table()
        self.save_hyperparameters()
    
    def construct_model(self):
        ### model.load_ssl_model, model.PhonemeEncoder, model.DomainEmbedding
        self.feature_extractors = nn.ModuleList([
            hydra.utils.instantiate(feature_extractor) for feature_extractor in self.cfg.model.feature_extractors
        ])
        ### それぞれの層での出力次元
        output_dim = sum([ feature_extractor.get_output_dim() for feature_extractor in self.feature_extractors])
        output_layers = []
        ### output_layers は  model.LDConditioner, model.Projection, torch.nn.ReLU
        for output_layer in self.cfg.model.output_layers:
            output_layers.append(
                hydra.utils.instantiate(output_layer,input_dim=output_dim)
            )
            ### 最終出力の次元
            output_dim = output_layers[-1].get_output_dim()

        self.output_layers = nn.ModuleList(output_layers)

        self.criterion = self.configure_criterion()

    def prepare_domain_table(self):
        self.domain_table = {}
        data_sources = self.cfg.dataset.data_sources
        for i, datasource in enumerate(data_sources):
            if not hasattr(datasource,'val_mos_list_path'):
                continue
            ### table1 → main, table2 → oddとか
            self.domain_table[i] = datasource["name"]

    def forward(self, inputs):
        outputs = {}
        for feature_extractor in self.feature_extractors:
            ### {"ssl-feature":x}, {"phoneme-feature": feature}, {"domain-feature": self.embedding(batch['domains'])}
            ### 各特徴量抽出器は、全てのデータ(wav, phonomeとか)をbatchとして受け取るが、その中で必要なもの (wav, phoneme)しか処理しない。
            outputs.update(feature_extractor(inputs))
        x = outputs
        ### {"domain-feature": self.embedding(batch['domains'])}, 
        for output_layer in self.output_layers:
            x = output_layer(x,inputs)
        ### 最終的には、ProjectionからのMOS予測値が戻ってくる。
        return x

    def training_step(self, batch, batch_idx):
        ### self()はforwardと同じ。
        outputs = self(batch)
        loss = self.criterion(outputs, batch['score'], batch["num_class"])
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.cfg.train.train_batch_size
        )
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = self.criterion(outputs, batch['score'], batch["num_class"])
        if outputs.dim() > 1:
            outputs = outputs.mean(dim=1).squeeze(-1)
        # このreturnがlogに残るのか
        return {
            "loss": loss,
            "outputs": outputs.cpu().numpy()[0]*5+5,
            "filename": batch["wavname"][0],
            "domain": batch["domain"][0],
            "raw_avg_score": batch["raw_avg_score"][0].item()
        }

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([out["loss"] for out in outputs]).mean().item()
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True, logger=True)
        for domain_id in self.domain_table:
            outputs_domain = [out for out in outputs if out["domain"] == domain_id]
            if len(outputs_domain) == 0:
                continue
            _, SRCC, MSE = self.calc_score(outputs_domain)
            print("val_SRCC_system_print",SRCC)
            self.log(
                "val_SRCC_system",
                SRCC,
                on_epoch=True,
                prog_bar=True,
                logger=True
            )
            self.log(
                "val_MSE_system",
                MSE,
                on_epoch=True,
                prog_bar=True,
                logger=True
            )

    def test_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = self.criterion(outputs, batch['score'], batch["num_class"])
        print(outputs, "output")
        print(batch['score'])
        labels = batch['score']
        filenames = batch['wavname']
        if outputs.dim() > 1:
            outputs = outputs.mean(dim=1).squeeze(-1)
        return {
            "loss": loss,
            "outputs": outputs.cpu().detach().numpy()[0]*5+5,
            "labels": labels.cpu().detach().numpy()[0]*5+5,
            "filename": filenames[0],
            "domain": batch["domain"][0],
            "i_cv": batch["i_cv"][0],
            "set_name": batch["set_name"][0],
            "raw_avg_score": batch["raw_avg_score"][0].item()
        }

    def test_epoch_end(self, outputs):
        outfiles = [datasource["outfile"] + '{}_{}'.format(outputs[0]['set_name'],outputs[0]['i_cv']) for datasource in self.cfg.dataset.data_sources if hasattr(datasource,'outfile')]
        for domain_id in self.domain_table:
            outputs_domain = [out for out in outputs if out["domain"] == domain_id]
            predictions, SRCC, MSE = self.calc_score(outputs_domain)
            self.log(
                "test_SRCC_SYS_{}_i_cv_{}_set_name_{}".format(self.domain_table[domain_id], outputs[0]['i_cv'], outputs[0]['set_name']),
                SRCC,
            )
            if domain_id == 0:
                self.log(
                    "test_SRCC_SYS".format(self.domain_table[domain_id]),
                    SRCC
                )
            # answer-main.csv_-1
            with open(outfiles[domain_id], "w") as fw:
                for k, v in predictions.items():
                    outl = k.split(".")[0] + "," + str(v) + "\n"
                    fw.write(outl)
            try:
                wandb.save(outfiles[domain_id])
            except:
                print('outfile {} saved'.format(outfiles[domain_id]))

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            self.cfg.train.optimizer,
            params=self.parameters()
        )
        scheduler = hydra.utils.instantiate(
            self.cfg.train.scheduler,
            optimizer=optimizer
        )
        scheduler = {"scheduler": scheduler,
                     "interval": "step", "frequency": 1}
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def configure_criterion(self):
        return hydra.utils.instantiate(self.cfg.train.criterion,_recursive_=True)

    def calc_score(self, outputs, verbose=False):


        predictions = {}
        true_MOS = {}
        for out in outputs:
            predictions[out["filename"]] = out["outputs"]
            true_MOS[out["filename"]] = out["raw_avg_score"]

        sorted_filenames = sorted(predictions.keys())
        ts = []
        ps = []
        for filename in sorted_filenames:
            t = true_MOS[filename]
            p = predictions[filename]
            ts.append(t)
            ps.append(p)

        truths = np.array(ts)
        preds = np.array(ps)
        stacked_array = np.stack((truths, preds), axis=1)
        df = pd.DataFrame(stacked_array, columns=['truths', 'preds'])
        df.to_csv('output.csv', index=False)

        ### raw
        MSE = np.mean((truths - preds) ** 2)
        LCC = np.corrcoef(truths, preds)
        SRCC = scipy.stats.spearmanr(truths.T, preds.T)
        KTAU = scipy.stats.kendalltau(truths, preds)
        if verbose:
            print("[RAW] Test error= %f" % MSE)
            print("[RAW] Linear correlation coefficient= %f" % LCC[0][1])
            print("[RAW] Spearman rank correlation coefficient= %f" % SRCC[0])
            print("[RAW] Kendall Tau rank correlation coefficient= %f" % KTAU[0])
        return predictions, SRCC[0], MSE



class DeepSpeedBaselineLightningModule(UTMOSLightningModule):
    def __init__(self, cfg):
        super().__init__(cfg)

    def configure_optimizers(self):
        from deepspeed.ops.adam import DeepSpeedCPUAdam
        return DeepSpeedCPUAdam(self.parameters())