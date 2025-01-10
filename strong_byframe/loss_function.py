from numpy import dtype
import torch
import torch.nn as nn

class ContrastiveLoss(nn.Module):
    '''
    Contrastive Loss
    Args:
        margin: non-neg value, the smaller the stricter the loss will be, default: 0.2        
        
    '''
    def __init__(self, margin, beta):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.beta = beta
    
    def forward(self, pred_score, gt_score, num_class):
        ### 3次元以上なら。 batch x flame x score
        if pred_score.dim() > 2:
            ### batch x score に squeeze(1)なので、 batch の数だけ、scoreが並んでるのか。
            pred_score = pred_score.mean(dim=1).squeeze(1)
        # pred_score, gt_score: tensor, [batch_size]  
        gt_diff = gt_score.unsqueeze(1) - gt_score.unsqueeze(0)
        # 差行列になるらしい。
        pred_diff = pred_score.unsqueeze(1) - pred_score.unsqueeze(0)
        ### 行列間で差をとっているのか。
        ### 最大の損失？？
        ### margin = 0.1だが?
        ### marginより小さい部分は0にして差行列取得
        loss = torch.maximum(torch.zeros(gt_diff.shape).to(gt_diff.device), torch.abs(pred_diff - gt_diff) - self.margin) 
        ### 読み込んで、行列に。torch.outer(vector1, vector2)。　ルートとって、　要素積をとる。

        # 読み込んで unsqueeze, torch.mul(vector1, vector2)で要素ごとの積
        ### 12 x 1
        num_class = num_class
        weights = (1 - self.beta) / (1 - torch.pow(self.beta, num_class))
        class_mat = torch.outer(weights, weights)
        class_mat = torch.sqrt(class_mat)

        loss = torch.mul(loss, class_mat)
        ### 全要素で平均をとって２で割る。
        loss = loss.mean().div(2)
        return loss


class ClippedMSELoss(nn.Module):
    """
    clipped MSE loss for listener-dependent model
    """
    def __init__(self, criterion,tau, mode, beta):
        super(ClippedMSELoss, self).__init__()
        self.tau = torch.tensor(tau,dtype=torch.float)

        self.criterion = criterion
        self.mode = mode
        self.beta = beta


    def forward_criterion(self, y_hat, label, num_class):
        ### batch x flame x 1 (score)
        ### squeeze で最後の次元を削除
        
        y_hat = y_hat.squeeze(-1)
        ### MSEloss
        loss = self.criterion(y_hat, label)
        threshold = torch.abs(y_hat - label) > self.tau
        ### lossに書くクラスの頻度の逆数をかける
        # 読み込んで unsqueeze, torch.mul(vector1, vector2)で要素ごとの積
        ### 12 x 1
        num_class = num_class.unsqueeze(1)
        weights = (1 - self.beta) / (1 - torch.pow(self.beta, num_class))
        loss = torch.mul(loss, weights)

        ### thresholdを超えてるものはその値、超えてないものは0として平均とる。
        loss = torch.mean(threshold * loss)
        return loss

    def forward(self, pred_score, gt_score, num_class):
        """
        Args:
            pred_mean, pred_score: [batch, time, 1/5]
        """
        # repeat for frame level loss
        time = pred_score.shape[1]
        if self.mode == 'utt':
            pred_score = pred_score.mean(dim=1)
        else:
            gt_score = gt_score.unsqueeze(1).repeat(1, time)
        main_loss = self.forward_criterion(pred_score, gt_score, num_class)
        return main_loss # lamb 1.0  

class CombineLosses(nn.Module):
    '''
    Combine losses
    Args:
        loss_weights: a list of weights for each loss
    '''
    def __init__(self, loss_weights:list, loss_instances:list):
        super(CombineLosses, self).__init__()
        self.loss_weights = loss_weights
        self.loss_instances = nn.ModuleList(loss_instances)
    def forward(self, pred_score, gt_score, num_class):
        loss = torch.tensor(0,dtype=torch.float).to(pred_score.device)
        for loss_weight, loss_instance in zip(self.loss_weights, self.loss_instances):
            loss += loss_weight * loss_instance(pred_score,gt_score, num_class)
        return loss
