from sklearn.metrics import accuracy_score
import numpy as np
class DataClass:
    def __init__(self, type):
        self.loss_list=[]
        self.preds=[]
        self.gts=[]
        self.mean_acc = 0
        self.mean_loss = 0
        self.type = type
    def calculate_mean_acc(self):
        for i in range(len(self.gts)):
            self.mean_acc += accuracy_score(self.gts[i].round(), [1 if i >=0.5 else 0 for i in self.preds[i]] )
        self.mean_acc /= len(self.gts)

    def calculate_mean_loss(self):
        self.mean_loss = np.mean(self.loss_list)

    def add_log(self, loss, pred_logits, labels):
        self.loss_list.append(loss.item())
        self.preds.extend(pred for pred in pred_logits.detach().cpu().numpy())
        self.gts.extend(gt for gt in labels.detach().cpu().numpy())