import copy
import time

import numpy as np
import torch
from tqdm import tqdm
from sklearn import metrics

import himallgg

log = himallgg.utils.get_logger()


class Prediction:

    def __init__(self, testset, model, args):
        self.testset = testset
        self.model = model
        self.args = args
        self.best_dev_f1 = None
        self.best_tes_f1 = None
        self.test_f1_when_best_dev = None
        self.best_epoch = None
        self.best_state = None

    def load_ckpt(self, ckpt):
        self.best_dev_f1 = ckpt["best_dev_f1"]
        self.best_tes_f1 = ckpt["best_tes_f1"]
        self.test_f1_when_best_dev = ckpt['test_f1_when_best_dev']
        self.best_epoch = ckpt["best_epoch"]
        self.best_state = ckpt["best_state"]
        self.model.load_state_dict(self.best_state)

    def evaluate(self):
        dataset = self.testset

        self.model.eval()
        with torch.no_grad():
            golds = []
            preds = []
            for idx in tqdm(range(len(dataset)), desc="test"):
                data = dataset[idx]
                golds.append(data["label_tensor"])
                for k, v in data.items():
                    if k == 'sentence':
                        continue
                    else:
                        data[k] = v.to(self.args.device)
                y_hat = self.model(data)
                preds.append(y_hat.detach().to("cpu"))

            golds = torch.cat(golds, dim=-1).numpy()
            preds = torch.cat(preds, dim=-1).numpy()

            print(metrics.classification_report(golds, preds, digits=4))
            f1 = metrics.f1_score(golds, preds, average="weighted")

        return f1
