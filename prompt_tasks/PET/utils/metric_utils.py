from typing import Optional, List
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix


class ClassEvaluator():
    def __init__(self):
        self.goldens = []
        self.predictions = []

    def add_batch(self, pred_batch: List[List], gold_batch: List[List]):
        assert len(pred_batch) == len(gold_batch)
        print(f'pred_batch0-->{pred_batch}')
        print(f'gold_batch0-->{gold_batch}')
        if type(gold_batch[0]) in [List, tuple]:
            pred_batch = ["".join([str(e) for e in ele]) for ele in pred_batch]
            gold_batch = ["".join([str(e) for e in ele]) for ele in gold_batch]
        self.goldens.extend(gold_batch)
        self.predictions.extend(pred_batch)

    def compute(self, round_num=2) -> dict:
        classes, class_metrics, res = sorted(list(set(self.goldens) | set(self.predictions))), {}, {},
        # 构建全局指标
        res['accuracy'] = round(accuracy_score(self.goldens, self.predictions), round_num)
        res['precision'] = round(precision_score(self.goldens, self.predictions, average='weighted'), round_num)
        # average=weighted 代表:考虑类别的不平衡关系,需要计算类别的加权平均,如果是二分类问题,则选择参数'binary'
        res['recall'] = round(recall_score(self.goldens, self.predictions, average='weighted'), round_num)
        res['f1'] = round(f1_score(self.goldens, self.predictions, average='weighted'), round_num)
        try:
            conf_matrix = np.array(confusion_matrix(self.goldens, self.predictions))
            assert conf_matrix.shape[0] == len(classes)
            for i in range(conf_matrix.shape[0]):
                precision = 0 if sum(conf_matrix[:, i]) == 0 else conf_matrix[i, i] / sum(conf_matrix[:, i])
                recall = o if sum(conf_matrix[i, :]) == 0 else conf_matrix[i, i] / sum(conf_matrix[i, :])
                f1 = 0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
                class_metrics[classes[i]] = {
                    'precision': round(precision, round_num),
                    'recall': round(recall, round_num),
                    'f1': round(f1, round_num)
                }
            res['class_metrics'] = class_metrics
        except Exception as e:
            print(f'[Warning] Something wrong when calculate class_metrics: {e}')
            print(f'-> goldens: {set(self.goldens)}')
            print(f'-> predictions: {set(self.predictions)}')
            print(f'-> diff elements: {set(self.predictions) - set(self.goldens)}')
            res['class_metrics'] = {}
        return res

    def reset(self):
        # 重置积累的数值
        self.goldens = []
        self.predictions = []

