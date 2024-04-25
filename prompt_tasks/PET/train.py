from typing import Optional, List
import time
import os
from transformers import AutoModelForMaskedLM, AutoTokenizer, get_scheduler
import sys
from data_handle.data_loader import *

sys.path.append(r'F:\pythonProject1\bigmodelchatglm6-B\prompt_tasks\PET\data_handle')
sys.path.append(r'F:\pythonProject1\bigmodelchatglm6-B\prompt_tasks\PET\utils')
# from utils.metric_utils import ClassEvaluator
from utils.verbalizer import Verbalizer
from pet_config import *

pc = ProjectConfig()


def model2train():
    model = AutoModelForMaskedLM.from_pretrained(pc.pre_model)
    tokenizer = AutoTokenizer.from_pretrained(pc.pre_model)
    verbalizer = Verbalizer(verbalizer_file=pc.verbalizer, tokenizer=tokenizer, max_label_len=pc.max_label_len)
    no_decay = ['bias', 'LayerNorm.weights']
    optimizer_grouped_parameters = [{
        'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        'weight_decay': pc.weight_decay
    },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        }]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=pc.learning_rate)
    model.to(pc.device)

    train_dataloader, dev_dataloader = get_data()
    # 学习率预热
    # 根据训练轮数计算最大训练步数，以便于scheduler动态调整lr
    num_update_steps_per_epoch = len(train_dataloader)
    # 指定总的训练步数,它会被学习率调度器用来确定学习率的变化规律,确保学习率在整个训练过程中得以合理地调节
    max_train_steps = pc.epochs * num_update_steps_per_epoch
    warm_steps = int(pc.warmup_ratio * max_train_steps)
    lc_scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=warm_steps,
        num_training_steps=max_train_steps
    )
    loss_list=[]
