from typing import Optional, List
import torch
import sys

# print(sys.path)


class ProjectConfig():
    def __init__(self):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.pre_model = r'F:\bert-base-chinese'
        self.train_path = r'F:\pythonProject1\bigmodelchatglm6-B\prompt_tasks\PET\data\train.txt'
        self.dev_path = r'F:\pythonProject1\bigmodelchatglm6-B\prompt_tasks\PET\data\dev.txt'
        self.prompt_file = r'F:\pythonProject1\bigmodelchatglm6-B\prompt_tasks\PET\data\prompt.txt'
        self.verbalizer = r'F:\pythonProject1\bigmodelchatglm6-B\prompt_tasks\PET\data\verbalizer.txt'
        self.max_seq_len = 512
        self.batch_size = 8
        self.learning_rate = 5e-5
        self.weight_decay = 0
        self.warmup_ratio = 0.06
        self.max_label_len = 2
        self.epochs = 50
        self.logging_steps = 20
        self.save_dir = r'F:\pythonProject1\bigmodelchatglm6-B\prompt_tasks\PET\checkpoints'
if __name__ == '__main__':
    pc=ProjectConfig()
    print(pc.prompt_file)
    print(pc.pre_model)
