from typing import Optional, List
from rich import print
from transformers import AutoTokenizer
import numpy as np
import sys

sys.path.append('..')
from prompt_tasks.PET.pet_config import *


class HardTemplate():
    def __init__(self, prompt):
        self.prompt = prompt
        self.inputs_list = []
        self.custom_tokens = set(['MASK'])
        self.prompt_analysis()

    def prompt_analysis(self):
        idx = 0
        while idx < len(self.prompt):
            str_part = ''
            if self.prompt[idx] not in ['{', '}']:
                self.inputs_list.append(self.prompt[idx])
            if self.prompt[idx] == '{':
                idx += 1
                while self.prompt[idx] != '}':
                    str_part += self.prompt[idx]
                    idx += 1
            elif self.prompt[idx] == '}':
                raise ValueError('Unmatched bracket "}",check your prompt')
            if str_part:
                self.inputs_list.append(str_part)
                self.custom_tokens.add(str_part)
            idx += 1

    def __call__(self, inputs_dict: dict, tokenizer, max_length, max_seq_len=512):
        outputs = {
            'text': '',
            'input_ids': [],
            'token_type_ids': [],
            'attention_mask': [],
            'mask_position': []
        }
        str_formated = ''
        for value in self.inputs_list:
            if value in self.custom_tokens:
                if value == 'MASK':
                    str_formated += inputs_dict[value] * max_length
                else:
                    str_formated += inputs_dict[value]
            else:
                str_formated += value
        # print('str_formated-->', str_formated)
        encoded = tokenizer(text=str_formated, truncation=True, max_length=max_seq_len, padding='max_length')
        outputs['input_ids'] = encoded['input_ids']
        outputs['token_type_ids'] = encoded['token_type_ids']
        outputs['attention_mask'] = encoded['attention_mask']
        token_list = tokenizer.convert_ids_to_tokens(encoded['input_ids'])
        # print('token_list-->', token_list)
        # outputs['text'] = ''.join(token_list)
        mask_token_id = tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
        # print('mask_token_id-->', mask_token_id)
        condition = np.array(outputs['input_ids']) == mask_token_id
        # print('condition-->', condition)
        mask_position = np.where(condition)[0].tolist()
        # print('mask_position-->', mask_position)
        outputs['mask_position'] = mask_position
        # print('outputs-->', outputs)
        return outputs


if __name__ == '__main__':
    ht=HardTemplate('"这是一条{MASK}评论：{textA}。"')

    # print(ht.inputs_list)
    # print(ht.custom_tokens)
    pc = ProjectConfig()
    tokenizer = AutoTokenizer.from_pretrained(pc.pre_model)
    hard_template = HardTemplate(prompt='这是一条{MASK}评论：{textA}')
    # print('\nhard_template.inputs_list-->',hard_template.inputs_list)
    # print('\nhard_template.custom_tokens-->',hard_template.custom_tokens)
    tep = hard_template(
        inputs_dict={'textA': '包装不错，苹果挺甜的，个头也大。', 'MASK': '[MASK]'},
        tokenizer=tokenizer,
        max_seq_len=30,
        max_length=2)
    # print('\ntep-->',hard_template.custom_tokens)
    # print(tep)
    # print(tokenizer.convert_ids_to_tokens([3819, 3352]))
    # print(tokenizer.convert_tokens_to_ids(['水', '果']))
