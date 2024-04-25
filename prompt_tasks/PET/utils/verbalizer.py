from typing import Optional, List
import os
from typing import Union, List
from pet_config import *

pc = ProjectConfig()


class Verbalizer():
    '''
    Verbalizer类,用于将一个Label对应到其子Label的映射
    '''

    def __init__(self, verbalizer_file: str, tokenizer, max_label_len: int):
        self.tokenizer = tokenizer
        self.label_dict = self.load_label_dict(verbalizer_file)
        self.max_label_len = max_label_len

    def load_label_dict(self, verbalizer_file: str):
        """
                读取本地文件，构建verbalizer字典。
                Args:
                    verbalizer_file (str): verbalizer文件存放地址。
                Returns:
                    dict -> {
                        '体育': ['篮球', '足球','网球', '排球',  ...],
                        '酒店': ['宾馆', '旅馆', '旅店', '酒店', ...],
                        ...
                    }
        """
        label_dict = {}
        with open(verbalizer_file, 'r', encoding='utf8') as f:
            for line in f.readlines():
                label, sub_labels = line.strip().split('\t')
                label_dict[label] = list(set(sub_labels.split(',')))
        return label_dict

    def find_sub_labels(self, label: Union[list, str]):
        if type(label) == list:
            while self.tokenizer.pad_token_id in label:
                label.remove(self.tokenizer.pad_token_id)
            label = ''.join(self.tokenizer.convert_ids_to_tokens(label))
        if label not in self.label_dict:
            raise ValueError(f'Lable Error:"{label}"not in label_dict')
        sub_labels = self.label_dict[label]
        ret = {'sub_labels': sub_labels}
        token_ids = [_id[1:-1] for _id in self.tokenizer(sub_labels)['input_ids']]
        for i in range(len(token_ids)):
            token_ids[i] = token_ids[i][:self.max_label_len]
            if len(token_ids[i]) < self.max_label_len:
                token_ids[i] = token_ids[i] + [self.tokenizer.pad_token_id] * (self.max_label_len - len(token_ids[i]))
        ret['token_ids'] = token_ids
        return ret

    def batch_find_sub_labels(self, label: List[Union[list, str]]):
        """
                批量找到子标签。

                Args:
                label (List[list, str]): 标签列表, [[4510, 5554], [860, 5509]] or ['体育', '电脑']

                Returns:
                    list -> [
                                {
                                 'sub_labels': ['足球', '网球'],
                                         'token_ids': [[6639, 4413], [5381, 4413]]
                                },
                                ...
                            ]
                """
        return [self.find_sub_labels(l) for l in label]

    def get_common_sub_str(self, str1: str, str2: str):
        '''
        寻找最大公共子串
        :param str1: abcd
        :param str2: abadbcdba
        '''
        lstr1, lstr2 = len(str1), len(str2),
        # 生成0句子,为方便后续计算,比字符串长度多了一列
        record = [[0 for i in range(lstr2 + 1)] for j in range(lstr1 + 1)]
        p = 0
        maxNum = 0
        for i in range(lstr1):
            for j in range(lstr2):
                if str1[i] == str2[j]:
                    record[i + 1][j + 1] = record[i][j] + 1
                    if record[i + 1][j + 1] > maxNum:
                        maxNum = record[i + 1][j + 1]
                        p = j + 1
        return str1[p - maxNum:p], maxNum

    def hard_mapping(self, sub_label: str):
        '''
        强匹配函数，当模型生成的子label不存在时，通过最大公共子串找到重合度最高的主label。
        :param sub_label:子label
        :return:str:主label
        '''
        label, max_overlap_str = '', 0
        for main_label, sub_labels in self.label_dict.items():
            overlap_num = 0
            for s_label in sub_labels:  # 求所有子label与当前推理label之间的最长公共子串长度
                overlap_num += self.get_common_sub_str(sub_label, s_label)[1]
            if overlap_num >= max_overlap_str:
                max_overlap_str = overlap_num
                label = main_label
        return label

    def find_main_label(self, sub_label: List[Union[list, str]], hard_mapping=True):
        if type(sub_label) == list:
            pad_token_id = self.tokenizer.pad_token_id
            while pad_token_id in sub_label:
                sub_label.remove(pad_token_id)
            sub_label = ''.join(self.tokenizer.convert_ids_to_tokens(sub_label))
        main_label = '无'
        for label, s_labels in self.label_dict.items():
            if sub_label in s_labels:
                main_label = label
                break
        if main_label == '无' and hard_mapping:
            main_label = self.hard_mapping(sub_label)
        ret = {
            'label': main_label,
            'token_ids': self.tokenizer(main_label)['input_ids'][1:-1],
        }
        return ret

    def batch_find_main_label(self, sub_label: List[Union[list, str]], hard_mapping=True):
        return [self.find_main_label(l, hard_mapping) for l in sub_label]


from transformers import AutoTokenizer

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(pc.pre_model)
    ver = Verbalizer(pc.verbalizer, tokenizer, pc.max_label_len)
    # print(ver.load_label_dict(pc.verbalizer))
    # print(ver.find_sub_labels('电脑'))
    print(tokenizer.pad_token_id)

