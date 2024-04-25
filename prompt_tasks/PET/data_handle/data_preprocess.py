from typing import Optional, List
from template import *
from rich import prompt
from datasets import load_dataset
from functools import partial
from PET import pet_config


def convert_example(
        examples: dict,
        tokenizer,
        max_seq_len: int,
        max_label_len: int,
        hard_template: HardTemplate,
        train_model=True,
        return_tensor=False
) -> dict:
    tokenized_output = {
        'input_ids': [],
        'token_type_ids': [],
        'attention_mask': [],
        'mask_positions': [],
        'mask_labels': []
    }
    # print("examples['text']-->", examples['text'])
    for i, example in enumerate(examples['text']):
        
        if train_model:
            label, content = example.strip().split('\t')
        else:
            content = example.strip()
        inputs_dict = {
            'textA': content, 'MASK': '[MASK]'
        }
        encoded_inputs = hard_template(
            inputs_dict=inputs_dict, tokenizer=tokenizer,
            max_seq_len=max_seq_len, max_length=max_label_len
        )
        tokenized_output['input_ids'].append(encoded_inputs['input_ids'])
        tokenized_output['token_type_ids'].append(encoded_inputs["token_type_ids"])
        tokenized_output['attention_mask'].append(encoded_inputs["attention_mask"])
        tokenized_output['mask_positions'].append(encoded_inputs["mask_position"])
        if train_model:
            label_encoded = tokenizer(text=[label])
            label_encoded = label_encoded['input_ids'][0][1:-1]
            label_encoded = label_encoded[:max_label_len]
            add_pad = [tokenizer.pad_token_id] * (max_label_len - len(label_encoded))
            label_encoded = label_encoded + add_pad
            tokenized_output['mask_labels'].append(label_encoded)
    for k, v in tokenized_output.items():
        if return_tensor:
            tokenized_output[k] = torch.LongTensor(v)
        else:
            tokenized_output[k] = np.array(v)
    return tokenized_output


if __name__ == '__main__':
    pc = ProjectConfig()
    train_dataset = load_dataset('text', data_files=pc.train_path)
    print('*' * 80)
    # for i,example in enumerate(train_dataset['train']):
    #     print('i-->',i)
    #     print('Example:', example)
    # print(type(train_dataset))
    # print(train_dataset)
    # print('*'*80)
    # print(train_dataset['train']['text'])
    tokenizer = AutoTokenizer.from_pretrained(pc.pre_model)
    hard_template = HardTemplate(prompt='这是一条{MASK}评论：{textA}')

    convert_func = partial(convert_example,
                           tokenizer=tokenizer,
                           hard_template=hard_template,
                           max_seq_len=30,
                           max_label_len=2)
    print('train_dataset-->',train_dataset)
    dataset = train_dataset.map(convert_func, batched=True)
    print('dataset-->', dataset)
    for value in dataset['train']:
        print(value)
        print(len(value['input_ids']))
        break

