from transformers import AutoTokenizer
tokenizer=AutoTokenizer.from_pretrained(r'F:\bert-base-chinese')
print(tokenizer('你好'))
