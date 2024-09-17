from datasets import load_from_disk
from tqdm import tqdm
import json
# path='./review_data/chinese_tokenized_data'
# tokenized_datasets = load_from_disk(path)
# word_freq={'total_word_freq':0}
# #
# def count_freq(example):
#     for id in example['chinese_tokens']:
#         word_freq[id]=word_freq.get(id,0)+1
#         word_freq['total_word_freq']+=1
#
# # for sentence in tqdm(tokenized_datasets['train']['input_ids']):
# #     for id in sentence:
# #         word_freq[id]=word_freq.get(id,0)+1
# #         word_freq['total_word_freq']+=1
#
# tokenized_datasets.map(count_freq)
#
# with open('./review_data/chinese_tokenized_data/word_freq.json','w') as f:
#     json.dump(word_freq,f)
#
# vocab=["[PAD]","[UNK]","[MASK]"]+list(word_freq.keys())
# print(len(vocab))
# with open('./review_data/chinese_tokenized_data/vocab.txt','w') as f:
#     f.writelines(vocab)

#
#
# # #-
# with open('./review_data/chinese_tokenized_data/word_freq.json','r') as f:
#     word_freq=json.load(f)
#
# invert_freq={}
# total_num=word_freq['total_word_freq']
# base=0
# for id,freq in word_freq.items():
#     invert_freq[id]=total_num/freq
#     base+=invert_freq[id]
#
# for id in invert_freq.keys():
#     invert_freq[id]/=base
#
#
# with open('./review_data/chinese_tokenized_data/invert_freq.json','w') as f:
#     json.dump(invert_freq,f)


# #-
# from transformers import AutoTokenizer
# import itertools
# tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext',cache_dir="./cache")
# with open('./review_data/chinese_tokenized_data/invert_freq.json','r') as f:
#     invert_freq=json.load(f)
#
# path='./review_data/chinese_tokenized_data'
# tokenized_datasets = load_from_disk(path)
# print("load finish")
#
# def count_inver_freq(example):
#         ids=tokenizer(example['chinese_tokens'],add_special_tokens=False)['input_ids']
#         example['chinese_token_idx']=[]
#         input_ids = itertools.chain(*ids)
#         example['input_ids'] = list(input_ids)
#         word_ratio=[]
#         n=1
#         for word,word_id in zip(example['chinese_tokens'][1:-1],ids[1:-1]):
#             word_ratio.append(invert_freq.get(word,0))
#             l=len(word_id)
#             example['chinese_token_idx'].append(list(range(n,n+l)))
#             n+=l
#         example['sample_ratio']=word_ratio
#         return example
#
# print("start map")
# tokenized_datasets=tokenized_datasets.map(count_inver_freq,num_proc=100)
#

import torch
tokenized_datasets = load_from_disk('./review_data/chinese_tokenized_data/data_with_freq')

def get_mask(e):
    masked_indices=torch.zeros(len(e["input_ids"]),dtype=torch.bool)
    sample_indices = torch.multinomial(torch.tensor(e["sample_ratio"]),
                                       max(1, int(len(e["chinese_tokens"]) * 0.15)))
    for idx in sample_indices:
        masked_indices[e["chinese_token_idx"][idx]] = True

    e["masked_indices"]=masked_indices

    return e

tokenized_datasets=tokenized_datasets.map(get_mask,num_proc=76)

max_seq_length=512
def group_texts(examples):
    # Concatenate all texts.

    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    total_length = (total_length // max_seq_length) * max_seq_length
    # Split by chunks of max_len.
    result = {
        k: [t[i: i + max_seq_length] for i in range(0, total_length, max_seq_length)]
        for k, t in concatenated_examples.items()
    }
    return result

tokenized_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    num_proc=76,
    desc=f"Grouping texts in chunks of {max_seq_length}",
)
print("save file")
tokenized_datasets.save_to_disk('./review_data/chinese_tokenized_data/data_with_freq')



# def filter_fuc(example):
#     return sum(example['sample_ratio'])>0
# tokenized_datasets = load_from_disk('./review_data/chinese_tokenized_data/data_with_freq')
# print(tokenized_datasets)
# tokenized_datasets=tokenized_datasets.filter(filter_fuc,num_proc=76)
# print(tokenized_datasets)
# tokenized_datasets.save_to_disk('./review_data/chinese_tokenized_data/data_with_freq')

