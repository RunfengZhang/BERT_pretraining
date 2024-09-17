from datasets import load_from_disk
from tqdm import tqdm
import json
# path='./review_data/raw_tokenized_data'
# tokenized_datasets = load_from_disk(path)
# word_freq={'total_word_freq':0}
#
# def count_freq(example):
#     for id in example['input_ids']:
#         word_freq[id]=word_freq.get(id,0)+1
#         word_freq['total_word_freq']+=1
#
# # for sentence in tqdm(tokenized_datasets['train']['input_ids']):
# #     for id in sentence:
# #         word_freq[id]=word_freq.get(id,0)+1
# #         word_freq['total_word_freq']+=1
#
# tokenized_datasets.map(count_freq,num_proc=100)
#
# with open('./review_data/raw_tokenized_data/word_freq.json','w') as f:
#     json.dump(word_freq,f)
#
#
# # #-
# # with open('./asr-data-0628-0704/tokenized_data/word_freq.json','r') as f:
# #     word_freq=json.load(f)
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
# with open('./review_data/raw_tokenized_data/invert_freq.json','w') as f:
#     json.dump(invert_freq,f)


#-
with open('./review_data/raw_tokenized_data/invert_freq.json','r') as f:
    invert_freq=json.load(f)

path='./review_data/raw_tokenized_data'
tokenized_datasets = load_from_disk(path)
print("load finish")

def count_inver_freq(example):
        word_ratio=[]
        for id in example['input_ids']:
            word_ratio.append(invert_freq[str(id)])
        example['sample_ratio']=word_ratio
        return example

print("start map")
tokenized_datasets=tokenized_datasets.map(count_inver_freq,num_proc=100)

print("save file")
tokenized_datasets.save_to_disk('./review_data/data_with_freq')