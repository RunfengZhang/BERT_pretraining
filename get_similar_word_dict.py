import json

import synonyms
from datasets import load_dataset,Dataset

path='/home/web_server/anaconda3/lib/python3.6/site-packages/synonyms/data/vocab.txt'

data = load_dataset('text', data_files={"train":path}, cache_dir="./cache/")

def get_simliar_word(example):
    words=map(lambda x:x.split(' ')[0],example["text"])
    example["word"]=[]
    example["similar_word"]=[]
    for word in words:
        candidate = word
        similar_word_list, _ = synonyms.nearby(word, 5)
        similar_word_list = similar_word_list[1:]
        for w in similar_word_list:
            if '\u4e00' <= w <= '\u9fa5' and len(w) == len(word):
                candidate = w
                break
        example["word"].append(word)
        example["similar_word"].append(candidate)
    return example

# example=['1 2 3 ','4 5 6','1 2 3 ','4 5 6']
#
# data=Dataset.from_dict({"text":example})
data=data.map(
    get_simliar_word,
    batched=True,
    num_proc=100,
    batch_size=100)

data=data["train"]
words=list(data['word'])
similar_words=list(data['similar_word'])
word_similar_dict=dict(zip(words,similar_words))
with open('./similar_word_dict.json','w') as f:
    json.dump(word_similar_dict,f)
