import json
from pypinyin import pinyin, Style
from transformers import AutoTokenizer, AutoConfig


model_path = '/home/web_server/antispam/project/wangkai/review-pretrain/output-sample'

tokenizer = AutoTokenizer.from_pretrained(model_path)
    
vocab=tokenizer.get_vocab()    

# vocab={'你':1,'word':0,'##好':3,'s@是':4,'[unk]':5,'宁':7,'您':6}
vocab=sorted(vocab.items(),key=lambda x:x[1])

word2pinyin={}
pinyin2idx={}
wordidx2pinidx={}

for w,idx in vocab:
#     print(w)
    pinyin_list = pinyin(w, style=Style.NORMAL, heteronym=True,errors=lambda x: [['not chinese'] for _ in x])
    if not pinyin_list:
        continue
    p=pinyin_list[0][0]
    if not p or p=='not chinese':
        p=w
        word2pinyin[w]=w
    else:
        if 'ing' in p:
            p=p[:-1]
        word2pinyin[w]=p
    if p not in pinyin2idx.keys():
        pinyin2idx[p]=len(pinyin2idx)
    wordidx2pinidx[idx]=pinyin2idx[p]
    
# print(word2pinyin)
# print(pinyin2idx)
# print(wordidx2pinidx)

with open('./pinyin_files/word2pinyin.json','w') as fin:
    json.dump(word2pinyin,fin)
with open('./pinyin_files/pinyin2idx.json','w') as fin:
    json.dump(pinyin2idx,fin)
with open('./pinyin_files/wordidx2pinidx.json','w') as fin:
    json.dump(wordidx2pinidx,fin)


vocab=tokenizer.get_vocab()    
w='[PAD]'
print(vocab[w])
print(pinyin2idx[w])