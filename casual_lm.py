# -*- coding: utf-8 -*-
from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import torch.nn.functional as F



model_name_or_path='uer/gpt2-chinese-cluecorpussmall'
config=AutoConfig.from_pretrained(model_name_or_path)
# print(config)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,config=config)
tokenizer=AutoTokenizer.from_pretrained(model_name_or_path)

max_length=16

# def tokenize_function(texts):
#     return tokenizer(texts, max_length=max_length,padding='max_length',
#                       truncation=True,
#                      return_tensors="pt")

# # file_path='gpt2-chinese/tmp.txt'
# # with open(file_path,'r') as f:
# #     texts=f.readlines()
# texts=['谭念寒珸鏪39','石寻雪腃梉78','12345678','你好','how are you','给榜样点个赞']
# model.to('cuda')
# model.eval()


# nlls = []

# with torch.no_grad():
#     for i,t in enumerate(texts):
#         texts[i]=t.strip()
#         t=texts[i]
#         inputs=tokenizer(t, max_length=max_length,
#                       truncation=True,
#                      return_tensors="pt")
#         inputs['labels']=inputs['input_ids'].clone()
#         inputs = {k: v.to('cuda') for k, v in inputs.items()}

#         output = model(**inputs)
#         nlls.append(output.loss)

# ppl = torch.exp(torch.stack(nlls))
# results={t:v for t,v in zip(texts,ppl.tolist())}
# print(results)



def tokenize_function(texts):
    return tokenizer(texts, max_length=max_length,padding='max_length',
                      truncation=True,
                     return_tensors="pt")

file_path='gpt2-chinese/tmp.txt'
# with open(file_path,'r') as f:
#     texts=f.readlines()
texts=['谭念寒珸鏪39','石寻雪腃梉78','12345678','你好','how are you','给榜样点个赞']
model.to('cuda')
model.eval()


ppls = []
dataloader=DataLoader(texts, batch_size=32, num_workers=16, collate_fn=tokenize_function)

with torch.no_grad():
     for inputs in tqdm(dataloader):
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
        print(inputs['input_ids'])
        output = model(**inputs)
        logits=output.logits
        labels=inputs['input_ids'].clone()
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss=F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1),reduction='none')

        loss=loss.view(logits.shape[0],-1)
        att_mask=inputs['attention_mask'][..., 1:].contiguous()
        loss=torch.sum(loss*att_mask,1)/torch.sum(att_mask,1)
        ppl=torch.exp(loss)
        ppls.extend(ppl.tolist())

# ppl = torch.exp(torch.stack(nlls))
# print(ppl.shape)
# print(ppl)
results={t:v for t,v in zip(texts,ppls)}
print(results)
