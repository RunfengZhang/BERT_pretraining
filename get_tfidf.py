import jieba.analyse
import datasets
import pandas as pd
from collections import Counter
import json

def extract_keywords():
    file_path = "data/algo_risk_user_comment_in_riskscene_0801.csv"
    idf_path='./tokenize/idf.txt'
    save_path='./data'
    csv_save_path='data/algo_risk_user_comment_in_riskscene_0801_keyword.csv'
    tokenized_datasets = datasets.load_dataset('csv', sep='\x01',error_bad_lines=False,
                                         data_files={'train': file_path},
                                         cache_dir='./cache')

    idf={}

    def get_tf(examples):
        tokens = jieba.lcut(examples['text'])
        tf = Counter(tokens)
        examples['tokens'] = tokens
        examples['tf'] = json.dumps(tf)
        return examples

    tokenized_datasets = tokenized_datasets.map(
        get_tf,
        batched=False,
        num_proc=50)


    def get_idf(examples):
        tf = json.loads(examples['tf'])
        for k in tf.keys():
            idf[k] = idf.get(k, 0) + 1

    tokenized_datasets=tokenized_datasets.map(
        get_idf,
        batched=False)

    with open(idf_path,'w') as f:
        for k,v in idf.items():
            if k.strip()!='':
                f.write(k+' '+str(v)+'\n')


    jieba.analyse.set_idf_path(idf_path)

    def get_keywords(examples):
        data = pd.DataFrame()

        keywords=jieba.analyse.extract_tags(examples['text'], topK=5, withWeight=True, allowPOS=())
        keyword_list=[]
        for k, v in keywords:
            data = data.append({'pid': int(examples['photo_id']), 'word': k, 'tfidf': v}, ignore_index=True)
            keyword_list.append(k)
        data.to_csv(csv_save_path, mode='a', header=None, index=False,sep='\t')

        examples['keywords']=keyword_list
        return examples


    tokenized_datasets = tokenized_datasets.map(
        get_keywords,
        batched=False,
        num_proc=50)


    tokenized_datasets.save_to_disk(save_path)

if __name__=='__main__':
    extract_keywords()