
import torch
import datasets
from transformers import AutoTokenizer, AutoConfig
from datasets import load_from_disk
from rcalgo_torch.training import Trainer, get_training_parser
from transformers import BertForSequenceClassification

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score,average_precision_score
from rcalgo_torch.training import Trainer, get_training_parser
import numpy
import pandas as pd

checkpoint_path = 'ckpt/politic/checkpoint-470/state_dict.pt'
# checkpoint_path = 'ckpt/search_word_except_subtag_0525/checkpoint-7450/state_dict.pt'


threshold = 0.1
model_path = '/home/web_server/antispam/project/wangkai/review-pretrain/output-sample'

badcase_path = './badcase-politic-sequence-pretrain.json'


def load_dataset():
    test_file = "/home/web_server/antispam/project/wangkai/user_embedding/data/politic/politic_review_random5w_labeled_model_scoreV4.csv"
    test_data = pd.read_csv(test_file, sep='\x01')
    raw_datasets = datasets.Dataset.from_pandas(test_data)
    rm_col=list(set(raw_datasets.column_names)-set(['comment']))
    raw_datasets = raw_datasets.remove_columns(rm_col)


    tokenizer = AutoTokenizer.from_pretrained(model_path)

    def tokenize_function(examples):
        return tokenizer(examples["comment"], max_length=128, padding="max_length", truncation=True)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, num_proc=10, load_from_cache_file=True)
    tokenized_datasets=tokenized_datasets.remove_columns(['comment'])
    tokenized_datasets.set_format("torch")

    return tokenized_datasets, test_data


def build_model():
    config = AutoConfig.from_pretrained(model_path)
    config.max_position_embeddings = 128
    config.type_vocab_size = 1
    config.num_labels = 2
    model = BertForSequenceClassification(config)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=False)
    return model

def calc_metrics(probs, labels, threshold=0.5):
    preds = (probs > threshold)
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    auc = roc_auc_score(labels, probs)
    ap=average_precision_score(labels,probs)
    print("positive num: %s" % sum(preds))
    print(f"accuracy: {accuracy}\nprecision: {precision}\nrecall: {recall}"
          f"\nf1_score: {f1}\nauc: {auc}\nap: {ap}\n")


def save_bad_case(probs, labels, raw_data, threshold=0.5):
    preds = (probs > threshold)
    raw_data = raw_data.add_column('pred', probs.tolist())
    bad_case_idx = preds != labels
    idx = numpy.arange(len(labels))
    bad_case_idx = idx[bad_case_idx]
    bad_case = raw_data[bad_case_idx]
    with open(badcase_path, 'w') as f:
        json.dump(bad_case, f)

def save_result(probs, labels, raw_data, threshold=0.5):
    raw_data = raw_data.add_column('pred', probs.tolist())
    with open(badcase_path, 'w') as f:
        json.dump(raw_data[:], f)


def train(args):
    model = build_model()
    
    eval_dataset, raw_data = load_dataset()


    trainer = Trainer(args, model)
    output = trainer.predict(eval_dataset, output_keys=["logits"])

    probs = torch.nn.functional.softmax(output['logits'])

    raw_data['probs']=probs[:,1].tolist()

    save_path= "politic_data/result.csv"

    raw_data.to_csv(save_path,sep='\x01',index=False)



if __name__ == '__main__':
    parser = get_training_parser()
    args = parser.parse_args()
    args.per_device_eval_batch_size=256

    train(args)