import torch
import datasets
from transformers import AutoTokenizer, AutoConfig
from datasets import load_from_disk
from rcalgo_torch.training import Trainer, get_training_parser
from modeling_bert_pinyin import BertForSequenceClassification
import json

import torchmetrics
import torch.nn.functional as F
model_path = '/home/web_server/antispam/project/wangkai/review-pretrain/output-pinyin'


def load_dataset_female():
    train_file = "/home/web_server/antispam/project/caominghui/photo_comment/female_vulgar_comment/data/Leakage_data/train_datasets/train_19w_all.csv"
    validation_file = "/home/web_server/antispam/project/caominghui/photo_comment/female_vulgar_comment/data/Leakage_data/test_datasets/female_vulgar_test_data_20220209.csv"

    raw_datasets = datasets.load_dataset('csv', error_bad_lines=False,
                                         data_files={'train': train_file,'validation':validation_file},
                                         cache_dir='./cache')

    raw_datasets = raw_datasets.rename_column('label', 'labels')

    rm_col=list(set(raw_datasets['train'].column_names)-set(['text','labels']))
    raw_datasets= raw_datasets.remove_columns(rm_col)


    tokenizer = AutoTokenizer.from_pretrained(model_path)

    with open('./pinyin_files/wordidx2pinidx.json') as fin:
        id2pinyin = json.load(fin)

    def tokenize_function(examples):
         e=tokenizer(examples["text"], max_length=128, padding="max_length", truncation=True)
         p_id = []
         for id in e["input_ids"]:
             p_id.append(id2pinyin[str(id)])
         e["pinyin_ids"]=p_id
         return e


    tokenized_datasets = raw_datasets.map(tokenize_function, batched=False, num_proc=10, load_from_cache_file=True)
    tokenized_datasets=tokenized_datasets.remove_columns(['text'])


    tokenized_datasets.set_format("torch")

    return tokenized_datasets["train"], tokenized_datasets["validation"]

def load_dataset_politic():
    train_file = "/home/web_server/antispam/project/wangkai/data/politic/train_data.csv"
    validation_file = "/home/web_server/antispam/project/wangkai/data/politic/test_data.csv"

    raw_datasets = datasets.load_dataset('csv', error_bad_lines=False,
                                         data_files={'train': train_file,'validation':validation_file},
                                         cache_dir='./cache')

#     raw_datasets = raw_datasets.rename_column('label', 'labels')

    rm_col=list(set(raw_datasets['train'].column_names)-set(['text','labels']))
    raw_datasets= raw_datasets.remove_columns(rm_col)


    tokenizer = AutoTokenizer.from_pretrained(model_path)

    with open('./pinyin_files/wordidx2pinidx.json') as fin:
        id2pinyin = json.load(fin)

    def tokenize_function(examples):
         e=tokenizer(examples["text"], max_length=128, padding="max_length", truncation=True)
         p_id = []
         for id in e["input_ids"]:
             p_id.append(id2pinyin[str(id)])
         e["pinyin_ids"]=p_id
         return e


    tokenized_datasets = raw_datasets.map(tokenize_function, batched=False, num_proc=10, load_from_cache_file=True)
    tokenized_datasets=tokenized_datasets.remove_columns(['text'])


    tokenized_datasets.set_format("torch")

    return tokenized_datasets["train"], tokenized_datasets["validation"]
def build_model():
    config = AutoConfig.from_pretrained(model_path)
    config.num_labels = 2
    model = BertForSequenceClassification(config)
    checkpoint_path = '/home/web_server/antispam/project/wangkai/review-pretrain/output-pinyin/checkpoint-1000000/pytorch_model.bin'
    msg=model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=False)
    print(msg)
    return model


class AUROC(torchmetrics.AUROC):
    def update(self, output, batch):
        probabilities = F.softmax(output['logits'], dim=-1)
        super().update(probabilities, batch['labels'])

class Precision(torchmetrics.Precision):
    def update(self, output, batch):
        preds = output["logits"].argmax(dim=-1)
        super().update(preds, batch['labels'])

class Recall(torchmetrics.Recall):
    def update(self, output, batch):
        preds = output["logits"].argmax(dim=-1)
        super().update(preds, batch['labels'])

class F1(torchmetrics.F1):
    def update(self, output, batch):
        preds = output["logits"].argmax(dim=-1)
        super().update(preds, batch['labels'])

class Accuracy(torchmetrics.Accuracy):
    def update(self, output, batch):
        probabilities = F.softmax(output["logits"], dim=-1)
        super().update(probabilities[:, 1], batch['labels'])

class AveragePrecision(torchmetrics.AveragePrecision):
    def update(self, output, batch):
        probabilities = F.softmax(output["logits"], dim=-1)
        super().update(probabilities[:, 1], batch['labels'])
def train(args):
    train_dataset, eval_dataset = load_dataset_politic()

    model = build_model()
    #     freeze_layer_params(model, 3)

    metrics = torchmetrics.MetricCollection({
        "acc": Accuracy(),
        "precision": Precision(compute_on_step=False, num_classes=2,
                               ignore_index=0),
        "recall": Recall(compute_on_step=False, num_classes=2, ignore_index=0),
        "f1": F1(compute_on_step=False, num_classes=2, ignore_index=0),
        "auc": AUROC(compute_on_step=False, num_classes=2),
        "ap":AveragePrecision(compute_on_step=False)
    })

    trainer = Trainer(args, model, optimizer="adamw", metric=metrics)
    trainer.train(train_dataset, eval_dataset)

if __name__ == '__main__':
    parser = get_training_parser()
    args = parser.parse_args()
    train(args)

