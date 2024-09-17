import torch
import datasets
from transformers import AutoTokenizer, AutoConfig,DataCollatorWithPadding
from transformers import BertPreTrainedModel,BertModel
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy

class BertForEmbedding(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        embedding = torch.sum(outputs[0]*torch.unsqueeze(attention_mask,-1),1)/torch.sum(attention_mask,1,keepdim=True)

        return embedding


# model_path='/home/web_server/antispam/project/wangkai/simcse-master/result/my-unsup-simcse-bert-base-uncased'
# model_path='bert-base-chinese'
# model_path='hfl/chinese-roberta-wwm-ext'
# model_path='/home/web_server/antispam/project/wangkai/review-pretrain/output-sample'
model_path = '/home/web_server/antispam/project/wangkai/review-pretrain/output-sample/checkpoint-20000'


# chenggang
# def build_model():
#     config = AutoConfig.from_pretrained('bert-base-chinese')
#     config.max_position_embeddings = 128
#     config.type_vocab_size = 1
#     config.num_labels = 2
#     model = BertForSequenceClassification(config)
#     checkpoint_path = '/home/web_server/antispam/project/zhouyalin/learn/huggingface/model_transfer/models/chengang_roberta.pth'
#     model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=False)
#     return model

def main():
    config = AutoConfig.from_pretrained(model_path)
    model = BertForEmbedding.from_pretrained(model_path, config=config)
    train_file = "/home/web_server/antispam/project/wangkai/multimodal-review-pretrain/labeled-data/train_data.csv"

    raw_datasets = datasets.load_dataset('csv', sep='\t',
                                         data_files={'train': train_file},
                                         cache_dir='./cache')

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    def tokenize_function(examples):
        return tokenizer(examples["text"], max_length=512, padding="max_length", truncation=True)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, num_proc=5, load_from_cache_file=True)
    tokenized_datasets=tokenized_datasets.remove_columns(['text','user_id'])
    tokenized_datasets.set_format("torch")


    train_dataloader = DataLoader(tokenized_datasets["train"],
                                  batch_size=32, shuffle=False)

    device=torch.device('cuda')
    model.to(device)


    all_embeddings=[]
    for step, batch in enumerate(tqdm(train_dataloader)):
        for k, v in batch.items():
            batch[k] = v.to(device)
        embeddings = model(**batch)

        all_embeddings.extend(embeddings.tolist())

    all_embeddings = numpy.array(all_embeddings)
    save_path = ''
    numpy.save(save_path, all_embeddings)



