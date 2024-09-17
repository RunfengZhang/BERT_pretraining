# coding=utf-8
import argparse
import os
from typing import List

import torch
from transformers import BertForSequenceClassification,AutoModelForCausalLM
from transformers import AutoConfig, AutoTokenizer

from rcalgo_torch.torchscript.tokenizers import BERTPreprocessor
from rcalgo_torch.utils.logging import logger


class GPT2PPLExporter(torch.nn.Module):
    def __init__(self, preprocessor, model, max_length: int = 128):
        super().__init__()
        self.preprocessor = preprocessor
        self.model = model
        self.max_length = max_length

    def forward(self, texts: List[str]):
        input_ids, input_masks, segment_ids = self.preprocessor(texts, self.max_length)
        loss = self.model(input_ids.cuda(), input_masks.cuda(), segment_ids.cuda(),labels=input_ids.cuda()).loss
        ppl = torch.exp(loss)
        return ppl


def load_model(model_name_or_path):
    config = AutoConfig.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, config=config)
    return model


def load_tokenizer(model_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    return tokenizer


def trace_model(model, max_seq_length=128):
    device = torch.device("cuda")
    model.to(device)
    model.eval()

    logger.info("Use TorchScript mode")
    fake_input_id = torch.LongTensor(1, max_seq_length)
    fake_input_id.fill_(1)
    fake_input_id = fake_input_id.to(device)
    fake_mask = torch.ones(1, max_seq_length).to(device)
    fake_type_id = fake_input_id.clone().detach()
    fake_type_id.fill_(0)

    with torch.no_grad():
        traced_model = torch.jit.trace(
            model, (fake_input_id, fake_mask, fake_type_id))
    return traced_model


def sample_test(tokenizer, model, traced_model, max_seq_length=128):
    # inputs =['谭念寒珸鏪39','石寻雪腃梉78','12345678','你好','how are you','给榜样点个赞']
    inputs =['谭念寒珸鏪39']

    features = tokenizer(inputs, max_length=max_seq_length, padding="max_length",
                         truncation=True, return_tensors="pt")

    expected_output = model(input_ids=features['input_ids'].cuda(),
                            attention_mask=features['attention_mask'].cuda(),
                            token_type_ids=features['token_type_ids'].cuda(),labels=features['input_ids'].cuda())
    expected_output = torch.exp(expected_output.loss)

    output = traced_model(inputs)
    print(f"expected output: {expected_output}")
    print(f"traced model output: {output}")


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    # 1. create traced model
    model = load_model(args.model_name_or_path)
    traced_model = trace_model(model, args.max_seq_length)

    # 2. create preprocessor
    tokenizer = load_tokenizer(args.model_name_or_path)
    vocab = tokenizer.get_vocab()
    preprocessor = torch.jit.script(BERTPreprocessor(vocab))

    # 3. export
    m = torch.jit.script(GPT2PPLExporter(
        preprocessor, traced_model, max_length=args.max_seq_length))
    torch.jit.save(m, args.output_file)
    print(f"save torchscript model to {args.output_file}")

    # 4. sample test
    imported = torch.jit.load(args.output_file)
    sample_test(tokenizer, model, imported)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name_or_path",
        default='uer/gpt2-chinese-cluecorpussmall',
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--output_file",
        default="traced_model.pt",
        type=str,
        help="Path to export model"
    )

    args = parser.parse_args()
    main(args)
