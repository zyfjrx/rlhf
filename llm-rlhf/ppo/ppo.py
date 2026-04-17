from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding
import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
#%%
model_path = "/Users/zhangyf/llm/gpt2-chinese-cluecorpussmall"
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.eos_token = tokenizer.pad_token
REWARD_TOKEN_ID = tokenizer.eos_token_id
#%%
ds = load_dataset("csv", data_files="online_shopping_10_cats.csv")
ds_train = ds['train']

ds_train = ds_train.filter(lambda x: x["review"] != None and len(
    x["review"]) > 20 and len(x["review"]) < 1024)

print("数据集的数量：", len(ds_train))
#%%
def tokenize(batch):
    outputs = tokenizer(batch["review"])
    outputs["score"] = [0] * len(outputs["input_ids"])
    outputs["score_index"] = [0] * len(outputs["input_ids"])
    for i in range(len(outputs["input_ids"])):
        # 第 i 条数据的末尾添加一个 eos token，作为reward token
        outputs["input_ids"][i].append(REWARD_TOKEN_ID)
        # reward token的掩码设置为 1 。
        outputs["attention_mask"][i].append(1)
        # 正向情感的文本评分为 1 。负向情感的评分为 0 。
        outputs["score"][i] = float(batch["label"][i])
        # 对 reward token 进行评分，也就是评分的索引为 reward token 的索引。
        outputs["score_index"][i] = len(outputs["input_ids"][i]) - 1
    return outputs
#%%
map_kwargs = {
    "batched": True,
    "batch_size": 4,
    "remove_columns": ["cat", "label", "review"]
}
tokenized_dataset_train = ds_train.map(tokenize, **map_kwargs)
tokenized_dataset_train.set_format(type="torch")