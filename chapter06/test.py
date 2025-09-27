from transformers import AutoTokenizer
model_path = '/Users/zhangyf/llm/gpt2'
tokenizer = AutoTokenizer.from_pretrained(model_path)

from datasets import load_dataset
dataset_path = './sst2'
dataset = load_dataset(dataset_path)
print(dataset)

ds_train, ds_val = dataset['train'], dataset['validation']
print(ds_train[4])


# 定义一个新的token，奖励token eos
REWARD_TOKEN_ID = tokenizer.eos_token_id
print(REWARD_TOKEN_ID)

def tokenize(batch):
    # 提取出文本内容
    outputs = tokenizer(batch['sentence'])
    # 每条数据一个评分，初始化为 0 。
    outputs['score'] = [0] * len(outputs['input_ids'])
    # 对每条数据的最后的reward token进行评分
    outputs['score_index'] = [0] * len(outputs['input_ids'])
    for i in range(len(outputs['input_ids'])):
        # 第 i 条数据的末尾添加一个 eos token，作为reward token
        outputs['input_ids'][i].append(REWARD_TOKEN_ID)
        # reward token的掩码设置为 1 。
        outputs['attention_mask'][i].append(1)
        # 正向情感的文本评分为 1 。负向情感的评分为 0 。
        outputs['score'][i] = float(batch['label'][i])
        # 对 reward token 进行评分，也就是评分的索引为 reward token 的索引。
        outputs['score_index'][i] = len(outputs['input_ids'][i]) - 1
    return outputs

map_kwargs = {
    "batched": True,
    "batch_size": 512,
    "remove_columns": ['idx', 'sentence', 'label']
}

tokenized_dataset_train = ds_train.map(tokenize, **map_kwargs)
tokenized_dataset_val = ds_val.map(tokenize, **map_kwargs)

print(tokenized_dataset_train[4])

tokenized_dataset_train.set_format(type='torch')
tokenized_dataset_val.set_format(type='torch')

print(tokenized_dataset_train[4])

tokenized_dataset_train = tokenized_dataset_train.filter(lambda x: len(x['input_ids']) > 6)
tokenized_dataset_val = tokenized_dataset_val.filter(lambda x: len(x['input_ids']) > 6)

print(len(tokenized_dataset_train))

import torch
from torch import nn
import numpy as np
from transformers import AutoModelForCausalLM

class RewardHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # llm最后输出的隐藏层的维度
        self.hidden_size = config.hidden_size
        # 线性层用来对llm最后输出的隐藏层给奖励
        self.reward = nn.Linear(self.hidden_size, 1)
        self._post_init()

    def _post_init(self):
        # 使用正态分布初始化权重
        nn.init.normal_(
            self.reward.weight,
            std=(1.0 / np.sqrt(self.hidden_size + 1))
        )
        # 将偏置初始化为0
        nn.init.zeros_(self.reward.bias)

    def forward(self, hidden_states):
        # 给出奖励
        return self.reward(hidden_states)

class GPT2RewardHead(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.llm = AutoModelForCausalLM.from_pretrained(model_name)
        self.reward_head = RewardHead(self.llm.config)

    def forward(self, input_ids, attention_mask):
        transformer_outputs = self.llm.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        last_hidden_state = transformer_outputs.hidden_states[-1]
        # 给出奖励
        reward = self.reward_head(last_hidden_state).squeeze(-1)
        # sigmoid用来将奖励搞到(0,1)范围内
        return torch.sigmoid(reward)

model = GPT2RewardHead(model_path)


from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
# 还是将 eos token 作为 pad token
tokenizer.pad_token = tokenizer.eos_token

data_collator = DataCollatorWithPadding(tokenizer)
dataloader_params = {
    'batch_size': 32,
    'shuffle': True,
    'collate_fn': data_collator
}
train_dataloader = DataLoader(tokenized_dataset_train, **dataloader_params)
val_dataloader = DataLoader(tokenized_dataset_val, **dataloader_params)

batch = next(iter(train_dataloader))
print(batch.keys())

print(batch['input_ids'][1])
print(batch['attention_mask'][1])
print(batch['score'][1])
print(batch['score_index'][1])
print(tokenizer.decode(batch['input_ids'][1]))
print(batch['attention_mask'][1].nonzero()[-1])

outputs = model(batch['input_ids'], batch['attention_mask'])
print(outputs.shape)


device = torch.device('mps') if torch.mps.is_available() else torch.device('cpu')
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
# 二分类交叉熵损失
criterion = nn.BCELoss()
num_epochs = 1 # N+ Implementation Detail paper

def validate():
    model.eval()
    total_loss = 0
    for i, batch in enumerate(val_dataloader):
        inputs = batch.to(device)
        model_inputs = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask']
        }
        with torch.no_grad():
            # 对输出进行评分
            scores = model(**model_inputs)
            # 批次中每条数据的索引
            batch_indices = torch.arange(scores.shape[0])
            # 根据索引拿出评分，也就是reward token的评分
            score = scores[batch_indices, inputs['score_index']]
            # 目标评分，0 或者 1 。
            target = inputs['score']
            # 计算误差
            loss = criterion(score, target)
        total_loss += loss.item()
    print('validation loss:', total_loss / len(val_dataloader))
model.to(device)
validate()