from qwen2_model import Transformer
import torch
from pathlib import Path
from tokenizer import Tokenizer
from countdown_task import CountdownTasksDataset, reward_function
from grpo import rollout, normalize_rewards_per_group, update_policy
from torch.utils.data import DataLoader
from pprint import pprint

t = Tokenizer("/Users/zhangyf/llm/Qwen2.5-0.5B/tokenizer.json")
c = CountdownTasksDataset(tokenizer=t, data_path="Countdown-Tasks-3to4")
pprint(c.encode_prefix(numbers=[1, 2, 3], target=6)["prefix"])
print(c.encode_prefix(numbers=[1, 2, 3], target=6)["prefix_tokens"])
print(c.encode_prefix(numbers=[1, 2, 3], target=6)["prefix_token_ids"])

response = """
<think>我认为应该。。。</think>
<answer>1+2+3</answer>
"""

pprint(reward_function(response=response, numbers=[1, 2, 3], target=6))

pretrained_model_path = Path("/Users/zhangyf/llm/Qwen2.5-0.5B")
device = torch.device("mps")
dtype = torch.bfloat16

torch.set_default_device(device)
torch.random.manual_seed(1337)
BATCH_SIZE = 4  # 一批数据4条
NUM_QUESTIONS_PER_BATCH = 2  # 每批数据2个问题
NUM_ANSWERS_PER_QUESTION = BATCH_SIZE // NUM_QUESTIONS_PER_BATCH

train_dataset = CountdownTasksDataset(
    data_path="Countdown-Tasks-3to4",
    tokenizer=t,
    split="train",
    test_size=128,
)
generator = torch.Generator(device=device)
train_dataloader = DataLoader(
    train_dataset,
    shuffle=True,
    collate_fn=CountdownTasksDataset.collate_fn,
    generator=generator,
    batch_size=NUM_QUESTIONS_PER_BATCH,
)

print(train_dataloader)
b = next(iter(train_dataloader))
print(b.prefix[0])
print(len(b.prefix))

model = Transformer.from_pretrained(
    "/Users/zhangyf/llm/Qwen2.5-0.5B", device=device).train()

episodes = rollout(
    model=model,
    tokenizer=t,
    batch=b,
    max_gen_len=1024,
    num_answer_per_question=2,
    reward_function=reward_function,
    device=device,
    dtype=dtype,
)
print("\n=========采集轨迹===========")
for episode in episodes:
    print("=============prefix start=============")
    pprint(episode.prefix)
    print("=============prefix end===============")
    print("===============text=======================")
    pprint(episode.text)
    print("===============text=======================")
    print("===============text=======================")
    print("===============text=======================")
    print("===============text=======================")
    print("===============text=======================")
episodes = normalize_rewards_per_group(episodes)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1.0e-5,
    weight_decay=0.0,
    betas=[0.9, 0.999],
)
results = update_policy(
    model=model,
    optimizer=optimizer,
    episodes=episodes,
    micro_batch_size=2,  # 微批次大小为2
    pad_token_id=t.pad_token_id,
    max_grad_norm=1.0,  # 梯度裁剪到1.0
    device=device,
    dtype=dtype,
)
print(episodes)
