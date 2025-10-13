from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset, Dataset, interleave_datasets, concatenate_datasets
import re


# Load and prep dataset
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

# uncomment middle messages for 1-shot prompting


def get_datasets(split="train") -> Dataset:
    data = load_dataset('gsm8k',
                        'main')[split]  # type: ignore
    data = data.map(lambda x: {  # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer']),
        'db_set': 'gsm8k'
    })  # type: ignore
    data = data.remove_columns(['question'])

    # two times more than other datasets
    data_qa = load_dataset(
        "PubMedQA", "pqa_artificial")[split]
    data_qa = data_qa.filter(lambda x: len(
        "\n".join(x['context']['contexts'])) < 1024)  # avoid long traces
    data_qa = data_qa.map(lambda x: {  # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "Given the scientific context below:\n" +
                "\n".join(x['context']['contexts']) +
                "\n\nAnswer the following question:\n" +
                x['question'] +
                " with 'yes', 'no' or 'maybe'. You need to carefully review the context and reason before answering."
            },
        ],
        'answer': x['final_decision'],
        'db_set': 'pubmedqa'
    })  # type: ignore
    data_qa = data_qa.remove_columns(
        ['pubid', 'question', 'context', 'long_answer', 'final_decision'])

    categories = ['Lab_Medicine', 'Wearables', 'Dermatology', 'Gastroenterology', 'Internal_Medicine', 'Oncology', 'Orthopedics', 'General_Surgery', 'Ophthalmology', 'Audiology', 'Head_Neck_Surgery', 'Elderly_Care', 'Pediatrics', 'Allergy_Immunology', 'Rheumatology', 'Pharmacy', 'Obstetrics_Gynecology', 'Microbiology', 'Dentistry', 'Physical_Medicine_and_Rehabilitation', 'Neurology', 'Psychiatry', 'Pathology', 'Genetics', 'Rare_Diseases', 'Hematology',
                  'Emergency', 'Endocrinology', 'Radiology', 'Cardiology', 'Pulmonology', 'Infectious_Diseases', 'Critical_Care', 'Pediatric_Surgery', 'Neuroscience', 'Epidemiology', 'Fitness_Sports', 'Health_Education', 'Health_Economics', 'Health_Entrepreneurship', 'Hospital_Management', 'Mental_Health', 'Nutrition', 'Palliative_Care', 'Preventive_Medicine', 'Public_Health', 'Social_Media_Addiction', 'Sleep', 'Supplements', 'Vaccination', 'Work_Health', 'Wellbeing']
    data_mc = concatenate_datasets(
        [load_dataset("Health_Benchmarks", i)[i] for i in categories])
    data_mc = data_mc.map(lambda x: {  # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "\n\nAnswer the following question:\n" +
                x['Questions'] +
                "\n With 'A', 'B', 'C' or 'D'. You need to carefully review the context and reason before answering."
            },
        ],
        'answer': x['Answers'],
        'db_set': 'med_mc'
    })  # type: ignore
    data_mc = data_mc.remove_columns(['Answers', 'Questions'])

    dataset = concatenate_datasets([data, data_qa, data_mc])
    return dataset


dataset = get_datasets()
dataset = dataset.shuffle(seed=42)
train_test_split = dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]
print(f"train size: {len(train_dataset)}, test size: {len(test_dataset)}")


# Reward functions
def correctness_reward_func(prompts, completions, answer, db_set, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}",
          f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    rewards = []
    for r, a, dt in zip(extracted_responses, answer, db_set):
        if dt == "gsm8k":
            if a in r:
                rewards.append(1.0)
            elif r == a:
                rewards.append(2.0)
            else:
                rewards.append(0.0)
        else:
            rewards.append(2.0 if r.lower() == a.strip().lower() else 0.0)
    return rewards


def int_reward_func(completions, db_set, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    rewards = []
    for r, dt in zip(extracted_responses, db_set):
        if dt == "gsm8k":
            rewards.append(0.5 if r.isdigit() else 0.0)
        elif dt == "pubmedqa":
            rewards.append(0.5 if (
                'yes' in r.lower() or 'no' in r.lower() or 'maybe' in r.lower()) else 0.0)
        else:
            rewards.append(0.5 if ('a' in r.lower() or 'b' in r.lower(
            ) or 'c' in r.lower() or 'd' in r.lower()) else 0.0)
    return rewards


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count


def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


training_args = GRPOConfig(output_dir="outputs", num_generations=2, per_device_train_batch_size=4)


# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    "../GRPO/Qwen2.5-3B-Instruct/",
    torch_dtype="auto",
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(
    "../GRPO/Qwen2.5-3B-Instruct/")


trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
    ],
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)
trainer.train()