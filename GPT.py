from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset

def preprocess(example): 
    example['prompt'] = f"{example['instruction']} {example['input']} {example['output']}"
    return example

def tokenize_dataset(dataset): 
    tokenized_dataset = dataset.map(lambda example: tokenizer(example['prompt'], truncation=True, max_length=128), batched=True, remove_columns=['prompt'])
    return tokenized_dataset

dataset = load_dataset("hakurei/open-instruct-v1", split='train')
dataset.to_pandas().sample(20)

dataset = dataset.map(preprocess, remove_columns=['instruction', 'input', 'output'])

dataset =  dataset.shuffle(42).select(range(100)).train_test_split(test_size=0.1)

train_dataset = dataset['train']
test_dataset = dataset['test']

MODEL_NAME = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

train_dataset = tokenize_dataset(train_dataset)
test_dataset = tokenize_dataset(test_dataset)

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./DialoGPT-gpt",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16
)

trainer = Trainer(
    model=model, 
    args= training_args, 
    train_dataset=train_dataset, 
    eval_dataset=test_dataset, 
    data_collator=data_collator
)

trainer.train()
trainer.save_model()

model = AutoModelForCausalLM.from_pretrained("./DialoGPT-gpt")
prompt = ''

def generate_text(prompt):
    inputs = tokenizer.encode(prompt, return_tensors='pt') #.to("cuda") # <-- if running on GPU, uncomment this
    outputs = model.generate(inputs, max_length=64, pad_token_id=tokenizer.eos_token_id)
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated[:generated.rfind('.')+1]

print(generate_text("What's the best way to cook a chicken breast?"))