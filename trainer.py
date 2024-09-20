from transformers import GPTNeoForCausalLM, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset

#initialize model and tokenizer
model = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-1.3B')
tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B')

#load the file
dataset = load_dataset('json', data_files={'train': './data/intents.json'})

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Define trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
)

# Train the model
trainer.train()
