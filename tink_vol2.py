import torch
import wandb
from transformers import (
    AutoTokenizer,
    AutoModelWithLMHead,
    TextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)
from sklearn.model_selection import train_test_split
from datasets import Dataset


def train_model(dataset_path, output_dir, wandb_project):
    wandb.init(project=wandb_project)

    tokenizer = AutoTokenizer.from_pretrained("tinkoff-ai/ruDialoGPT-medium")
    model = AutoModelWithLMHead.from_pretrained("tinkoff-ai/ruDialoGPT-medium")

    with open(dataset_path, "r", encoding="utf-8") as file:
        texts = file.read().split("\n")

    train_texts, eval_texts = train_test_split(texts, test_size=0.1, shuffle=True, random_state=42)

    train_dataset = Dataset.from_dict({"text": train_texts})
    eval_dataset = Dataset.from_dict({"text": eval_texts})

    train_dataset = train_dataset.map(lambda examples: tokenizer(examples["text"], truncation=True, padding="max_length"), batched=True)
    eval_dataset = eval_dataset.map(lambda examples: tokenizer(examples["text"], truncation=True, padding="max_length"), batched=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=10,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        save_steps=500,
        save_total_limit=5,
        logging_steps=50,
        evaluation_strategy="epoch",
        report_to="wandb",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()

    wandb.finish()


if __name__ == "__main__":
    dataset_path = "data_1.csv"
    output_dir = "result"
    wandb_project = "tink_keys_vol2"

    train_model(dataset_path, output_dir, wandb_project)
