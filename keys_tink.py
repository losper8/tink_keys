import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelWithLMHead,
    TextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)
from torch.utils.tensorboard import SummaryWriter


def train_model():
    # Загрузка датасета из CSV-файла
    data_path = "sirius_test/data_1.csv"
    df = pd.read_csv(data_path)

    # Использование токенизатора и базовой модели
    tokenizer = AutoTokenizer.from_pretrained("tinkoff-ai/ruDialoGPT-medium")
    model = AutoModelWithLMHead.from_pretrained("tinkoff-ai/ruDialoGPT-medium")

    # Создание датасета с контекстами и ответами
    contexts = []
    for _, row in df.iterrows():
        context = {
            "context_3": row["context_3"],
            "context_2": row["context_2"],
            "context_1": row["context_1"],
            "response": row["response"],
        }
        contexts.append(context)

    # Создание файл датасета формата .txt
    with open("dialogs.txt", "w", encoding="utf-8") as f:
        for context in contexts:
            f.write(
                f"{context['context_3']}\t{context['context_2']}\t{context['context_1']}\t{context['response']}\n"
            )

    # Загрузка датасета в формате TextDataset
    train_dataset = TextDataset(
        tokenizer=tokenizer, file_path="dialogs.txt", block_size=128
    )

    # Определение параметров тренировки
    training_args = TrainingArguments(
        output_dir="./output_dir",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=-1,  # Отключение сохранения промежуточных моделей
        save_total_limit=0,  # Отключение сохранения моделей
        logging_steps=-1,  # Отключение логирования
        prediction_loss_only=True,
        report_to="tensorboard",  # Использование TensorBoard
    )

    # Создание датасета и загрузчика данных
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Задаем seed для воспроизводимости
    set_seed(42)

    # Перемещение модели на выбранное устройство
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Инициализация TensorBoard для логирования
    writer = SummaryWriter(log_dir="./logs")

    # Создание объекта Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    # Запуск обучения с логированием вручную
    for epoch in range(training_args.num_train_epochs):
        trainer.train()
        
        # Логирование метрик в TensorBoard
        losses = []
        for batch in trainer.get_train_dataloader():
            losses.append(batch.loss.item())
        avg_loss = sum(losses) / len(losses)
        writer.add_scalar("Training Loss", avg_loss, global_step=trainer.global_step)
            # Закрытие writer
    writer.close()


if __name__ == "__main__":
    # Задаем устройство для обучения (GPU, если доступен)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")

    # Запускаем обучение
    train_model()
