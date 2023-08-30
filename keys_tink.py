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
        num_train_epochs=10,  # Увеличение количества эпох
        per_device_train_batch_size=2,  # Уменьшение размера пакета обучения
        save_steps=500,  # Увеличение частоты сохранения промежуточных результатов
        save_total_limit=5,  # Сохранение только последних 5 промежуточных результатов
        logging_steps=100,  # Логирование каждые 100 шагов
        evaluation_strategy="epoch",
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

    # Запуск обучения с логированием в TensorBoard и сохранением промежуточных результатов
    trainer.train()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    # Закрытие writer
    writer.close()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")

    train_model()
