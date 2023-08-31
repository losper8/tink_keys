import logging
from aiogram import Bot, Dispatcher, types, executor
import torch
from transformers import AutoTokenizer, AutoModelWithLMHead

# Включаем логирование
logging.basicConfig(level=logging.INFO)

# Создаем экземпляр бота и диспетчера
bot = Bot(token="6682894350:AAF3jPt8vMsBVaOS21EgQZTs0r6yDs0I_h8")
dp = Dispatcher(bot)

# Загрузка предобученной модели и токенизатора
model_path = r"output_dir"
tokenizer = AutoTokenizer.from_pretrained("tinkoff-ai/ruDialoGPT-medium")
model = AutoModelWithLMHead.from_pretrained(model_path)

# Переводим модель и токенизатор на CUDA, если доступно
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Функция для генерации ответа на входное сообщение
def generate_response(message):
    input_ids = tokenizer.encode(message, return_tensors="pt").to(device)
    outputs = model.generate(input_ids, max_length=100, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Обработчик команды /start
@dp.message_handler(commands=['start'])
async def start_command(message: types.Message):
    await message.reply("Привет! Я готов отвечать на твои вопросы.")

# Обработчик текстовых сообщений
@dp.message_handler(content_types=types.ContentTypes.TEXT)
async def handle_text(message: types.Message):
    response = generate_response(message.text)
    await message.reply(response)

# Запускаем бота
if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
