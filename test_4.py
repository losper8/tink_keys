import torch 
print(torch.cuda.is_available())
import pandas as pd
import torch
import transformers
print(f"Версия pandas: {pd.__version__}")
print(f"Версия torch: {torch.__version__}")
print(f"Версия transformers: {transformers.__version__}")