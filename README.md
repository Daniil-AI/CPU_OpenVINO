# **OpenVINO Текстовая Классификация с MiniLM-L6-H384**

Этот Jupyter-ноутбук демонстрирует процесс работы с моделью классификации текста с использованием OpenVINO, включая конвертацию, бенчмаркинг, асинхронный инференс и квантизацию.

---
В качестве испытуемого выступает предобученная модель philschmid/MiniLM-L6-H384-uncased-sst2 с Hugging Face Hub:
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
model_id = "philschmid/MiniLM-L6-H384-uncased-sst2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
hf_model = AutoModelForSequenceClassification.from_pretrained(model_id)
```
Модель конвертируется в формат OpenVINO и сохраняется в model.xml. После чего идет сравнение точности оригинальной и OpenVINO-модели на датасете SST-2.

## Результаты
|| Метрика	Оригинал (FP32)                    |Квантизация (INT8)      |
|:----------------------------|:----------------------------|:--------------:|
|Точность (Accuracy)	|0.901	|0.895|
|FPS	|165.3|401.6|
|Задержка (latency)	|3.43 мс	|1.43 мс|


## Установка зависимостей
Для работы требуется:
- Python 3.8+
- Зависимости:
  ```bash
  pip install openvino nncf transformers[torch] datasets evaluate ipywidgets

