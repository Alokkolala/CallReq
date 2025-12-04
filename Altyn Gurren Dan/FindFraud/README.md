# **FindFraud**
Пайплайн для обнаружения аномалий в транзакциях, сочетающий машинное обучение и настраиваемые правила с объяснимостью и отчетностью.

## **Структура проекта**

```
src/findfraud/
├── data_loader.py    
├── features.py       
├── model.py          
├── graph_builder.py  
├── graph_model.py    
├── rules.py          
├── scorer.py         
├── report.py         
└── cli.py            
```

## **Как работает FindFraud**

* **Инжекция данных**: Валидация CSV-файлов с использованием схемы в стиле PaySim для обеспечения согласованности колонок и типов.
* **Пайплайн признаков**: Для табличных данных используются скользящие подсчеты, делты и кодированные категории; для графов строятся связи между аккаунтами.
* **Модели**: Используются **IsolationForest** или **GraphSAGE** для обнаружения аномалий.
* **Правила**: Включают бизнес-логику, такую как большие переводы, быстрые всплески, разрывы в балансе и т. д.
* **Объединение оценок**: Результаты модели и срабатывающие правила комбинируются в одну оценку мошенничества с объяснениями.
* **Отчетность**: Генерирует подробные отчеты в HTML/PDF, включая метаданные модели, подозрительные транзакции и профили.
* **Развертывание**: Работает как через CLI, так и через FastAPI, поддерживая как пакетную, так и реальную оценку.

## **Использование**

### **Обучение модели**

Для обучения модели **IsolationForest**:

```powershell
py -m findfraud.cli train data\transactions.csv models\anomaly.joblib
```

Для обучения модели **GraphSAGE**:

```powershell
py -m findfraud.cli train data\transactions.csv models\gnn.pt --model-type gnn --graph-artifacts outputs\graph.pt --window-size 24 --min-edge-count 2 --gnn-hidden 64 --gnn-layers 2 --gnn-epochs 50
```

### **Оценка новых транзакций**

Для оценки новых транзакций:

```powershell
py -m findfraud.cli score data\new_transactions.csv models\anomaly.joblib outputs\scores.csv --html-report outputs\report.html --pdf-report outputs\report.pdf --profiles-csv outputs\profiles.csv
```

Для **GraphSAGE** оценки:

```powershell
py -m findfraud.cli score data\new_transactions.csv models\gnn.pt outputs\graph_scores.csv --model-type gnn --graph-artifacts outputs\scored_graph.pt
```

### **Запуск модели как API**

Запустите сервер FastAPI:

```powershell
$env:FINDFRAUD_MODEL_PATH="models\anomaly.joblib"
py -m uvicorn findfraud.api:app --host 0.0.0.0 --port 8000
```

### **Конфигурация CORS для внешнего доступа**

```powershell
$env:FINDFRAUD_MODEL_PATH="models\anomaly.joblib"
$env:FINDFRAUD_CORS_ORIGINS="https://my-frontend.example.com,http://localhost:3000"
py -m uvicorn findfraud.api:app --host 0.0.0.0 --port 8000
```

## **Резюме**

* **FastAPI** предоставляет API для обнаружения мошенничества и отчетности.
* **Инструменты CLI** для обучения, оценки и генерации отчетов.
* Поддержка как **табличных**, так и **графовых** моделей для обнаружения аномалий.
* Гибкая настройка **CORS** для внешнего доступа.
