# simple-llm-docx-api

FastAPI сервис для ответов на вопросы по `.docx` через OpenRouter.

Сервис намеренно запускается одним процессом FastAPI. Фоновые задачи остаются внутри приложения, поэтому single-process режим здесь выбран осознанно.

## Запуск

```bash
cp .env.example .env
docker compose up --build
```

## API

- `POST /documents` -> `{"document_id":"<uuid>","status":"pending"}`
- `GET /documents/{document_id}` -> статус индексации документа
- `POST /questions` -> `{"question_id":"<uuid>"}`
- `GET /questions/{question_id}` -> статус или ответ

После перехода на OpenRouter embeddings старые документы нужно загрузить заново.

После возврата на SQLite старая Postgres-база не используется. При первом старте будет создан новый `/app/data/app.db`.

## Демо

Подставьте `document_id` из первой команды в команды с вопросами, а затем `question_id` из каждой команды вопроса в следующую команду получения ответа.

```bash
curl -s -X POST "http://localhost:8000/documents" -F "file=@./data/ADC_8.docx"
curl -s "http://localhost:8000/documents/<DOCUMENT_ID>"
curl -s -X POST "http://localhost:8000/questions" -H "Content-Type: application/json" -d '{"document_id":"<DOCUMENT_ID>","question":"Укажи предмет договора"}'
curl -s "http://localhost:8000/questions/<QUESTION_ID_1>"
curl -s -X POST "http://localhost:8000/questions" -H "Content-Type: application/json" -d '{"document_id":"<DOCUMENT_ID>","question":"Какой номер и дата у этого договора?"}'
curl -s "http://localhost:8000/questions/<QUESTION_ID_2>"
curl -s -X POST "http://localhost:8000/questions" -H "Content-Type: application/json" -d '{"document_id":"<DOCUMENT_ID>","question":"Какие штрафные санкции предусматривает этот договор в отношении поставщика?"}'
curl -s "http://localhost:8000/questions/<QUESTION_ID_3>"
curl -s -X POST "http://localhost:8000/questions" -H "Content-Type: application/json" -d '{"document_id":"<DOCUMENT_ID>","question":"Какие штрафные санкции предусматривает этот договор в отношении покупателя?"}'
curl -s "http://localhost:8000/questions/<QUESTION_ID_4>"
```
