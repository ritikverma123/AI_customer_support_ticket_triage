# AI_customer_support_ticket_triage
ai_customer_support_ticket_triage
# AI Customer Support Ticket Triage

## Project objective
Build an AI system that reads incoming support tickets, predicts their **category** and **urgency**, and routes the ticket to the appropriate support queue.

### Categories
- Billing
- Technical
- Account
- Product

### Urgency
- Low
- Medium
- High

## How this project matches the 6 requested steps

1. **Collect historical ticket data**
   - `data/tickets.csv` is a 1,200-row balanced synthetic starter dataset.
   - For a real project, replace it with historical ticket text, category, urgency and resolution information.

2. **Clean text and create labels**
   - The training script removes missing rows.
   - Labels are category and urgency.

3. **Create text features**
   - TF-IDF with unigram + bigram features.

4. **Train models**
   - One Logistic Regression classifier predicts category.
   - A second Logistic Regression classifier predicts urgency.

5. **Evaluate**
   - The script prints precision, recall, F1-score and confusion matrices.
   - Metrics are also saved to `models/metrics.json`.

6. **API/dashboard + human review**
   - Flask dashboard at `/`.
   - JSON API at `/api/predict`.
   - Low-confidence predictions (< 0.60) are logged to `logs/low_confidence.csv`.

## Run the project

### 1. Create/activate a virtual environment
Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

Linux/macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the AI models
```bash
python train_model.py
```

### 4. Start the dashboard
```bash
python app.py
```

Open:
`http://127.0.0.1:5000`

## API example

POST JSON to `/api/predict`:
```json
{
  "ticket": "I was charged twice for my order"
}
```

The response contains:
- category
- urgency
- confidence
- support queue
- human_review flag

## Important academic note
The included 1,200-row dataset is a synthetic demonstration dataset. Its metrics should **not** be presented as production performance. For an MCA project, use a larger historical/realistic dataset and report train/test results honestly.

## Suggested future improvements
- Add more ticket categories.
- Compare Logistic Regression, Naive Bayes and Linear SVM.
- Use sentence embeddings/transformers.
- Add an admin dashboard with charts.
- Store tickets in MySQL/PostgreSQL.
- Add authentication.
- Add feedback from support agents and retrain periodically.


## Updated dataset
The starter dataset now contains **1,200 tickets**:
- 4 categories: billing, technical, account, product
- 3 urgency levels: low, medium, high
- 100 tickets for every category × urgency combination
- Unique ticket IDs
- More varied wording for better NLP training

The dataset is balanced for demonstration and academic testing. For a final real-world deployment, use an approved historical support-ticket dataset.
