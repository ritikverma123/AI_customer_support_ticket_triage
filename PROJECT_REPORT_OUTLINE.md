# MCA Project Report Outline
## Title
AI Customer Support Ticket Triage

## 1. Introduction
Customer-support teams receive many tickets. Manual classification can be slow. This project uses Natural Language Processing and Machine Learning to automatically classify a ticket by category and urgency.

## 2. Objectives
1. Read support-ticket text.
2. Predict ticket category.
3. Predict urgency.
4. Route tickets to the correct support queue.
5. Detect low-confidence predictions for human review.

## 3. Technologies
Python, Pandas, Scikit-learn, TF-IDF, Logistic Regression, Flask, HTML/CSS, Joblib.

## 4. System Flow
Ticket -> Text preprocessing -> TF-IDF -> Category model + Urgency model -> Confidence check -> Queue routing / Human review.

## 5. Dataset
Fields: ticket_text, category, urgency.
Categories: billing, technical, account, product.
Urgency: low, medium, high.

## 6. Machine Learning
TF-IDF converts text into numerical features. Logistic Regression learns the relationship between ticket features and labels.

## 7. Evaluation
Use precision, recall, weighted F1-score and confusion matrix. Use a held-out test set.

## 8. Deployment
Flask provides a browser dashboard and REST API.

## 9. Human-in-the-loop
Predictions with confidence below 0.60 are logged for human review.

## 10. Future Scope
Larger dataset, multilingual support, embeddings/transformers, database integration, authentication, analytics dashboard and continuous learning.

## 11. Conclusion
The project demonstrates an end-to-end NLP classification workflow from ticket input to automated queue routing with a safety mechanism for uncertain predictions.
