# Task 3 Report: Bank Marketing

Imamsidikov, MOP 3231 Machine Learning (Python), Narxoz University

## 1. The task

The data is from a Portuguese bank that called clients and offered them a term deposit. The target y shows if the client said yes. I had to predict y for 8,236 clients in test.csv, using only info the bank has before the call. I used 2954 as random seed everywhere, so results are always the same.

## 2. The data

Train has 32,940 rows and 20 features, with no empty cells or duplicates. But some values are hidden missing: six columns have "unknown", and pdays uses 999 for "never contacted".

Classes are very imbalanced, only 11.3% said yes. A model that always says "no" gets 88.7% accuracy but finds nobody, so I used average precision (AP) instead. Random guessing gives AP about 0.113.

I dropped duration (call length), because you only know it after the call, when the answer is already known. It's also not in test.csv. I dropped id too.

## 3. What I found

- **Economic indicators** were the strongest. When Euribor was under 1%, 46% said yes, above 4% only 4–6%. These columns basically show *when* the call happened.
- **Age** has a U shape: young (≤24) and old (60+) clients said yes much more often then people 30–60.
- **More calls** to the same person means lower chance, from 13% to 6%.
- **Previous campaign success** was the best categorical feature: 65% said yes again vs 9% for new clients.
- **Month** and **phone type** matter too (mobile 15% vs landline 5%).
- **"Unknown"** values were useful, so I kept them as separate category instead of deleting 26% of the data.

## 4. Preparing the data

All preprocessing is in one ColumnTransformer inside a Pipeline: pdays 999 → -1 plus a 0/1 flag, StandardScaler for numbers, one-hot for categories. This way nothing leaks between CV folds. I split train into 80% development (5-fold CV) and 20% hold-out that I didn't touch until the end.

## 5. Models and tuning

| Model | AP |
|---|---|
| Dummy | 0.113 |
| Decision tree | 0.428 |
| Logistic regression | 0.457 |
| Random forest | 0.476 |
| Gradient boosting | 0.476 |

Class weights in logistic regression didn't improve AP, they just moved the precision/recall balance. Boosting was 6× faster then random forest, so I tuned it. After random search over 40 settings it got AP 0.480. It beat logistic regression on all 5 folds, but was only a tiny bit better than default boosting, so tuning didn't help much honestly.

## 6. Threshold and results

0.5 threshold was too strict, so I picked 0.211 by best F1 on out-of-fold predictions. On the hold-out set the model got **AP 0.444** and ROC AUC 0.792, which is about 4 times better than guessing. It's a bit lower then CV, because of noise and some optimism from tuning.

In practice: the model would call 14% of clients, and 45% of those calls end with yes (vs 11% random). Calling the top 20% by score reaches 63% of all subscribers.

## 7. Final model

I retrained on all of train.csv and saved it with joblib as Imamsidikov_task3_model.joblib, wrapped in FixedThresholdClassifier so predict() gives yes/no directly. I checked it in a seperate Python process and got the same predictions. It predicts yes for 14% of test clients. Results are in predictions.csv (id, y, p_yes).

## 8. Limitations and conclusion

The strongest features show *when* the call was made, not *who* the client is, so on a new campaign in different economy the model would probably work worse. F1 also treats a wasted call and a missed client the same, but a real bank would use real costs.

Overall, even without duration we can predict subscribers much better than chance. Gradient boosting was best, but logistic regression is close and easier to explain, so it would be a good simpler choice.
