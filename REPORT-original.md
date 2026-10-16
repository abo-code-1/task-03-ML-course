# Task 3 Report: Bank Marketing

Imamsidikov, MOP 3231 Machine Learning (Python), Narxoz University

## 1. The task

The data is from a Portuguese bank that called clients and offered them a term deposit. The target y shows if the client said yes. I had to predict y for 8,236 clients in test.csv, using only info the bank has before the call. I used 2954 as random seed everywhere, so results are always the same.

## 2. The data

Train has 32,940 rows and 20 features, with no empty cells or duplicates. But some values are missing in a hidden way: six columns have "unknown", and pdays uses 999 for "never contacted".

Classes are very imbalanced, only 11.3% said yes. A model that always says "no" gets 88.7% accuracy but finds nobody, so I used average precision (AP) instead. Random guessing gives AP about 0.113.

I dropped duration (call length), because you only know it after the call, when the answer is already known. It's also not in test.csv. I dropped id too.

## 3. What I found

Economic indicators were the strongest features. When Euribor was under 1%, 46% of clients said yes, but when it was above 4% only 4 to 6% did. These columns basically show when the call happened.

Age has a U shape. Young clients (24 and younger) and old clients (over 60) said yes much more often then people from 30 to 60. Also the more times the bank called the same person, the lower the chance was, from 13% down to 6%.

The best categorical feature was the result of previous campaign. 65% of clients with a past success said yes again, but only 9% of new clients. Month and phone type also matter, mobile calls worked better then landline (15% vs 5%). The "unknown" values were actually useful, so I kept them as separate category instead of deleting 26% of the data.

## 4. Preparing the data

All preprocessing is in one ColumnTransformer inside a Pipeline. I changed pdays 999 to -1 and added a 0/1 column for it, scaled the numbers with StandardScaler and used one-hot encoding for categories. This way nothing leaks between CV folds. I split train into 80% development set (5-fold CV) and 20% hold-out that I didn't touch until the end.

## 5. Models and tuning

The dummy model got AP 0.113, decision tree 0.428, logistic regression 0.457, and random forest and gradient boosting both got 0.476.

Class weights in logistic regression didn't improve AP, they just changed the balance between precision and recall. Boosting was about 6 times faster then random forest, so I tuned it. After random search over 40 settings it got AP 0.480. It beat logistic regression on all 5 folds, but was only a tiny bit better than default boosting, so honestly tuning didn't help much.

## 6. Threshold and results

The 0.5 threshold was too strict, so I picked 0.211 by best F1 score on out-of-fold predictions. On the hold-out set the model got AP 0.444 and ROC AUC 0.792, which is about 4 times better than guessing. It's a bit lower then in CV, because of noise and some optimism from tuning.

In practice the model would call 14% of clients, and 45% of those calls end with yes (random calls give only 11%). If the bank calls the top 20% by score, it reaches 63% of all subscribers.

## 7. Final model

I retrained on all of train.csv and saved it with joblib as Imamsidikov_task3_model.joblib. I wrapped it in FixedThresholdClassifier so predict() gives yes or no directly. I checked it in a seperate Python process and got the same predictions. It predicts yes for 14% of test clients. Results are in predictions.csv with columns id, y and p_yes.

## 8. Limitations and conclusion

The strongest features show when the call was made, not who the client is, so on a new campaign in different economy the model would probably work worse. Also F1 treats a wasted call and a missed client the same, but a real bank would use real costs.

Overall, even without duration we can predict subscribers much better than chance. Gradient boosting was the best, but logistic regression is close and easier to explain, so it would be a good simpler choice.
