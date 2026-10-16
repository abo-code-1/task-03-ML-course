# Task 3 Report: Bank Marketing

Imamsidikov, MOP 3231 Machine Learning (Python), Narxoz University

## 1. The task

The data is from a Portuguese bank that was calling its clients and offering them a term deposit. For each call we have info about the client (age, job, education, loans), about the call (month, day, mobile or landline), about previous campaigns, and some economic indicators. The target y shows if the client said yes.

The business question is simple: before calling, which clients are worth calling? The call centre can only call 10% of the clients on the list, so the model has to put the clients who will most likely say yes at the top of the list, and then the bank calls the top 10%. My goal was to build this model and score the 8,236 clients in test.csv, using only information the bank has before the call. Everything is in the notebook Imamsidikov_task3.ipynb. I used 2954 as random seed everywhere, so the results are always the same.

## 2. First look at the data

The training file has 32,940 rows and 20 features. There are no empty cells or duplicates, but some data is missing in a hidden way: six columns have the value "unknown", and pdays uses 999 when the client was never contacted.

The classes are very imbalanced, only 11.3% of calls ended with yes. A model that always says "no" gets 88.7% accuracy but never finds anyone, so I compared models by average precision (AP) instead. AP shows how good the model is at putting yes clients at the top of the list, which is exactly what matters when you can call only 10%. Random guessing gives AP around 0.113.

I dropped duration (the call length), because you only know it after the call, when the answer is already known. It's also not in test.csv. I dropped id too because it's just a row number.

## 3. What the data showed

Economic indicators were the strongest features. When the Euribor rate was under 1%, 46% of calls were successful, but above 4% only 4 to 6% were. These columns basically tell the model when the call happened.

Age has kind of a U shape: young clients up to 24 and clients over 60 said yes much more often then people between 30 and 60. Also the more the bank called the same person, the lower the chance was, from 13% to 6%.

The result of previous campaign was the strongest categorical feature. 65% of clients with a past success said yes again, compared to 9% for new clients. Month matters too, May had the most calls but the lowest success rate (6%). Mobile calls worked better then landline (15% vs 5%).

About 26% of rows have at least one "unknown" value. They were actually useful, for example clients with unknown credit default said yes only 5.2% of the time vs 12.9% with no default. So I kept unknown as separate category.

## 4. Preparing the data

I put all preprocessing into one ColumnTransformer inside a Pipeline with the model. For pdays I changed 999 to -1 and added a 0/1 column for it. Numbers are scaled with StandardScaler and categories are one-hot encoded. Because everything is in a Pipeline, nothing leaks between CV folds.

I also checked my choice for "unknown" against the usual alternative, replacing it with the most common value. With logistic regression on the same folds both gave the same AP (0.4573 vs 0.4572), so the difference is just noise. I kept unknown as its own category because it costs nothing and keeps the information.

I split train.csv into 80% development set, where I used 5-fold stratified cross-validation, and 20% hold-out set that I didn't touch until the end.

## 5. Models I tried

The dummy model got AP 0.113, decision tree 0.428, logistic regression 0.457, and random forest and gradient boosting both got 0.476. ROC AUC was 0.794 for logistic regression and 0.803 for gradient boosting. The difference between folds was about 0.02 AP for every model, so there is real signal even without duration, but the best models are close to each other.

I also tried balanced class weights in logistic regression, but AP stayed the same. It only moved recall up and precision down, so it works like changing the threshold. Gradient boosting was about six times faster then random forest with the same score, so I picked it for tuning.

## 6. Tuning

For logistic regression different C values almost didn't change anything, I think because the model is linear. For gradient boosting I did a random search over 40 settings and the best one got AP 0.480. It beat logistic regression on all five folds, so that difference is real. But it was only 0.004 better than default boosting, which is inside the noise, so honestly there is no real winner between those two. I still used the tuned one as final model.

## 7. Choosing the threshold

Since the call centre can call only 10% of clients, I set the threshold so that 10% of clients are above it. I did this on the development set, using out-of-fold predictions, so the hold-out set still wasn't used for any decisions. The threshold is 0.331.

## 8. Results on the hold-out set

On the hold-out set the model got AP 0.444 and ROC AUC 0.792. It's a bit lower than in cross-validation, because of random noise and because picking the best of 40 settings makes CV a little too optimistic. Still, it is about four times better than guessing.

With the 0.331 threshold the model calls 637 of 6,588 hold-out clients (9.7%). 322 of these calls end with a deposit and 315 are wasted, so about every second call works. Calling random people gives only 11%, so the model is 4.5 times better (this is the lift). These calls reach 43% of all subscribers. The other 420 subscribers are never called, that's the price of calling only 10%. If the bank could call the top 20%, it would reach 63% of all subscribers.

## 9. Final model and predictions

At the end I trained the model on all of train.csv and saved it with joblib as model.joblib. It is wrapped in FixedThresholdClassifier, so predict() gives 1 (call) or 0 (don't call) directly, using the 0.331 threshold. I loaded it in a seperate Python process and got exactly the same predictions as in the notebook. The model says 1 for 9.5% of test clients, and the average predicted probability is 0.110, close to 0.113 in train, so test data looks similar to training data. Predictions are in predictions.csv with columns id, y_pred and y_score.

## 10. Limitations

The strongest features show more when the call was made than who the client is. The hold-out is a random part of the same campaigns from 2008 to 2010, so it has the same months and the same economy as the training data, and this makes the results look better than they would be in real life. On a new campaign with different economy the model would probably work worse. I couldn't check this with a split by time, because the data has no year column. Also the 10% rule says how many people to call, but not how much a call costs or how much a deposit earns. And test.csv has no labels, so I can't check how good the test predictions really are.

## 11. Conclusion

Even without call duration, we can predict who will subscribe much better than by chance (AP 0.444 vs 0.113). If the bank calls the top 10%, about half of the calls work and it finds 43% of all subscribers. The most useful information was the previous campaign, the economy, the month and the phone type. Gradient boosting was the best, but logistic regression is close and easier to explain, so it's a good simpler option.
