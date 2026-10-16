# Task 3 Report: Bank Marketing

Imamsidikov, MOP 3231 Machine Learning (Python), Narxoz University

## 1. The task

The data is from a Portuguese bank that was calling its clients and offering them a term deposit. For each call we have some info about the client (age, job, education, marital status, loans), about the call itself (month, day of week, mobile or landline), about previous campaigns, and some economic indicators. The target y shows if the client said yes to the deposit or not.

My goal was to make a model that predicts y for 8,236 clients in test.csv, using only information the bank already has before the call. Everything is in the notebook Imamsidikov_task3.ipynb. I used 2954 as random seed everywhere (last digits of my student ID), so the results are the same every time.

## 2. First look at the data

The training file has 32,940 rows and 20 features. It is pretty clean, there are no empty cells, no duplicates, and ids in train and test don't overlap. But some data is still missing in a hidden way. Six text columns have the value "unknown", and pdays uses 999 when the client was never contacted before.

The main thing I noticed is that the classes are very imbalanced. Only 11.3% of calls ended with yes (3,711 from 32,940). So accuracy is not a good metric here. A model that always says "no" already gets 88.7% accuracy, but it never finds any client who would subscribe. Because of that I mostly compared models by average precision (AP). A model that just guesses gets AP around 0.113.

There was also one tricky column, duration, which is the length of the call in seconds. It's in train.csv but not in test.csv. Successful calls are much longer (median 449 seconds vs 164), so it looks like a very strong feature. But you only know the call length after the call is finished, and then you already know the answer too. Using it would be like giving the model the answer. So I dropped duration, and id too because it's just a row number.

## 3. What the data showed

Economic indicators were the strongest numeric features. When the 3-month Euribor rate was under 1%, 46% of calls were successful, but when it was above 4% only 4 to 6% were. These columns have only about ten different values each, because they are monthly or quarterly numbers, so really they just tell the model when the call happened. Three of them are almost the same variable.

Age has kind of a U shape. Young clients up to 24 said yes 24% of the time and clients over 60 said yes 46% of the time, while people between 30 and 60 were only at 8 to 11%. The number of calls works opposite: the more the bank called the same person, the lower the chance, from 13% after one call to 6% after six or more.

From the categorical features, result of previous campaign was the strongest by far. If the last campaign was a success, 65% of those clients said yes again, compared to 9% for clients who were never in a campaign. Month also matters a lot. May was the busiest month with almost 11,000 calls but had the lowest success rate (6%), while March, September, October and December had 43 to 51%. Calling a mobile worked better then a landline (15% vs 5%). Students and retired people agreed the most, and blue-collar workers the least.

About "unknown" values: 26% of rows have at least one, so deleting them means losing a quarter of the data. They were also useful. Clients with unknown credit default status said yes only 5.2% of the time, compared to 12.9% for clients with no default. So I kept unknown as separate category.

## 4. Preparing the data

I put all preprocessing into one scikit-learn ColumnTransformer. For pdays I changed 999 to -1 and added a 0/1 column that says if pdays was 999, so the model doesn't think it means "999 days ago". Numeric columns are scaled with StandardScaler and categorical ones are one-hot encoded. After this there are 63 columns.

The transformer is inside a Pipeline together with the model, so during cross-validation it learns only from the training folds and nothing leaks from the test fold. I split train.csv in two parts. 80% was a development set, where I compared and tuned models with 5-fold stratified cross-validation. The other 20% was a hold-out set that I didn't touch until the very end, so I would have one honest score.

## 5. Models I tried

All models had the same preprocessing and same five folds, so the comparison is fair. The dummy model got AP 0.113, as expected. Logistic regression got AP 0.457, so there is real signal even without duration. A decision tree with depth 6 was the weakest real model with AP 0.428. Random forest and gradient boosting were the best, both with AP 0.476.

I also tried logistic regression with balanced class weights. I thought it would help because of the imbalance, but AP stayed basically the same. It only changed precision and recall: recall went up from 0.25 to 0.63 and precision went down from 0.66 to 0.36. So class weights work like moving the threshold, they don't make the model better.

Gradient boosting was about six times faster to train then random forest with the same score, so I picked boosting and logistic regression to tune.

## 6. Tuning

For logistic regression I tried different C values with and without class weights. It almost didn't change anything, AP stayed 0.457. I think the model is limited because it is linear, not because of its settings.

For gradient boosting I did a random search over 40 combinations. The best one had small learning rate, small trees and quite strong regularisation. It got AP 0.480. Tuned boosting beat logistic regression on all five folds, so this difference is real. But compared to default boosting, it was only 0.004 better. Honestly, tuning didn't help much, default settings were already good. I still chose tuned gradient boosting as final model.

## 7. Choosing the threshold

The model gives a probability, and the threshold decides who the bank will call. With only 11% yes, the default 0.5 is too strict and finds only about a quarter of the clients who would subscribe. I picked the threshold with the best F1 score on the development set, so the hold-out set still wasn't used. The best threshold was 0.211, but anything from 0.15 to 0.30 gives almost same result.

## 8. Results on the hold-out set

Then I tested the model one time on the hold-out set. It got AP 0.444 and ROC AUC 0.792. This is a bit lower than in cross-validation (0.480). I think there's two reasons: the hold-out set is small, so there is some random noise, and choosing the best of 40 settings makes the CV score a little too optimistic. Still, it is about four times better than guessing.

In practice, with threshold 0.211 the model would call 14% of clients, and 45% of these calls end with yes, while random calls give only 11%. If the bank called only the top 20% of clients by score, it would reach 63% of all subscribers. Accuracy went down a little, from 0.899 to 0.872, but that's ok because the point is finding subscribers.

## 9. Final model and predictions

In the end I trained the model again on all of train.csv and saved it with joblib as Imamsidikov_task3_model.joblib. I wrapped it in FixedThresholdClassifier, so predict() already uses the 0.211 threshold and gives yes or no directly.

To check that the file really works by itself, I loaded it in a seperate Python process and scored test.csv. The predictions were exactly the same as in the notebook. The model says yes for 14% of test clients, almost the same as on the hold-out set. Predictions are saved in predictions.csv with columns id, y and p_yes.

## 10. Limitations

The strongest features, economic indicators and month, describe more when the call was made than who the client is. Train and test are from the same campaigns in 2008 to 2010, so on a new campaign with different economy the model would probably work worse. I also chose the threshold with F1, which treats a wasted call and a missed client as equally bad, but a real bank would use the real cost of a call and profit from a deposit. And test.csv has no labels, so I can't say how good the test predictions really are.

## 11. Conclusion

Even without call duration, it's possible to predict who will subscribe much better than by chance. The final model has AP 0.444 vs 0.113 for guessing, and calling the top 20% of clients would reach almost two thirds of all subscribers. The most useful information was the previous campaign result, the economic situation, the month and the phone type. Gradient boosting was the best model, but only a little better than logistic regression, so logistic regression would be a good simpler option if the bank wants something easier to explain.
