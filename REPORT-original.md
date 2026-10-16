# Task 3 Report: Bank Marketing

Imamsidikov, MOP 3231 Machine Learning (Python), Narxoz University

## 1. The task

The data comes from a Portuguese bank that called its clients by phone and offered them a term deposit. For every
call we know some things about the client (age, job, education, marital status, whether they have loans), some things
about the call (month, day of the week, mobile or landline), what happened in previous campaigns, and a few economic
indicators at the time of the call. The target y says whether the client agreed to open the deposit or not.

The goal was to build a model that predicts y for the 8,236 clients in test.csv, using only information that the bank
has before it makes the call. All the work is in the notebook Imamsidikov_task3.ipynb. I used 2954 as the random seed
everywhere (the last digits of my student ID), so the notebook gives the same results every time it is run.

## 2. First look at the data

The training file has 32,940 rows and 20 features. It is quite clean: there are no empty cells, no duplicated rows,
and the ids in train and test don't overlap. Some information is still missing, though, just in a hidden way. Six of
the text columns have the value "unknown", and pdays (days since the client was last contacted) uses 999 to mean that
there was no previous contact.

The most important thing I noticed is that the classes are very imbalanced. Only 11.3% of the calls ended with yes
(3,711 out of 32,940). Because of this, accuracy is not a good way to judge a model here. A model that always answers
no already has 88.7% accuracy, but it never finds a single client who would subscribe. So I mainly compared models by
average precision (AP), which looks at how well the model finds the yes clients. A model that just guesses gets an AP
of about 0.113.

There was also one column I had to think about: duration, the length of the call in seconds. It is in train.csv but
not in test.csv. Successful calls are much longer (the median is 449 seconds against 164), so at first it looks like a
very strong feature. But the length of a call is only known after the call ends, and at that moment the answer is
already known too. Using it would basically mean giving the model the answer, and it couldn't be used on test.csv
anyway. So I dropped duration, and also id, because it is just a row number.

## 3. What the data showed

The economic indicators turned out to be the strongest numeric features. When the 3-month Euribor rate was below 1%,
46% of the calls were successful, but when it was above 4%, only 4 to 6% were. The employment variation rate and the
number of employees show the same picture. These columns have only around ten different values each, because they are
monthly or quarterly figures, so in practice they tell the model when the call was made. Three of them are also almost
the same variable (correlations between 0.91 and 0.97).

Age has a U shape. Young clients up to 24 said yes 24% of the time and clients over 60 said yes 46% of the time,
while people between 30 and 60 were at 8 to 11%. The number of calls in the current campaign works the other way: the
more often the bank called the same person, the lower the chance, from 13% after one call down to 6% after six or
more. Clients who had been contacted in earlier campaigns were much more likely to agree.

Among the categorical features, the result of the previous campaign was by far the strongest. If the last campaign
was a success, 65% of those clients said yes again, compared to 9% of clients who had never been in a campaign
before. The month also matters a lot. May was the busiest month with almost 11,000 calls but had the lowest success
rate (6%), while March, September, October and December had 43 to 51%, but only a few hundred calls each. Calls to a
mobile phone worked better than calls to a landline (15% against 5%). Students (31%) and retired people (25%) agreed
most often, and blue-collar workers least often (7%). Housing loan, personal loan and day of the week hardly made any
difference.

About the "unknown" values: 26% of the rows have at least one of them, so deleting those rows would mean losing a
quarter of the data. They also turned out to be useful. Clients with an unknown credit default status said yes only
5.2% of the time, compared to 12.9% for clients with no default. Replacing unknown with the most common value would
hide that difference, so I kept it as a separate category.

## 4. Preparing the data

I put all the preprocessing into one scikit-learn ColumnTransformer. For pdays I replaced 999 with -1 and added a
separate 0/1 column that says whether pdays was 999, so the model doesn't treat 999 as "999 days ago". The numeric
columns are scaled with StandardScaler, and the categorical columns are one-hot encoded, with unknown kept as a normal
level. After this there are 63 columns.

The transformer is placed inside a Pipeline together with the model. This way, during cross-validation it only learns
the scaling and the categories from the training folds, and nothing leaks from the fold that is being tested. I also
used only standard scikit-learn classes and no functions of my own, so the saved model can be loaded later in a new
Python session that has only the required libraries.

To evaluate the models I split train.csv into two parts. 80% became a development set, where I compared and tuned
models with 5-fold stratified cross-validation. The other 20% was a hold-out set that I didn't look at until the very
end, so that I would have one honest score for the final model.

## 5. Models I tried

All models used the same preprocessing and the same five folds, so the comparison is fair. The dummy model, which just
predicts the base rate, got AP 0.113 and ROC AUC 0.5, as expected. Logistic regression got AP 0.457 and ROC AUC 0.794,
so there is clearly real signal in the data even without duration. A single decision tree with depth 6 was the weakest
real model with AP 0.428. Random forest and gradient boosting (HistGradientBoostingClassifier) were the best, both
with AP 0.476, and ROC AUC 0.798 and 0.803.

I also tried logistic regression with class_weight="balanced". I expected it to help because of the imbalance, but the
AP stayed basically the same (0.454). What changed was the balance between precision and recall at the 0.5 threshold:
recall went up from 0.25 to 0.63 and precision went down from 0.66 to 0.36. So class weights act like moving the
threshold, they don't make the model better at separating the clients.

One more interesting thing came up in the logistic regression coefficients. The number of employees got a positive
coefficient, even though on its own it is negatively related to yes. This happens because it is so strongly
correlated with the other economic columns, so I decided not to interpret those coefficients one by one.

Gradient boosting was about six times faster to train than the random forest with the same score, so I chose
boosting and logistic regression as the two models to tune.

## 6. Tuning

For logistic regression I tried a grid of C values from 0.001 to 100, with and without class weights. It barely
changed anything, and AP stayed at 0.457. I think the model is limited by being linear, not by its settings.

For gradient boosting I ran a random search over 40 combinations of learning rate, tree size, minimum leaf size and L2
regularisation. The best one had a small learning rate (about 0.027), small trees with 15 leaves, at least 20 rows per
leaf and fairly strong regularisation. It reached AP 0.480 and ROC AUC 0.804.

Since all models used the same folds, I could compare them fold by fold. Tuned boosting beat tuned logistic
regression on all five folds, by 0.022 AP on average, so that difference is real. Compared with default boosting
though, the tuned version was only 0.004 better and won just three folds out of five. To be honest, tuning didn't help
much here, the default settings were already good. I still went with tuned gradient boosting as the final model.

## 7. Choosing the threshold

The model gives a probability, and the threshold decides who the bank would call. With only 11% yes, the default
threshold of 0.5 is too strict and finds only about a quarter of the clients who would subscribe. I chose the
threshold that gives the best F1 score, using out-of-fold predictions on the development set, so the hold-out set
still wasn't used for any decision. The best threshold was 0.211. F1 is quite flat around this value, so anything
between 0.15 and 0.30 would give almost the same result.

## 8. Results on the hold-out set

After that I trained the model on the whole development set and tested it once on the hold-out set. It got AP 0.444
and ROC AUC 0.792. This is a bit lower than in cross-validation (0.480 and 0.804). I think there are two reasons. The
hold-out set has only 742 yes clients, so there is some random noise. And when you pick the best of 40 settings on the
same folds, the cross-validation score ends up slightly too optimistic. That is exactly why I kept a hold-out set, and
0.444 is the number I would report as the real performance. It is still about four times better than guessing.

In practical terms, with the 0.211 threshold the model would call 911 of the 6,588 hold-out clients, which is 14%. Of
those calls, 45% end with a subscription, while calling people at random gives only 11%. Together they reach 55% of
all the clients who would subscribe. If the bank only called the top 10% of clients by score, it would already reach
44% of the subscribers, and the top 20% would reach 63%. Accuracy dropped from 0.899 at threshold 0.5 to 0.872 at
0.211, but that is fine, because the point is to find subscribers, not to get the highest accuracy.

## 9. Final model and predictions

At the end I trained the model again on all 32,940 rows of train.csv and saved it with joblib as
Imamsidikov_task3_model.joblib. I wrapped it in FixedThresholdClassifier, so predict() already uses the 0.211
threshold and answers yes or no directly, and predict_proba() still gives the probabilities. It also accepts test.csv
as it is, including the id column, because the pipeline picks its columns by name.

To check that the file really works on its own, I loaded it in a separate Python process without any of my notebook
code and scored test.csv. The predictions were exactly the same as the ones in the notebook. The model predicts yes
for 1,154 of the 8,236 test clients (14%), almost the same share as on the hold-out set, and the average predicted
probability is 0.110, close to the 0.113 in the training data. So the test data looks similar to the training data.
The predictions are saved in predictions.csv with three columns: id, y and p_yes.

## 10. Limitations

The strongest features, the economic indicators and the month, describe when a call was made more than who the
client is. Train and test are a random split of the same campaigns from 2008 to 2010, so they share the same periods.
On a new campaign in a different economic situation the model would probably do worse. I couldn't check this with a
split by time, because the data has no year column.

I chose the threshold with F1, which treats a wasted call and a missed client as equally bad. A real bank would choose
it based on the cost of a call and the profit from a deposit, which I don't know. I also can't say how good the test
predictions really are, because test.csv has no labels. And of course this is a teaching dataset, so the results
shouldn't be taken as facts about real banks or clients.

## 11. Conclusion

Even without the call duration, it is possible to predict who will subscribe much better than by chance. The final
model has an AP of 0.444 compared to 0.113 for guessing, and if the bank called the 20% of clients with the highest
scores, it would reach almost two thirds of all subscribers. The most useful information was the result of the
previous campaign, the economic situation, the month and the type of phone. Gradient boosting was the best model, but
only slightly better than logistic regression, so logistic regression would be a reasonable simpler choice if the bank
wanted predictions that are easier to explain.
