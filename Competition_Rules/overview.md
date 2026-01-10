### Description

You'll predict a continuous numerical value and be scored according to a measure inspired by the skill score.

The test set selected for scoring remains partially hidden (75%) throughout the process, to ensure a true out-of-sample evaluation.

### Evaluation

Submissions will be evaluated on the basis of this measure. The public ranking is calculated from approximately 25% of the test data. Final results will be based on the remaining 75%, so the final ranking may differ! **Keep working, you could be the first even if you are not the first in the public rankings because of the risk of overfitting.** The formula for calculating the metrics is attached:

$$\mathrm{Score} = \sqrt{1 - \operatorname{min}(\operatorname{max}(\frac{\sum_{i \in I} w_{i}(y_{i} - \hat{y_i})²}{\sum_{i \in I} w_{i}y_{i}² }, 0),1))}$$

with I a set of lines that will be 25% of the test for the public leaderboard and the remaining 75% for the private leaderboard. Below is a sample code that calculates the metric.

```python
def _clip01(x: float) -> float:
    return float(np.minimum(np.maximum(x, 0.0), 1.0))

def weighted_rmse_score(y_target, y_pred, w) -> float:
    denom = np.sum(w * y_target ** 2)
    ratio = np.sum(w * (y_target - y_pred) ** 2) / denom
    clipped = _clip01(ratio)
    val = 1.0 - clipped
    return float(np.sqrt(val))
```

This community contest does not follow the usual code-only format: you can submit your predictions as CSV files for evaluation. However, to ensure full reproducibility, we strongly recommend that you also provide a Kaggle notebook.

If your team finishes in the lead, you must submit executable code (with exact dependency versions, Python version) so that we can reproduce your results and check that no "data leakage" was exploited to game the ranking. To ensure fairness between participants, if your results are not easily reproducible you may be disqualified from monetary prize eligibility.

To avoid any data leakage, or otherwise look-forward, **your code should predict ts_index t using only data from ts_index 0 to t, processing all data strictly sequentially.**

### Submission

The submission must be a CSV file with a primary key (**id**) and a column (**prediction**). Predictions must be made on the test file (**test.parquet**).
The submission file should look like this:

```csv
id,prediction
W2MW3G2L__STALY73S__9ZI8OAJB__1__2991 ; 5.764190326788755
83EG83KQ__R571RU17__PHHHVYZI__1__3353 ;   5.764190326788755
W2MW3G2L__STALY73S__Q101PRO5__3__2991 ;  5225125.5454
83EG83KQ__R571RU17__PZ9S1Z4V__1__3353 ;  4545.4545
```

Your notebook should be structured as follows.

1. The imports
2. The functions or classes used
3. The code that will fit the model
4. The prediction code

### Tips

Even though the exact times are hidden in the test data, it comes from a period after the training data. Feel free to focus on a specific window, for example by weighting the most recent periods, if this works best for your model.

Code, sub-code and sub-category have both similarities and differences. For instance similarities may exist between codes in same sub-category, or differences may relate between the same sub-codes across different codes. It is therefore essential to have an appropriate weighting between data in the same categories and data outside these categories.

One of the difficulties lies in the low signal-to-noise / ratio. Another difficulty lies in the fact that the underlying process is probably not completely stable with ts_index/time.

You may use all data that we provide or, in the case of external data, you may use it but it must comply with the rules set out in section 6. External data and tools.

Efforts in this competition can be directed towards two key aspects:

1. Data mining and feature analysis.
2. Advanced modeling techniques.

### Prizes

Total prizes available: \$10,000, divided between the top 5 teams

1st place - \$3,500
2nd place - \$2,500
3rd place - \$2,000
4th place - \$1,000
5th place - \$1,000

**The top five candidates may be offered a job interview at a hedge fund delivering around 30% annual returns at the end of the competition.** We offer competitive terms and use modern technologies in a stimulating environment.

Do not forget that scored published on the leaderboard are not the ultimate score as competitor with better score could be using data leakage in the leaderboard.

Note: **for the top teams eligible for a prize, submission of a Jupyter Notebook capable of successfully generating the prediction results is mandatory to qualify for the prize.** If we cannot reproduce your result locally (missing dependency, missing python version, use of external data that does not comply with section 6 - External data and tools -, use of data with a ts index greater than the ts index of the target), you will be disqualified.
**If your code or model training requires a specific execution environment or hardware configuration, this must be communicated clearly.**

You are NOT permitted to use the test data set to aid the modeling process, as all test data - with the exception of the current test data point to be predicted - contains future information that would not be available at the time of prediction in a real-world environment.

As with the use of future data, any external data set containing future information will be considered a violation of the rule. So make sure that everything you use would have been available at the time of test data prediction!

### FAQ

If you have any questions about the Kaggle competition, you can post them in the discussion forum with the title **[Question for the organization] + title**.
We will publish the answer here as soon as possible if the question is relevant.

### Citation

A data COMPANY. Hedge fund - Time series forecasting. https://kaggle.com/competitions/ts-forecasting, 2026. Kaggle.