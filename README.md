# bank_marketing_outcome

## Overview
* predict which individuals will make a bank deposit
* built the following models:
  * decision tree
  * random forest
  * neural network
* random forest performed best with a MSE of .15

## EDA
* whether or not individuals previously made a deposit and the individual's occupation affected the outcome the most
  * people who previously made a deposit are more likely to make another deposit
  * retired, self-employed, entrepreneurs and people who work in mangement are more likely to make a deposit
![Alt text](https://github.com/sesankm/bank_marketing_outcome_prediction/blob/main/plots/jobs.png)
![Alt text](https://github.com/sesankm/bank_marketing_outcome_prediction/blob/main/plots/poutcome.png)

## Libraries used:
<strong> Python Version: </strong> 3.8 <br>
<strong> Libraries: </strong> pandas, scikit-learn, keras, matplotlib

## Models' Performance
<strong> Decision tree: </strong> MSE=0.23 <br>
<strong> Random Forest: </strong> MSE=0.15 <br>
<strong> Neural Network: </strong> MSE=0.16 <br>
