# NCAA March Madness (MoneyBall edition)

## Description
This is the code I wrote and used to enter a submission into [Kaggle's 2019 Men's NCAA Machine Learning competition](https://www.kaggle.com/c/mens-machine-learning-competition-2019).
#### This notebook covers the complete process of building a classifier machine learning model to predict the win probability of each matchup in the 2019 NCAA March Madness Tournament.
### There are five major milestones:
1. Perform feature engineering to determine the optimum combination of features for a prediction dataset
2. Create training and test datasets
3. Identify the best classifier using GridSearch
4. Train the optimized classifier on the prediction dataset
5. Create submission and prediction files

### Part 1: Creating the training/test set
#### The dataset is generated from the following features:
* Shooting Efficiency
* Scoring Opportunity
* [True Shooting Percentage](https://captaincalculator.com/sports/basketball/true-shooting-percentage-calculator/)
* [The Four Factors](https://www.nbastuffer.com/analytics101/four-factors/)
* [Player Impact Estimate (PIE)](https://masseybasketball.blogspot.com/2013/07/player-impact-estimate.html)
* [Adjusted Offensive Efficiency](https://cbbstatshelp.com/efficiency/adjusted-efficiency/)
* [Adjusted Defensive Efficiency](https://cbbstatshelp.com/efficiency/adjusted-efficiency/)
* [Adjusted Efficiency Margin](https://cbbstatshelp.com/ratings/adjem/)
* [Defensive Rebounding Percentage](https://www.nbastuffer.com/analytics101/defensive-rebounding-percentage/)
* Rebound Percentage
* Offensive Rebound to Turnover Margin
* [Assist Ratio](https://www.nbastuffer.com/analytics101/assist-ratio/)
* Block Percentage
* Steal Percentage
* Score Differential
* [Rating Percentage Index (RPI)*](https://en.wikipedia.org/wiki/Rating_Percentage_Index)
* Tournament Seed
* Win Percentage

*The RPI Rankings are replaced with the NET rankings in 2019*


### Part 2: Create and train the machine learning model
#### Fit the dataset to the following classifiers:
* [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
* [Support Vector Machines](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC)
* [XGBoost](https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn)
* [Decision Tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
* [Random Forests](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
* [Gradient Boosting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
#### Workflow:
* For the first phase when only preliminary data is available:
  - Create training set of data from 2014 and prior
  - 2015-2018 data will be a true test set
  - Split training set further into a separate training and test set
* For the second phase when all the data is available, the entire data set is split:
  - 80% for the training set and 20% for the test set
* Initiate classifiers, create parameter and pipeline objects
* Use grid search validation to choose the best classifier and parameters
* Use best performing classifer and fit with full training set
* Create data to input into the model
* Create predictions

## Result
I dramatically improved from last year, from placing 498th out of 934 (top 54%) in 2018 to placing 39th out of 866 (top 5%). That was good enough for a silver medal. 

## What changed?
After replacing efficiency numbers with Ken Pomeroy's efficiency numbers adjusted for competition ([kenpom.com](kenpom.com)) and adding an Offensive Rebound to Turnover Margin resulted in improved accuracy with last year's model, I set out to add more features, and determine the best combination of stats. Just to be sure, I tested the model with less features and it didn't perform as well. After some research and different combinations of features, I ended up with the following additional features:
* Shooting Efficiency
* Scoring Opportunity
* True Shooting Percentage
* Rebound Percentage
* Block Percentage
* Steal Percentage

## What's Next?
Who knows? Did I pick the right stats, or was I just lucky? Is there any stat which is the true predictor of wins in the NCAA Tournament? Probably not. Actually, I know there is not. But, there is probably one combination of features that is better than others. Maybe next year, I'll find it.
In the mean time, I'd like to incorporate some Auto ML into my model. I perform a pretty robust Grid Search to determine the best model and parameters, but I'd like to have it selected automatically. Obviously, it doesn't improve the accuracy of the model or make it worse, but it does make it more efficient.