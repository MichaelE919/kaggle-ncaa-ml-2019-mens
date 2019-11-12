# NCAA March Madness (MoneyBall edition)

## Description
This is the code I wrote and used to enter a submission into [Kaggle's 2019 Men's NCAA Machine Learning competition](https://www.kaggle.com/c/mens-machine-learning-competition-2019).
#### This notebook will cover the complete process of building a classifier machine learning model to predict the win probability of each matchup in the 2019 NCAA March Madness Tournament.
### There are four major milestones:
1. Create training and test datasets
2. Create a machine learning model and train using the training set
3. Test the model using the test sets and create a submission file for Stage 1 of the Kaggle competition
4. Update datasets with 2019 data and create predictions for the 2019 NCAA March Madness Tournament

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
#### Workflow:
* For the first phase when only preliminary data is available:
  - Create training set of data from 2013 and prior
  - 2014-2017 data will be a true test set
  - Split training set into a separate training and test set
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

## What's Next
Who knows? Did I pick the right stats, or was I just lucky? Is there any stat which is the true predictor of wins in the NCAA Tournament? Probably not. Actually, I know there is not. But, there is probably one combination of features that is better than others. Maybe next year, I'll find it.
In the mean time, I'd like to incorporate some Auto ML into my model. I perform a pretty robust Grid Search to determine the best model and paramters, but I'd like to have it selected automatically. Obviously, it doesn't improve the accuracy of the model or make it worse, but it does make it more efficient.