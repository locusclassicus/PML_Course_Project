---
title: "PML_Course_Project"
output: 
  html_document: 
    keep_md: yes
bibliography: references.bib
---

## Outline

While many modern devices enable to collect a large amount of data about personal activity relatively inexpensively, enthusiasts of the quantified self movement rarely quantify how well they do particular activities. 

In this project, we use data from accelerometers on the belt, forearm, arm, and dumbell of 6 experiment participants trying to predict if they performed barbell lifts correctly or incorrectly. The variable we need to predict is the "classe" variable in the training set. 

The original TrainData was split into train and validation sets, and model evaluation was done on the validation set to avoid overfitting. Our experiments show that traditional classification tree gives the worst result, while bagging and random forests both offer high accuracy, with random forests being the most effective classification algorithm.

All data comes from this source: <http://groupware.les.inf.puc-rio.br/har>. Most of the methods adopted here come from [@james2013a] (see chapter 8 "Tree-based Methods").

## Download and Prepare the Data

```r
urlTrain <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
urlTest <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(urlTrain, destfile = "TrainData.csv")
download.file(urlTest, destfile = "TestData.csv")
```

```r
TrainData <- read.csv("TrainData.csv")
TestData <- read.csv("TestData.csv")
```
Visual inspection of the data shows that cols 1-7 contain information unnecessary for prediction, such as user name etc. We also remove cols with zero or close to zero variance and cols that have over 60% missing values.

```r
TrainData <- TrainData[,-c(1:7)]
TestData <- TestData[, -c(1:7)]
library(caret)
nzv <- nearZeroVar(TrainData)
TrainData <- TrainData[, -nzv]
TestData <- TestData[, -nzv]
notDeficient <-colSums(is.na(TrainData)) <= 0.6*nrow(TrainData)
TrainData<- TrainData[, notDeficient] 
TestData <- TestData[, notDeficient] 
```
The remaining observations are all complete:

```r
sum(complete.cases(TrainData)) == nrow(TrainData)
```

```
## [1] TRUE
```
We finally find the class we need to predict and transform it into factor, and split our TrainData into **Train** and **Validation Sets**.

```r
TrainData$classe <- as.factor(TrainData$classe)
levels(TrainData$classe)
```

```
## [1] "A" "B" "C" "D" "E"
```

```r
set.seed(1234)
inTrain <- createDataPartition(y = TrainData$classe, p=.6, list=FALSE)
subtrain <- TrainData[inTrain, ]
subvalid <- TrainData[-inTrain,]
```

## Classification tree
We first use classification trees to analyze the data set.

```r
## classification tree
library(tree)
tree.train <- tree(classe ~ ., subtrain)
sum.tree.train <- summary(tree.train)
## train data error
sum.tree.train$misclass ## 0.32%
```

```
## [1]  3763 11776
```

```r
## validation data error
tree.valid <- predict(tree.train, subvalid, type = "class")
table(tree.valid, subvalid$classe)
```

```
##           
## tree.valid    A    B    C    D    E
##          A 1842  296   59   88   47
##          B   77  748  178   68  202
##          C   38  119  909  160  131
##          D  241  355  222  938  285
##          E   34    0    0   32  777
```

```r
1 - ((1842+748+909+938+777)/7846) ## 0.34%
```

```
## [1] 0.3354576
```
As expected, the error is even higher on the validation set. We have performed cross-validation using the cv.tree() function to find the optimal number of terminal nodes, but this has not lead to any improvement of the accuracy.

## Bagging
Next, we apply bagging to our data, using the randomForest package in R. Since bagging is simply a special case of a random forest with m = p, the randomForest() function can be used to perform both random forests and bagging. The argument mtry specifies that all predictoris should be considered for each split of the tree - in other words, that bagging should be done.

```r
library(randomForest)
set.seed(1234)
bag.train <- randomForest(classe ~ ., data = subtrain, mtry = 52, importance = TRUE)
bag.train
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = subtrain, mtry = 52,      importance = TRUE) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 52
## 
##         OOB estimate of  error rate: 1.82%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3321   14    6    1    6 0.008064516
## B   33 2219   18    8    1 0.026327337
## C    3   25 2007   16    3 0.022882181
## D    4   11   36 1874    5 0.029015544
## E    0    6    6   12 2141 0.011085450
```
The confusion matrix suggests a significantly improved accuracy for each class, the classification error on the training data being about 2%

```r
1 - ((3321+2219+2007+1874+2141)/11776)
```

```
## [1] 0.01817255
```
As usual, we calculate accuracy on the validation set as well; the result is even slightly better.

```r
yhat.bag <- predict(bag.train, newdata = subvalid)
table(yhat.bag, subvalid$classe)
```

```
##         
## yhat.bag    A    B    C    D    E
##        A 2224   20    2    1    0
##        B    4 1491   16    2    5
##        C    3    4 1337   15    3
##        D    1    1   12 1266    4
##        E    0    2    1    2 1430
```

```r
1 - ((2224+1491+1337+1266+1430)/7846) 
```

```
## [1] 0.01249044
```
Let us see if we can improve accuracy even better using random forests.

## Random Forests
By default, randomForest() uses p/3 variables when building a random forest of regression trees, and sqrt(p) when building a random forest of classification trees. In our case, sqrt(52) = 7.2, so we shall use 7 variables. The error estimated on the validation set is even smaller than that obtrained with bagging. 


```r
set.seed(1234)
rf.train <- randomForest(classe ~ ., data = subtrain, mtry = 7, importance = TRUE)
rf.train
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = subtrain, mtry = 7,      importance = TRUE) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 7
## 
##         OOB estimate of  error rate: 0.64%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3342    5    0    0    1 0.001792115
## B   11 2260    8    0    0 0.008336990
## C    0   16 2037    1    0 0.008276534
## D    0    0   21 1907    2 0.011917098
## E    0    0    1    9 2155 0.004618938
```

```r
## validation
yhat.rf <- predict(rf.train, newdata = subvalid)
table(yhat.rf, subvalid$classe)
```

```
##        
## yhat.rf    A    B    C    D    E
##       A 2230   10    0    0    0
##       B    1 1506    8    0    1
##       C    0    2 1357   10    3
##       D    1    0    3 1274    8
##       E    0    0    0    2 1430
```

```r
1 - ((2230+1506+1357+1274+1430)/7846)
```

```
## [1] 0.00624522
```

## References
