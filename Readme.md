## About:
Predict Loan Sequence number for Washington Home Loans by performing Multiple Regression Testing on multiple Variables

## Project Description:
This project is to show the familiarity and understanding of Multiple Linear
Regression model and statistics measurement to conclude the Model and its parameters.

## Datasets and Description:
HDMA Washington State Home Loans dataset
https://www.kaggle.com/miker400/washington-state-home-mortgage-hdma2016/downloads/washington-state-home-mortgage-hdma2016.zip/1

Datasets consist of 467 rows and 47 variables. We chose only numerical variables in order to analyze the relationship of continuous variables with sequence variable.

#### Below are the 10 continuous variables used:
##### 1) Tract to msamd income: The percentage of the median family income for the tract compared to the median family income for the MSA/MD, rounded to two decimal places.
##### 2) Population: The total population in the tract.
##### 3) Minority population: The percentage of minority population to total population for the census tract, carried to two decimal places.
##### 4) Number of owner occupied units: The number of dwellings in the tract that are lived in by the owner.
##### 5) Number of 1 to 4 family units: The number of dwellings in the tract that are built to house fewer than 5 families.
##### 6) Loan amount 000s: The amount of the loan applied for, in thousands of dollars.
##### 7) Hud median family income: The median family income in dollars for the MSA/MD in which the tract is located.
##### 8) Applicant income 000s: The gross annual income that the lender relied on when evaluating the creditworthiness of the applicant, rounded to the nearest thousand.
##### 9) Sequence number: A one-up number scheme for each respondent to make each loan unique.
##### 10) Census tract number: The number of the census tract for the property. This code is only unique when combined with the state and county codes

## Data Exploration:
#### Data Head
![Alt text] (https://github.com/himanshu1214/Predict-Loan-Sequence-Number-For-Washington-Home-Loans-Using-Multiple-Regression-/blob/master/img/df.jpg)


#### BoxPlot:
To understand the raw data and its distribution.

![Alt text] (https://github.com/himanshu1214/Predict-Loan-Sequence-Number-For-Washington-Home-Loans-Using-Multiple-Regression-/blob/master/img/boxplot.jpg)

As we can see from boxplot, there are bunches of outliers. Therefore, we need to clean our raw data. In the column of applicant income 000s, we replaced the NA value by the mean of the existed data. Then we deleted other rows which contain NA value (300 out of 467k rows).

#### Standardizing and Normalizing Data
We used Chi Square Quantile-Quantile Plots to show the relationship between data-based values which should be distributed as χ^2 and corresponding quantiles from the χ^2 distribution.
In multivariate analyses, this is often used both to assess multivariate normality and check for outliers, using the Mahalanobis squared distances (𝐷2) of observations from the centroid.
When the slope that Chi Square displaces tend to 1, the data tends to standard normal distribution.

![Alt text] (https://github.com/himanshu1214/Predict-Loan-Sequence-Number-For-Washington-Home-Loans-Using-Multiple-Regression-/blob/master/img/chisquare_test.jpg)

As we can see from our Q-Q plot, the filtered data distributes far from the line that has slope of 1 (45°), which, refers that our filtered data is not perfectly normally distributed.

![alt text] (https://github.com/himanshu1214/Predict-Loan-Sequence-Number-For-Washington-Home-Loans-Using-Multiple-Regression-/blob/master/img/histogram.jpg)

When we continued to draw the histogram, it is clear in the histograms above that the filtered data is not normal distribution.
Therefore, it is necessary to standardize and normalize the filtered data.
![alt text] (https://github.com/himanshu1214/Predict-Loan-Sequence-Number-For-Washington-Home-Loans-Using-Multiple-Regression-/blob/master/img/normal_chi_square.jpg)

As is shown in the Q-Q plot above, after standardizing and normalizing the filtered data, the data points distributed concentrated near the line with slope of 1. That gave us a clue that the data was then normally distributed.

![alt text] (https://github.com/himanshu1214/Predict-Loan-Sequence-Number-For-Washington-Home-Loans-Using-Multiple-Regression-/blob/master/img/normal_hist.jpg)
The histograms reflect intuitively that the data is normally distributed.

##### Correlation Matrix
We used correlation matrix to check the independence between variables in a pair at the same time. For which, we wanted to find out the highly correlated variables (correlation coefficient is larger than 0.6), so as to delete one of the variables in the pair as its highly correlated variable can represent the deleted one. Upon which, we would get less variables to optimize our regression model.

![alt text] (https://github.com/himanshu1214/Predict-Loan-Sequence-Number-For-Washington-Home-Loans-Using-Multiple-Regression-/blob/master/img/correlation.jpg)

The closer correlation coefficient to 1 or -1, the stronger two variables correlated. We can tell the population and number of owner occupied units are highly correlated, number of owner occupied units and number of 1 to 4 family units are highly correlated, population and number of 1 to 4 family units are highly correlated.
Therefore, we deleted number of owner occupied units, so that population and population and number of 1 to 4 family units can represent its impact to the response (sequence number).

## Multiple Linear Regression Procedure
#### Hypothesis:
###### 𝐻0:the regression is not significant
###### 𝐻1: the regression issignificant
We built up 4 regression models to find the best model for our data.
###### 1st model:
Original multiple regression model, which contains all the 9 variables. 𝑦𝑖= 𝛽0+𝛽1𝑥1𝑖+𝛽2𝑥2𝑖+𝛽3𝑥3𝑖+𝛽4𝑥4𝑖+𝛽5𝑥5𝑖+𝛽6𝑥6𝑖+⋯+𝛽9𝑥9𝑖
After calculation,

![alt text] (https://github.com/himanshu1214/Predict-Loan-Sequence-Number-For-Washington-Home-Loans-Using-Multiple-Regression-/blob/master/img/model1.jpg)

After calculation, our 1𝑠𝑡model is: 𝑦𝑖= −5.309𝑒−15−5.978𝑒−3𝑥1𝑖−8.992𝑒−3𝑥2𝑖+1.833𝑒−2𝑥3𝑖+3.03𝑒−2𝑥4𝑖+⋯−2.538𝑒−2𝑥9𝑖

The test result:

![alt text] (https://github.com/himanshu1214/Predict-Loan-Sequence-Number-For-Washington-Home-Loans-Using-Multiple-Regression-/blob/master/img/model1_result.jpg)

𝑅𝑎𝑑𝑗2=0.008217, which is far from 1 implying the model has a poor fit. However, the p-value is near to zero, which tells we need to reject our 𝐻0. 𝑅𝑎𝑑𝑗2≈ 1.0 can be achieved at the expense of error degrees of freedom when an excess of model terms is employed. However, 𝑅𝑎𝑑𝑗2= 1, describing a model with a near perfect fit, does not always result in a model that predicts well. Therefore, the p-value<2.2*10−16 indicates that the regression explained by the model is significant.

###### 𝟐𝒏𝒅model:
Removal of variable number of owner occupied units(as is mentioned in the correlation matrix), then we obtain the 2𝑛𝑑model.

![alt text] (https://github.com/himanshu1214/Predict-Loan-Sequence-Number-For-Washington-Home-Loans-Using-Multiple-Regression-/blob/master/img/model2.jpg)

After calculation, our 2𝑛𝑑model is: 𝑦𝑖= −5.309𝑒−15+2.101𝑒−3𝑥1𝑖+1.283𝑒−3𝑥2𝑖+1.524𝑒−2𝑥3𝑖+6.008𝑒−4𝑥4𝑖+⋯−2.596𝑒−2𝑥9𝑖
The test result:
![alt text] (https://github.com/himanshu1214/Predict-Loan-Sequence-Number-For-Washington-Home-Loans-Using-Multiple-Regression-/blob/master/img/model2_result.jpg)

The 2𝑛𝑑model we built, comes to almost the same conclusion with even smaller 𝑅𝑎𝑑𝑗2 , yet p-value is near to zero. Therefore, we reject 𝐻0, the p-value<2.2*10−16 indicates that the regression explained by the model is significant.

##### 𝟑𝒓𝒅model:
We used stepwise selection to combine forward and backward selection.
Forward selection:        𝑦𝑖= 𝛽0+𝛽1𝑥1𝑖 𝑦𝑖= 𝛽0+𝛽1𝑥1𝑖+𝛽2𝑥2𝑖
                          𝑦𝑖= 𝛽0+𝛽1𝑥1𝑖+𝛽2𝑥2𝑖+𝛽3𝑥3𝑖
                          𝑦𝑖=𝛽0+𝛽1𝑥1𝑖+𝛽2𝑥2𝑖+𝛽3𝑥3𝑖+𝛽4𝑥4𝑖+𝛽5𝑥5𝑖+𝛽6𝑥6𝑖+⋯+𝛽9𝑥9𝑖

Backward Selection:               

                          𝑦𝑖=𝛽0+𝛽1𝑥1𝑖+𝛽2𝑥2𝑖+𝛽3𝑥3𝑖+𝛽4𝑥4𝑖+𝛽5𝑥5𝑖+𝛽6𝑥6𝑖+⋯+𝛽9𝑥9𝑖
…                         𝑦𝑖= 𝛽0+𝛽1𝑥1𝑖+𝛽2𝑥2𝑖+𝛽3𝑥3𝑖 𝑦𝑖=    
                          𝛽0+𝛽1𝑥1𝑖+𝛽2𝑥2𝑖 𝑦𝑖= 𝛽0+𝛽1𝑥1𝑖


After we applied our stepwise selection, we obtain the same variables as the 1st model.

![alt text] (https://github.com/himanshu1214/Predict-Loan-Sequence-Number-For-Washington-Home-Loans-Using-Multiple-Regression-/blob/master/img/model3.jpg)

That is, our stepwise selection model is: 𝑦𝑖= −5.309𝑒−15−5.978𝑒−3𝑥1𝑖−8.992𝑒−3𝑥2𝑖+1.833𝑒−2𝑥3𝑖+3.03𝑒−2𝑥4𝑖+⋯−2.538𝑒−2𝑥9𝑖
The result is:

 ![alt text] (https://github.com/himanshu1214/Predict-Loan-Sequence-Number-For-Washington-Home-Loans-Using-Multiple-Regression-/blob/master/img/model3_result.jpg)

As the final model is the same, so the conclusion is the same.

##### 𝟒𝒕𝒉model:
We chose 3 variables (minority population, hud median family income, loan amount 000s) with the highest correlation to sequence number to perform our regression model

![alt text] (https://github.com/himanshu1214/Predict-Loan-Sequence-Number-For-Washington-Home-Loans-Using-Multiple-Regression-/blob/master/img/model4.jpg)
After calculation, the 4𝑡ℎmodel came as: 𝑦𝑖= −5.247𝑒−15+1.601𝑒−2𝑥3𝑖+5.331𝑒−2𝑥6𝑖+4.022𝑒−2𝑥7𝑖
The result is:

![alt text] (https://github.com/himanshu1214/Predict-Loan-Sequence-Number-For-Washington-Home-Loans-Using-Multiple-Regression-/blob/master/img/model4_res.jpg)

We can see, the 𝑅𝑎𝑑𝑗2=0006865, which is near to zero. In the meanwhile, the p-value = 2.2*10−16, which gives us the confidence to reject 𝐻0.
Since all the p-value are the same (equal to 2.2*10−16), we would like to choose the 1st(3rd) as our best model as its 𝑅𝑎𝑑𝑗2 is the highest. 𝑦𝑖= −5.309𝑒−15−5.978𝑒−3𝑥1𝑖−8.992𝑒−3𝑥2𝑖+1.833𝑒−2𝑥3𝑖+3.03𝑒−2𝑥4𝑖+⋯−2.538𝑒−2𝑥9𝑖
Comparing all the 4 models above, we can see, all the 𝑅𝑎𝑑𝑗2 are far from 1, which implies the models have poor fit. However, all the 4 p-vale are almost to zero (far less than 0.0001), which indicates that the regression explained by the model is significant. In fact, in 𝑅𝑎𝑑𝑗2′𝑠 employment in multiple regression, the dangers are even more pronounced since the temptation to overfit is so great. 𝑅𝑎𝑑𝑗2≈ 1.0 can be achieved at the expense of error degrees of freedom when an excess of model terms is employed. However, 𝑅𝑎𝑑𝑗2= 1, describing a model with a near perfect fit, does not always result in a model that predicts well. Thus, according to our tests show, we can confidently reject 𝐻0, as our regression is significant.

##### Homoscedasticity Test:

To perform the residual test and come to a conclusion which models fit the best to clean data, we test all of the 4 models.
The white line is the expected model well as the red line is the actual model we performed. We can compare 4 models by the homoscedasticity test.

![alt text] (https://github.com/himanshu1214/Predict-Loan-Sequence-Number-For-Washington-Home-Loans-Using-Multiple-Regression-/blob/master/img/homoscedasticity.jpg)
The nearer the red line to the white line, the better the model fit. As we can see from the plot, combing with the 𝑅𝑎𝑑𝑗2 we obtained, the 1st and the 3rd red lines are the same nearer to the white lines than the 2nd and the 4th red lines.
All in all, the perfect model is the 1st(3rd) model we built up.
Conclusion and future
With enough and cleaning data, we can predict the sequence number a family unit would like to loan. This is the prediction for the numerical variable and it is the most reasonable response as it is related to all the numerical variables, so that we can take every number into consideration to make the best prediction for our response. In the end, we chose the 1st (3rd) model as our best model.
The acceptable value of Radj2 is not the same for all subject. “In the field of sociology, it is common that Radj2 is small because there are so many outside factors can affect altitude and behavior, 11%-15% of the social behavior can be predicted is fairly enough. “, which gives us a reasonable explanation why our 4 Radj2 are far from 1.
the P -value is less than 0.0001(which is 2.2*10−16). This should not be misinterpreted. Although it does indicate that the regression explained by the model is significant, this does not rule out the following possibilities:
1) The linear regression model for this set of x’s is not the only model that can be used to explain the data; indeed, there may be other models with transformations on the x’s that give a larger value of the F-statistic.
2) The model might have been more effective with the inclusion of other variables in addition to x1,x2...x9.
We may can transform some of the category data into numerical data as long as we have the specific standard, which may help us build a even better prediction model in the future.
We took two weeks to discuss our project. How to choose suitable topics was the first problem we encountered. Finally, we decided to choose economics-related topics because we believe that is most beneficial to our future self-development. Then the second struggling we met is how to use R studio to analyze the data. We learned a lot of references to figure out our problems. We sincerely thanks Kunmei for her kindly help and thanks professor for enlightened guidance.
