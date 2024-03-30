##################### Statistical Modeling in RStudio #####################

########  Testing of Hypothesis  ########

# Installing all the required packages for the R Notebook

install.packages("car")
library(car)
install.packages("ggplot2")
library(ggplot2)
install.packages("gplots")
library(gplots)
install.packages("qqplotr")
library(qqplotr)
install.packages("boot")
library(boot)

######## One Sample t Test #######

# Problem: Testing whether the average Processing Time of PO_Processing 
# data set is less than equal to 40.

# Step 1: Reading the data as mydata

# SESSION -> SET WORKING DIRECTORY -> TO SOURCE FILE LOCATION

mydata = read.csv('PO_Processing.csv',header = T,sep = ",")

# IMPORT DATASET -> FROM TEXT -> CHOOSE DATA -> HEADING = YES, RENAME THE DATA

PT = mydata$Processing_Time

# Step 2: Using the t Test function to test our hypothesis

# ?t.test

t.test(PT, alternative = 'greater', mu = 40)

# p-value < 0.05 => Reject H0.

######## Normality Test ########

# Problem: Checking whether the Processing Time Data is Normally Distributed

qqnorm(PT)
qqline(PT)

# Normality Check using Shapiro-Wilk test

shapiro.test(PT) 

######## One Way ANOVA ########

# Reading data and variables in R

mydata = read.csv('Sales_Revenue_Anova.csv',header = T,sep = ",")
location = mydata$Location 
revenue = mydata$Sales.Revenue

# Converting location to factor

location = factor(location)

# H0 : location-wise sales figures are equal. 

# Computing ANOVA table

fit = aov(revenue  ~  location)
summary(fit)
# H0 rejected. Sales figures are not equal. 
aggregate(revenue ~ location, FUN = mean)
boxplot(revenue ~ location)
plotmeans(revenue ~ location)

# Tukey's Honestly Significant Difference (HSD) Test

TukeyHSD(fit)

# Bartlett's test

bartlett.test(revenue, location, data = mydata) 


######## Session 9 ##################
######## Regression Analysis ########

#### Simple Linear Regression ####
#(dependent variable is regressed upon one predictor variable)

# Reading the data and variables

mydata = read.csv('DC_Simple_Reg.csv',header = T,sep = ",")
Temp = mydata$Dryer.Temperature 
DContent = mydata$Dry.Content

# Constructing Scatter Plot
plot(Temp, DContent)

# Computing Correlation Matrix
cor(Temp, DContent)
# Correlation between y & x need to be high (preferably 0.8 to 1 to -0.8 to -1.0)

# Fitting Regression Model
model = lm(DContent ~ Temp) 
summary(model)

# DContent = 2.183813 + 1.293432*Temp

# Regression Performance
anova(model)

# Residual Analysis
pred = fitted(model) 
Res = residuals(model) 

# write.csv(pred,"C:/Users/Datacore/Downloads/data_and_code/Data and Code/Pred.csv") 
# write.csv(Res,"C:/Users/Datacore/Downloads/data_and_code/Data and Code/Res.csv")

# Visualizing Actual vs Predicted Values
plot(DContent, pred)

# Checking whether the distribution of the Residuals is bell shaped (Normality Test)
qqnorm(Res) 
qqline(Res)

# Normality Check using Shapiro-Wilk test
shapiro.test(Res) 

# Visualizing the relationship between Residuals and Predicted values
plot(pred,Res)

# Visualizing the relationship between Residuals and Predictor
plot(Temp,Res)

# Bonferonni Outlier Test
library(car)
outlierTest(model) # Bonferroni p-value < 0.05 indicates potential outlier

# Leave One Out Cross Validation 
attach(mydata)
library(boot)
mymodel = glm(Dry.Content ~ Dryer.Temperature)
valid = cv.glm(mydata, mymodel)
valid$delta[1]

#### Multiple Linear Regression ####
#dependent variable is regressed upon two or more predictor variables

# Reading the data and variables
data = read.csv('Mult_Reg_Yield.csv',header = T,sep = ",")
mydata= data[,-1] # Removing SL.NO. Column 
attach(mydata)
# Computing Correlation Matrix
cor(mydata)

# Fitting Multiple Linear Regression
model = lm(X.Yield ~ Temperature + Time) 
summary(model)

# X.Yield = -67.88436 - 0.06419*Temperature + 0.90609*Time

# Temperature is NOT a causal variable. 

# Regression Model Performance
anova(model)

# From the ANOVA Table we can say only time is related to % yield as p value < 0.05, so we modify our model
# Fitting Linear Regression Model with Time as the only Predictor variable

model_m = lm(X.Yield ~  Time) 
summary(model_m)

# X.Yield = -81.6205 + 0.9065*Time 

# Regression Model Performance
anova(model_m)

# Residual Analysis
pred = fitted(model_m) 
Res = residuals(model_m) 
plot(Res)
qqnorm(Res)

# write.csv(pred,"C:/Users/Datacore/Downloads/data_and_code/Data and Code/Pred_m.csv") 
# write.csv(Res,"C:/Users/Datacore/Downloads/data_and_code/Data and Code/Res_m.csv")

#Standardizing the Residuals using scale function
#"center" parameter (when set to TRUE) is responsible for subtracting the mean on the numeric object from each observation.
#The "scale" parameter (when set to TRUE) is responsible for dividing the resulting difference by the standard deviation of the numeric object.
Std_Res = scale(Res, center = TRUE, scale = TRUE) 
write.csv(Std_Res,"C:/Users/Datacore/Downloads/data_and_code/Data and Code/Std_Res_m.csv")

# Normality Check using Shapiro - Wilk test
shapiro.test(Res) 

# Bonferonni Outlier Test
outlierTest(model_m) # Bonferroni p-value < 0.05 indicates potential outlier

# Leave One Out Cross Validation 
mymodel = glm(X.Yield ~ Time)
valid = cv.glm(mydata, mymodel)
valid$delta[1]

# Multiple Linear Regression (dependent variable is regressed upon two or more predictor variables)

# Reading the data and variables
data = read.csv('Mult_Reg_Conversion.csv',header = T,sep = ",")
mydata = data[,-1] # Removing SL.NO. Column 
attach(mydata)

# Computing Correlation Matrix
cor(mydata)
# High Correlation between X..Conversion and Temperature & Time
# High Correlation between Temperature & Time - Multicollinearity

# Fitting Multiple Linear Regression
model = lm(X..Conversion ~ Kappa.number + Temperature + Time) 
summary(model)

# Regression ANOVA
anova(model)

# Checking Multi-collinearity using Variance Inflation Factor 
vif(model)

# VIF > 5 indicates multi-collinearity. Hence, multi-collinearity exists between Time and Temperature

# Tackling Multi-collinearity

# Method 1: Removing highly correlated variable - Stepwise Regression

install.packages ("MASS")
library(MASS) 
mymodel = lm(X..Conversion ~ Temperature + Time + Kappa.number) 
step = stepAIC(mymodel, direction = "both")
summary(step)

# Check for multicollinearity in the new model
vif(step)
# vif values <5 indicates no multicollinearity

# Predicting the values
pred = predict(step) 
res = residuals(step)
cbind(X..Conversion, pred, res)
mse = mean(res^2)
rmse = sqrt(mse)

# K Fold Validation
##############DOUBT ###################

install.packages("DAAG")
library(DAAG) 
cv.lm(model, m = 16)
cv.lm(model, df = mydata, m = 16) 

# Method 2: Principal Component Regression

install.packages("pls")
library(pls) 
mymodel = pcr(X..Conversion ~ ., data = mydata, scale = TRUE)
summary(mymodel)
mymodel$loadings
mymodel$scores

pred = predict(mymodel, type = "response", ncomp = 1)
res = X..Conversion - pred 
mse = mean(res^2)
prednew = predict(mymodel, type = "response", ncomp = 2)
resnew = X..Conversion - prednew 
msenew = mean(resnew^2)

# Since there is not much reduction in MSE by including the second principal component , only PC1 is required for modelling

# Method 3: Partial Least Square  Regression

mymodel = plsr(X..Conversion ~ ., data = mydata, scale = TRUE)
summary(mymodel)
######### DOUBT
mymodel$loading
ps = mymodel$scores 
score = ps[,1:2]

#Identifying the required number of components in the model
pred = predict(mymodel, data = mydata, scale = TRUE, ncomp = 1)
res = X..Conversion - pred 
mse = mean(res^2)

prednew = predict(mymodel, data = mydata, scale = TRUE ,  ncomp = 2)
resnew = X..Conversion - prednew 
msenew = mean(resnew^2)
# Not much reduction in MSE by including the second component , only PLS1 is required for modelling

# Method 4: Ridge regression 

install.packages("glmnet")
library(glmnet)
set.seed(1)
y = mydata[,4]
x = mydata[,1:3]
x = as.matrix(x)
mymodel = cv.glmnet(x , y, alpha =0)
plot(mymodel)

# Choose the lambda which minimizes the mean square error
bestlambda = mymodel$lambda.min
bestlambda

# Develop the model with best lambda and identify the coefficients
mynewmodel = glmnet(x, y, alpha = 0) 
predict(mynewmodel, type = "coefficients", s = bestlambda)[1:4,]

######## Linear Regression with Dummy Variables #########

mydata =  read.csv('Travel_dummy_Reg.csv',header = T,sep = ",")
attach(mydata)
mydata = mydata[,2:4]

# Converting categorical x's to factors
gender = factor(Gender)
income = factor(Income)

# Fitting the model
mymodel =  lm(Attitude ~ gender + income)
summary (mymodel)

# Regression Model Performance
anova(mymodel)


######## Session 10 ###########

# Session -> Set Working Directory -> To Source File Location

########### Binary Logistic Regression ###########
# Response variable is Categorical

#Reading the file and variables
mydata =  read.csv('Resort_Visit.csv',header = T,sep = ",")
mydata
# either attach the data or use data$. to extract columns
visit = mydata$Resort_Visit
income = mydata$Family_Income
attitude = mydata$Attitude.Towards.Travel
importance = mydata$Importance_Vacation
size = mydata$House_Size
age = mydata$Age._Head

#Converting response variable to discrete
visit = factor(visit)

# Computing Correlation Matrix
cor(mydata)
# Correlation between X variables should be low otherwise it will indicate Multicollinearity

# Checking relation between Xs and Y
aggregate(income ~visit, FUN = mean)
aggregate(attitude ~visit, FUN = mean)
aggregate(importance ~visit, FUN = mean)
aggregate(size ~visit, FUN = mean)
aggregate(age ~visit, FUN = mean)
# Higher the difference in means, stronger will be the relation to response variable

# Checking relation between Xs and Y - box plot
boxplot(income ~ visit)
boxplot(attitude ~ visit) 
boxplot(importance ~ visit) 
boxplot(size ~ visit)
boxplot(age ~ visit)

# Fitting Logistic Regression Model
model = glm(visit ~ income + attitude + importance + size + age, family = binomial(logit))
summary(model)

# Perform Logistic regression - ANOVA
anova(model,test = 'Chisq')
# Since p value < 0.05 for Income redo the modeling with important factors only

# Modifying the Logistic Regression Model
model_m = glm(visit ~ income, family = binomial(logit))
summary(model_m)

# Perform Logistic regression - Anova
anova(model_m,test = 'Chisq')
cdplot(visit ~ income)

# Fitted Value & Residual
predict(model_m,type = 'response')
residuals(model_m,type = 'deviance')
predclass = ifelse(predict(model_m, type ='response')>0.5,"1","0")
predclass

# Model Evaluation
mytable = table(visit, predclass)
mytable
prop.table(mytable)


######### Ordinal Logistic Regression ###########

# Read the data file and variables
mydata = read.csv('ST_Defects.csv', header = T, sep = ",")
dd = mydata$DD
effort = mydata$Effort 
coverage = mydata$Test.Coverage
dd = factor(dd)
dd

# Install MASS Package

install.packages("MASS")
library(MASS) 

# Fitting the model
mymodel = polr(dd ~ effort + coverage)
summary(mymodel)

# Predicting Values
pred = predict(mymodel) 
pred
fit = fitted(mymodel)
fit
output = cbind(dd, pred)
output
# write.csv(output, "C:/Users/Downloads/data_and_code/Data and Code/Output.csv")

# Comparing Actual Vs Predicted
mytable = table(dd, pred)
mytable
prop.table(mytable)

#################### Nonlinear Regression ####################

# Read the data file and variables
mydata = read.csv('Nonlinear_Thrust.csv', header = T, sep = ",")
mydata
attach(mydata)
cor(mydata)
plot(x1,y) 
plot(x2,y) 
plot(x3,y) 
mymodel = lm(y ~ x1 + x2 + x3, data = mydata) 
summary(mymodel)

# Install car package 
install.packages("car")
library(car)
crPlots(mymodel)

# Design polynomial model-1
newmodel1 = lm(y ~ poly(x1, 2, raw = TRUE) + x2 + x3, data = mydata) 
crPlots(newmodel1)

# Design polynomial model-2
newmodel2 = lm(y ~ poly(x1, 3, raw = TRUE) + x2 + x3, data = mydata) 
crPlots(newmodel2)

# Design Final Polynomial model
finalmodel = lm(y ~ poly(x1, 3, raw = TRUE) + poly(x2, 2, raw = TRUE) + sqrt(x3), data = mydata) 
crPlots(finalmodel)
summary(finalmodel)
res = residuals(finalmodel) 
qqnorm(res)
qqline(res) 
shapiro.test(res)

##################### Regression Spline #####################
# Read the data file and variables
mydata = read.csv('Reg_Spline_DFR.csv', header = T, sep = ",")
mydata
design = mydata$Design
coding = mydata$Coding
plot(design, coding)
mymodel = lm(coding ~ design)
summary(mymodel)
pred = predict(mymodel)
plot(design, coding)
lines( design, pred, col = "blue")
design44 = design - 0.44
design44[design44 < 0] = 0
mymodel = lm(coding ~ design + design44)
summary(mymodel)
pred = predict(mymodel)
plot(design, coding)
lines(design, pred, col = "blue")
# designsq = design^2
# designcb = design^3
design44cb = design44^3
mymodel = lm(coding ~ poly(design, 3, raw = TRUE) + design44cb)
summary(mymodel)
pred = predict(mymodel)
plot(design, coding)
lines(design, pred, col = "blue")

################################# END OF SESSION ##################################



