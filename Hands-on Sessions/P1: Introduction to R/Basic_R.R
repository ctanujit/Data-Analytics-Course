
####################### Installation of R and RStudio #########################
# (editor comment symbol in R is # )

# Download R software from http://cran.r-project.org/bin/windows/base/
# Run the R set up (exe) file and follow instructions
# Double click on the R icon in the desktop and R window will open
# Download RStudio from http://www.rstudio.com/
# Run R studio set up file and follow instructions
# Click on R studio icon, R Studio IDE Studio will load
# Go to R-Script (Ctrl + Shift + N)
# Write 'Hello World'
# Save & Run (Ctrl + Enter)

################################################################################

'Hello World'

############ Simple Maths in R ####################

3+5
12+3/4-5+3*8
(12+3/4-5)+3*8
pi*2^3-sqrt(4)
factorial(4)
log(2,10)
log(2,base=10)
log10(2)
log(2)

x = 3+5
x
y = 12+3/4-5+3*8
y
z = (12+3/4-5)+3*8
z

########### R is case sensitive and no space should be between < & - ############

A <- 6+8
a
A

########## Write numeric / text data ################

data1 = c(3, 6, 9, 12, 78, 34, 5, 7, 7) ## numerical data
data1
data1.text = c('Mon', "Tue",'Wed') ##text data
data1.text
data2.text = c(data1.text , 'Thu' , "Fri") ## SIngle or double quote works
data2.text

############ Scan command for Read Data Values ###############

data3 = scan(what = 'character')
mon
tue
wed 
thur ### Double enter in Console to stop ###
data3
data3[2]
data3[2] = 'mon'
data3
data3[6]= 'sat'
data3
data3[2] = 'tue'
data3[5] = 'fri'
data3

############# Working directory ################

getwd() ### Session - Set Working Directory - To Source File Location ###
# setwd("C:/Users/TANUJIT/Dropbox/Teaching/Data Analytics (IIFT)/My Lectures/MBA (IB) 2020 - 2022")
dir()  ### working directory listing
ls()   ### Work space listing of objects
rm('object')  ### Remove an element "object", if exist
rm(list = ls(all = TRUE)) ### Cleaning


##### Importing Data and accessing it #####

# data = read.csv('Logistic_Reg.csv',header = T,sep = ",")
# data = read.table('Logistic_Reg.csv',header = T,sep = ",")
data
str(data)
data$Ind_Exp_Act_Score
data$Ind_Exp_Act_Score[1] = 5.2 ###This change has happened in work space only not in the file
data$Ind_Exp_Act_Score  ### Save the changes ###
write.table(data, file = 'Logistic_Reg_mod.csv',row.names = FALSE,sep = ",")
write.csv(data, file = 'Logistic_Reg_mod.csv',row.names = TRUE)

##### Vectors in R #####

x = c(1,2,3,4,56)
x
mean(x)
x[2]
x = c(3, 4, NA, 5)
mean(x)
mean(x, rm.NA = T)
x = c(3,4, NULL, 5)
mean(x)
length(x)

##### Long Vectors in AP #####

x = 1:20
x
y = seq(2,5,0.3)
y
length(y)

###### More in Vectors #####

x = 1:5
x
mean(x)
x^2
x+1
2*x
exp(sqrt(x))
y = c(0,3,4,0)
x+y
y = c(0,3,4,0,9)
x+y

##### Matrices in R #####

#a.matrix <- matrix(vector, nrow = r, ncol = c, byrow = FALSE, dimnames = list(char-vector-rownames, char-vector-col-names))
y <- matrix(1:20, nrow = 4, ncol = 5)
y
A = matrix(c(1,2,3,4),nrow=2,byrow=T)
A
A = matrix(c(1,2,3,4),ncol=2)
A
B = matrix(2:7,nrow=2)
B
C = matrix(5:2,ncol=2)
C
mr <- matrix(1:20, nrow = 5, ncol = 4, byrow = T)
mr
mc <- matrix(1:20, nrow = 5, ncol = 4)
mc
dim(B) #Dimension
nrow(B)
ncol(B)
A+C
A-C
A%*%C  #Matrix multiplication.
A*C    #Entry-wise multiplication
t(A)   #Transpose
A[1,2]
A[1,]
B[1,c(2,3)]
B[,-1]

##### Examples of lists in R #####

x = list(name = 'Tanujit', nationality = 'Indian' , height =5.5 , marks =c(95,45,80))
names(x)      
x$name
x$hei   #abbreviations are OK
x$marks
x$m[2] 

# Data Frame in R

d <- c(1,2,3,4)
e <- c("Tanujit", "Subhajit", "Indrajit", NA)
f <- c(TRUE,TRUE,TRUE,FALSE)
myframe <- data.frame(d,e,f)
names(myframe) <- c("ID","Name","Passed") # Variable names
myframe
myframe[1:3,] # Rows 1,2,3 of data frame
myframe[,1:2] # Columns 1,2 of data frame
myframe[c("ID","Name")] #Columns ID and color from data frame
myframe$ID # Variable ID in the data frame

##### Factors in R #####
### Example: variable gender with 20 "male" entries and 30 "female" entries ###

gender <- c(rep("male",20), rep("female", 30))
gender <- factor(gender)   # Stores gender as 20 1s and 30 2s
# 1=male, 2=female internally (alphabetically)
# R now treats gender as a nominal variable
summary(gender)

##### Functions in R #####

g = function(x,y) (x+2*y)/5
g(5,10)
g(10,5)

##################### Session 5 #############################################
###################### Use different packages after installation  ###########

install.packages('MASS')
library('MASS')

################### R Arithmetic operators #############################

# + : Addition
# - : subtraction
# * : multiplication
# / : division
# ^ or ** : exponiation
# x%%y : Modulus (Remainder from division) (x mod y) : exam; 5%%2 is 1
# x%/%y : integer division 

##### Define variables and array to do the above Arithmetic operations #####

x <- 6
y <- 20
print(x+y)
print(x-y)
print(x*y)
print(y/x)
print(y%%x)
print(y%/%x)
print(y^x)

##################  R Relational Operators #########################

# < : Less than
# > : Greater than
# <= : Less than or equal to
# >= : Greater than or equal to
# == : Equal to
# != : Not equal to

print(x<y)
print(x>y)
print(x<=y)
print(x>=y)
print(x==y)
print(x!=y)

###############  Operation on Vectors  ##################

x1 <- c(1,4,5)
y1 <- c(2,3,6)
print(x1+y1)
print(x1 > y1)

################# R Logical Operators ##########################

# ! : Logical NOT 
# & : Element-wise logical AND
# && : Logical AND
# | : Element-wise logical OR
# || : Logical OR

# Operators & and | perform element-wise operation producing result having length of the longer operand.
# But && and || examines only the first element of the operands resulting into a single length logical vector.
# Zero is considered FALSE and non-zero numbers are taken as TRUE. An example run.

x2 <- c(TRUE,FALSE,1,TRUE)
y2 <- c(FALSE,TRUE,0,TRUE)
print(!x2)
print(x2&y2)
print(x2&&y2)
print(x2|y2)
print(x2||y2)


######### R Assignment Operators #########################

# <-, = : Leftwards assignment
# -> : Rightwards assignment
x3=4
print(x3 <- 3)
print(10->x3)

###########  Matrix Multiplication ##################

?matrix ### Know about functions ###
A = matrix(c(21,57,89,31,7,98), nrow =2, ncol=3, byrow = TRUE)
B = matrix(c(24, 35, 15, 34, 56,25), nrow = 3, ncol = 2, byrow = TRUE)
print(A)
print(B)
C = A%*%B ### Do the multiplication of A and B and stored it into C matrix ###
print(C)
print(det(C))
Inv <- solve(C) ### Do the inverse of a matrix ###
print(Inv)
print(eigen(C)) ### Find Eigenvalues ###

######### alternative way to find inverse of a matrix A1 ##############

## Using the inv() function:
## inv() function is a built-in function in R which is especially used to find the inverse of a matrix
## For that you need to install the matlib package in your environment. and use it using library() 
## It takes time .. Do it later 
## install.packages('matlib')
## library("matlib")
## print(inv(C))

##### R Functions for Probability Distributions #####

# Every distribution that R handles has four functions. There is a root name, for example, 
# the root name for the normal distribution is norm. This root is prefixed by one of the letters

# p for "probability", the cumulative distribution function (c. d. f.)
# q for "quantile", the inverse c. d. f.
# d for "density", the density function (p. f. or p. d. f.)
# r for "random", a random variable having the specified distribution
# For the normal distribution, these functions are pnorm, qnorm, dnorm, and rnorm. 
# For the binomial distribution, these functions are pbinom, qbinom, dbinom, and rbinom. And so forth.
# R has functions to handle many probability distributions like, Beta, Cauchy, Gamma, Poisson, etc.. 

#### Example of Normal Distribution ##############

# Direct Look-Up
# pnorm is the R function that calculates the c. d. f.
# F(x) = P(X <= x) where X is normal. 

print(pnorm(27.4, 50, 20)) # Here it look up P(X < 27.4) when X is normal with mean 50 and standard deviation 20.

# Inverse Look-Up

# qnorm is the R function that calculates the inverse c. d. f. F-1 of the normal distribution The c. d. f. and the inverse c. d. f. are related by

# p = F(x)
# x = F-1(p)

# So given a number p between zero and one, qnorm looks up the p-th quantile of the normal distribution.

# Q: What is F^(-1)(0.95) when X has the N(100, 15^2) distribution?

print(qnorm(0.95, mean=100, sd=15))

### Random Variates

# rnorm is the R function that simulates random variates having a specified normal distribution. 
# As with pnorm, qnorm, and dnorm, optional arguments specify the mean and standard deviation of the distribution.

x <- rnorm(100, mean=10, sd=5)
print(x)

###### below it plots the histogram of the above 100 random points generated from normal distribution with mean=10 and sd=5

print(hist(x, probability=TRUE)) ### hist plots Histogram ###

### Home Task: Do the same for other distributions for hands on study 

##### For any Functional Help write the following #####

# ?rnorm()
# ?pnorm()
# ?dnorm()
# ?qnorm()

################## Descriptive Statistics #####################

# The monthly credit card expenses of an individual in 1000 rupees is 
# given in the file Credit_Card_Expenses.csv.
# Q1. Read the dataset
# Q2. Compute mean, median minimum, maximum, range, variance, standard
# deviation, skewness, kurtosis and quantiles of Credit Card Expenses
# Q3. Compute default summary of Credit Card Expenses
# Q4. Draw Histogram of Credit Card Expenses


#### read the csv file using read.csv() function 

Credit_Card_Expenses <- read.csv("Credit_Card_Expenses.csv")
# Credit_Card_Expenses <- read.csv("C:/Users/TANUJIT/Dropbox/Teaching/Data Analytics (IIFT)/My Lectures/MBA (IB) 2020 - 2022/data/Credit_Card_Expenses.csv")
Credit_Card_Expenses
mydata = Credit_Card_Expenses ### load it to another variable
print(mydata) ### print the data frame


### To read a particular column or variable of data set to a new variable Example: 
### Read CC_Expenses to CC

CC=mydata$CC_Expenses
print(CC)

######  Descriptive statistics for variable #####

Mean = mean(CC)
print(Mean)
Median=median(CC)
print(Median)
StandaradDeviation=sd(CC)
print(StandaradDeviation)
Variance=var(CC)
print(Variance)
Minimum=min(CC)
print(Minimum)
Maximum=max(CC)
print(Maximum)
Range=range(CC)
print(Range)
Quantile=quantile(CC)
print(Quantile)

Summary=summary(CC)
print(Summary)

##### Another way to calculate Descriptive statistics #####

# install.packages("psych")
library('psych')
data_descriptive=describe(CC)
print(data_descriptive)

#####  Plotting of the Descriptive Statistics #####

hist(CC)
hist(CC,col="blue")
dotchart(CC)
boxplot(CC)
boxplot(CC, col="dark green")

##### plot function : ggplot2 #####

#install.packages("ggplot2")
# Library Call (for use)
library("ggplot2")

################### Data Visualization techniques ################

# data_mart=read.csv("C:/Users/TANUJIT/Dropbox/Teaching/Data Analytics (IIFT)/My Lectures/MBA (IB) 2020 - 2022/data/Big_Mart_Dataset.csv")
data_mart = read.csv("Big_Mart_Dataset.csv")
print(data_mart) # Read and Print BigMart dataset

library(ggplot2) # Scatter Plot of Item_Visibility & Item_MRP 
print(ggplot(data_mart, aes(Item_Visibility, Item_MRP)) + geom_point() +
        scale_x_continuous("Item Visibility", breaks = seq(0,0.35,0.05))+
        scale_y_continuous("Item MRP", breaks = seq(0,270,by = 30))+ theme_bw())


# Now, we can view a third variable also in same chart, say a categorical variable (Item_Type) which will give the characteristic (item_type)
# of each data set. Different categories are depicted by way of different color for
# item_type in below chart. Another scatter plot using function ggplot() with geom_point().


print(ggplot(data_mart, aes(Item_Visibility, Item_MRP)) + 
        geom_point(aes(color = Item_Type)) + scale_x_continuous("Item Visibility", breaks = seq(0,0.35,0.05)) + 
        scale_y_continuous("Item MRP", breaks = seq(0,270,by = 30))+ theme_bw() + 
        labs(title="Scatterplot"))

# We can even make it more visually clear by creating separate
# scatter plots for each separate Item_Type as shown below.
# Another scatter plot using function ggplot() with geom_point().

print(ggplot(data_mart, aes(Item_Visibility, Item_MRP)) + 
        geom_point(aes(color = Item_Type)) + 
        scale_x_continuous("Item Visibility", breaks = seq(0,0.35,0.05)) + 
        scale_y_continuous("Item MRP", breaks = seq(0,270, by = 30))+ 
        theme_bw() + labs(title="Scatterplot") + facet_wrap( ~ Item_Type))

########## Histogram Plot ########################

# For Big_Mart_Dataset, if we want to know the count of items on basis of their
# cost, then we can plot histogram using continuous variable Item_MRP as shown below.
# Histogram plot using function ggplot() with geom_ histogram()

print(ggplot(data_mart, aes(Item_MRP)) + geom_histogram(binwidth = 2)+
        scale_x_continuous("Item MRP", breaks = seq(0,270,by = 30))+
        scale_y_continuous("Count", breaks = seq(0,200,by = 20))+ labs(title = "Histogram"))


##############  Bar Chart Plot ###############################

# For Big_Mart_Dataset, if we want to know item weights (continuous variable)
# on basis of Outlet Type (categorical variable) on single bar chart as shown below.
# Vertical Bar plot using function ggplot()

print(ggplot(data_mart, aes(Item_Type, Item_Weight)) + geom_bar(stat = "identity", fill =
        "darkblue") + scale_x_discrete("Outlet Type")+ 
        scale_y_continuous("Item Weight", breaks = seq(0,15000, by = 500))+ 
        theme(axis.text.x = element_text(angle = 90, vjust = 0.5)) + 
        labs(title = "Bar Chart"))

##################  Stack Bar Chart #########################

# For Big_Mart_Dataset, if we want to know the count of outlets on basis of
# categorical variables like its type (Outlet Type) and location (Outlet Location
# Type) both, stack chart will visualize the scenario in most useful manner.
# Stack Bar Chart using function ggplot()

print(ggplot(data_mart, aes(Outlet_Location_Type, fill = Outlet_Type)) +
        geom_bar()+labs(title = "Stacked Bar Chart", x = "Outlet Location Type", y =
                          "Count of Outlets"))

############ Box Plot  ##########################################

# For Big_Mart_Dataset, if we want to identify each outlet's detailed item sales
# including minimum, maximum & median numbers, box plot can be helpful. In
# addition, it also gives values of outlier of item sales for each outlet as shown
# in below chart.

print(ggplot(data_mart, aes(Outlet_Identifier, Item_Outlet_Sales)) + 
        geom_boxplot(fill = "red")+ 
        scale_y_continuous("Item Outlet Sales", breaks= seq(0,15000, by=500))+
        labs(title = "Box Plot", x = "Outlet Identifier"))

### To save these charts, click on Export - Save as ... ###

##################### Area Chart ####################################

# For Big_Mart_Data set, when we want to analyze the trend of item outlet sales,
# area chart can be plotted as shown below. It shows count of outlets on basis of sales.

print(ggplot(data_mart, aes(Item_Outlet_Sales)) + 
        geom_area(stat = "bin", bins = 30, fill = "steelblue") + 
        scale_x_continuous(breaks = seq(0,11000,1000))+ 
        labs(title = "Area Chart", x = "Item Outlet Sales", y = "Count"))

# Area chart shows continuity of Item Outlet Sales.


######################  Heat Map: ############################################

# For Big_Mart_Dataset, if we want to know cost of each item on every outlet,
# we can plot heatmap as shown below using three variables Item MRP, Outlet
# Identifier & Item Type from our mart dataset

print(ggplot(data_mart, aes(Outlet_Identifier, Item_Type))+ 
        geom_raster(aes(fill = Item_MRP))+ 
        labs(title ="Heat Map", x = "Outlet Identifier", y = "Item Type")+
        scale_fill_continuous(name = "Item MRP"))

# The dark portion indicates Item MRP is close 50. The brighter portion indicates
# Item MRP is close to 250.

##################### Correlogram  ##########################

# For Big_Mart_Dataset, check co-relation between Item cost, weight, visibility
# along with Outlet establishment year and Outlet sales from below plot.
# install.packages("corrgram")

install.packages("corrgram")
library(corrgram)
print(corrgram(data_mart, order=NULL, panel=panel.shade, text.panel=panel.txt,
               main="Correlogram"))

# Darker the color, higher the co-relation between variables. Positive co-
# relations are displayed in blue and negative correlations in red color. Color
# intensity is proportional to the co-relation value.

# We can see that Item cost & Outlet sales are positively correlated while Item
# weight & its visibility are negatively correlated.


################# Data Pre-processing ##################

### Use Import Dataset - From Text (base) - Missing_Values_Telecom - Heading - Yes ###

mydata = Missing_Values_Telecom
newdata = na.omit(mydata) # Discard all records with missing values
write.csv(newdata,"C:/Users/TANUJIT/Dropbox/newdata.csv") # save the new data
write.csv(newdata, file = 'newdata.csv',row.names = TRUE)

### Compute the means excluding the missing values
cmusage = mydata[,2] 
l3musage = mydata[,3] 
avrecharge = mydata[,4]
cmusage_mean = mean(cmusage, na.rm = TRUE) 
l3musage_mean = mean(l3musage, na.rm = TRUE)
avrecharge_mean = mean(avrecharge, na.rm = TRUE)

### Replace the missing values with mean
cmusage[is.na(cmusage)]=cmusage_mean
l3musage[is.na(l3musage)]= l3musage_mean 
avrecharge[is.na(avrecharge)]=avrecharge_mean

mynewdata = cbind(cmusage, l3musage, avrecharge, mydata[,5],mydata[,6]) 
mydata
mynewdata
write.csv(mynewdata, file = 'Missing_Values_Telecom_mod.csv',row.names = TRUE)

############## Data Normalization and Random Sampling #################

mydata = read.csv("Supply_Chain.csv")
# mydata = Supply_Chain
mystddata = scale(mydata)
mystddata

mydata= bank_data
nrow(mydata)
sample = sample(2, nrow(mydata), replace = TRUE, prob = c(0.750, 0.250))
sample1 = mydata[sample ==1,]
nrow(sample1)
sample2 = mydata[sample ==2,]
nrow(sample2)

################################################################################

