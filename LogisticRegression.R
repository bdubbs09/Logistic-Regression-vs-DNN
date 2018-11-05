# Splitting the data frame into testing and training sets
data<-load("/Users/Brandon/Desktop/spam.RData") 
attach(data)

install.packages("ggplot2", "usdm", "tensorflow", "tfestimators")

library(ggplot2)
library(usdm)
library(tensorflow)
library(tfestimators)

# Plotting the correlation matrix. We should really take a close look at this
ggcorr(spambase)
# Checking the vif and multicolinearity
vif(spambase)
# setting the threshhold to 5 and stepping though
vifstep(spambase, th = 5)

# Dropping the columns that are above the threshold
drop <- c("word_freq_857","word_freq_415")
data <- spambase[,!(names(spambase) %in% drop)]

# Splitting test and training set
indexes = sample(1:nrow(data), size=0.2*nrow(data))
test = data[indexes,]
dim(test)
train = data[-indexes,]
dim(train)

# initial logistic regression model
model <- glm (Spam ~ ., data = train, family = binomial)
summary(model)

pi <- predict(model, newdata=test,type= 'response')
pred_logistic <- rep('0', nrow(test))
pred_logistic[pi > .5] <- '1'
actual_outcome <- test$Spam
tab_logistic <- table(actual_outcome, pred_logistic)
tab_logistic

# Machine Learning Approach
# Creating a simple feed forward neural net
row_indices <- sample(1:nrow(data), size = 0.8 * nrow(data))
neural_train <- data[row_indices, ]
neural_test <- data[-row_indices, ]

# Creating feature column for features that are goine to be used in the neural net
# There has to be a better way to do this, but Tensorflow is being really picky.
feature_columns <- feature_columns(column_numeric("word_freq_make"), column_numeric("word_freq_address"),
                                   column_numeric("word_freq_all"), column_numeric("word_freq_3d"),
                                   column_numeric("word_freq_our"), column_numeric("word_freq_over"),
                                   column_numeric("word_freq_remove"), column_numeric("word_freq_internet"),
                                   column_numeric("word_freq_order"), column_numeric("word_freq_mail"),
                                   column_numeric("word_freq_receive"), column_numeric("word_freq_will"),
                                   column_numeric("word_freq_people"), column_numeric("word_freq_report"),
                                   column_numeric("word_freq_addresses"), column_numeric("word_freq_free"),
                                   column_numeric("word_freq_business"), column_numeric("word_freq_email"),
                                   column_numeric("word_freq_you"), column_numeric("word_freq_credit"),
                                   column_numeric("word_freq_your"), column_numeric("word_freq_font"),
                                   column_numeric("word_freq_000"), column_numeric("word_freq_money"),
                                   column_numeric("word_freq_hp"), column_numeric("word_freq_hpl"),
                                   column_numeric("word_freq_george"), column_numeric("word_freq_650"),
                                   column_numeric("word_freq_lab"), column_numeric("word_freq_labs"),
                                   column_numeric("word_freq_telnet"), #column_numeric("word_freq_857"),
                                   column_numeric("word_freq_data"), #column_numeric("word_freq_415"),
                                   column_numeric("word_freq_85"), column_numeric("word_freq_technology"),
                                   column_numeric("word_freq_1999"), column_numeric("word_freq_parts"),
                                   column_numeric("word_freq_pm"), column_numeric("word_freq_direct"),
                                   column_numeric("word_freq_cs"), column_numeric("word_freq_meeting"),
                                   column_numeric("word_freq_original"), column_numeric("word_freq_project"),
                                   column_numeric("word_freq_re"), column_numeric("word_freq_edu"),
                                   column_numeric("word_freq_table"), column_numeric("word_freq_conference"),
                                   column_numeric("char_freq_semicol"), column_numeric("char_freq_."),
                                   column_numeric("char_freq_..1"), column_numeric("char_freq_..2"),
                                   column_numeric("char_freq_..3"), column_numeric("char_freq_..4"),
                                   column_numeric("capital_run_length_average"), column_numeric("capital_run_length_longest"),
                                   column_numeric("capital_run_length_total"))


# Creating inputs with features and response variables.
# This is easiest as a function rather than just doing this line by line.
# The input function will go over the training set 10 times (epochs).
neural_prediction_function <- function(data) {
  input_fn(data, 
           features = c("word_freq_make", "word_freq_address","word_freq_all","word_freq_3d",
                        "word_freq_our","word_freq_over","word_freq_remove","word_freq_internet",
                        "word_freq_order", "word_freq_mail","word_freq_receive", "word_freq_will",
                        "word_freq_people", "word_freq_report","word_freq_addresses", "word_freq_free",
                        "word_freq_business","word_freq_email","word_freq_you", "word_freq_credit",
                        "word_freq_your", "word_freq_font","word_freq_000", "word_freq_money",
                        "word_freq_hp", "word_freq_hpl","word_freq_george","word_freq_650",
                        "word_freq_lab", "word_freq_labs","word_freq_telnet", #"word_freq_857",
                        "word_freq_data", #"word_freq_415",
                        "word_freq_85", "word_freq_technology",
                        "word_freq_1999", "word_freq_parts","word_freq_pm", "word_freq_direct",
                        "word_freq_cs","word_freq_meeting","word_freq_original","word_freq_project",
                        "word_freq_re","word_freq_edu","word_freq_table","word_freq_conference",
                        "char_freq_semicol", "char_freq_.","char_freq_..1", "char_freq_..2",
                        "char_freq_..3", "char_freq_..4","capital_run_length_average",
                        "capital_run_length_longest","capital_run_length_total"), 
           
           response = "Spam", 
           num_epochs = 10)
  
}

# Creating deep neural network classifier.
# Features hidden units (nodes), features that are used
# as well as the number of label classes.
classifier <- dnn_classifier(
  feature_columns = feature_columns, 
  hidden_units = c(100, 50, 40), 
  n_classes = 2)

# Training the classifier
train(classifier, input_fn = neural_prediction_function(neural_train))

# Making predictions
predictions_test <- predict(classifier, input_fn = neural_prediction_function(neural_test))
predictions_overall <- predict(classifier, input_fn = neural_prediction_function(data))

# Evaluating the models and their metrics
evaluation_test <- evaluate(classifier, input_fn = neural_prediction_function(neural_test))
evaluation_overall <- evaluate(classifier, input_fn = neural_prediction_function(data))

# Showing metrics
glimpse(evaluation_test)
glimpse(evaluation_overall)
