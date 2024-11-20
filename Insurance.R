params <-
list(number_of_test_sets = 1L, testing_set_percentage = 10L, 
    dataset_percentage = 100L)

## ----setup, include=FALSE-------------------------------------------------------------------------------------------------------------------------------------
start_time <- Sys.time()
knitr::opts_chunk$set(echo = FALSE)
knitr::opts_chunk$set(warning=FALSE)
#
# URL of source
# file1 <- "https://github.com/ehodgdon/Data_Science/raw/refs/heads/main/insurance-train.csv"
# to read
# dat <- read.csv(url(file1), header=TRUE)


## ----initialization of global variables-----------------------------------------------------------------------------------------------------------------------
# The '...results' variables need to be referenced as global variable if functions
columns <- c("Model", "Accuracy", "Sensitivity", "Specificity")
knn_results <- data.frame(matrix(nrow = 0, ncol = length(columns)))
rf_results <- data.frame(matrix(nrow = 0, ncol = length(columns)))
glm_results <- data.frame(matrix(nrow = 0, ncol = length(columns)))
ensemble_results <- data.frame(matrix(nrow = 0, ncol = length(columns)))
bayes_results <- data.frame(matrix(nrow = 0, ncol = length(columns)))
summary_table <- data.frame(matrix(nrow = 0, ncol = length(columns)))

colnames(knn_results) <- columns
colnames(rf_results) <- columns
colnames(ensemble_results) <- columns
colnames(glm_results) <- columns
colnames(bayes_results) <- columns
colnames(summary_table) <- columns

train_nzv <- 0

gender_levels <- c("Male", "Female")
vehicle_damage_levels <- c("No", "Yes")
vehicle_age_levels <- c("< 1 Year", "1-2 Year", "> 2 Years")




## ----load libraries, echo=FALSE, include=FALSE----------------------------------------------------------------------------------------------------------------
if (!require(tidyverse)) suppressMessages(install.packages(tidyverse, verbose=FALSE, quiet = TRUE))
if (!require(rvest)) install.packages("rvest", verbose=FALSE)
if (!require(tidyr)) install.packages("tidyr", verbose=FALSE)
if (!require(caret)) install.packages("caret", verbose=FALSE)
if (!require(rlist)) suppressMessages(install.packages("rlist", verbose=FALSE))
if (!require(naivebayes)) install.packages("naivebayes", verbose=FALSE)
if (!require(randomForest)) suppressMessages(install.packages("randomForest", verbose=FALSE))
if (!require(kableExtra)) install.packages("kableExtra", verbose=FALSE)
if (!require(dplyr)) instsall.packages("dplyr", verbose=FALSE)
if (!require(corrplot)) install.packages("corrplot", verbose=FALSE)
if (!require(gridExtra)) install.packages(("gridExtra"))


## ----function definitions, echo = FALSE, include=FALSE--------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# function definitions
# -----------------------------------------------------------------------------
data_prep <- function(df, nzv_arg = FALSE, response = TRUE) {
  # check for any rows containing NAs
  na_counts <- df %>% summarise_all(~ sum(is.na(.)))
  number_of_NAs <<- sum(na_counts)
  if (number_of_NAs > 0 ) {
    df <- na.omit(df)
  }
   # A value of zero indicates there are no NAs in the entire dataset. Next, check of missing data

   # Check for any rows containing blanks
  na_nulls <- sum(is.null(df))
  number_of_nulls <<- sum(na_nulls)
  if (number_of_nulls > 0) {
    df[df == "NULL"] <- NA
    df <- na.omit(df)
    
  }

  # Using the nearZeroVar() function from the caret package,we can determine which predictors are near zero variance and therefore would not be a
  # good predictor.
  ifelse (!nzv_arg,   train_nzv <<- nearZeroVar(raw_data), train_nzv <- nzv_arg)
  if (length(train_nzv) > 0) {
    df <- df[, -train_nzv]  # remove near zero variances
    
  }  # end of if (length(nzv) ...
   
  
  # knn algorithm requires all numerical columns
  # convert the Gender column from "Male", "Female" to 0 and 1
  df$Gender <- ifelse(df$Gender == "Male", 0, 1)

  # convert the Vehicle_Damage column from "Yes" and "No" ti 1 and 0
  df$Vehicle_Damage <- ifelse(df$Vehicle_Damage == "Yes", 1, 0)
  
  
  # convert the Vehicle_Age from a text to a numeric value
  unique <- unique(df$Vehicle_Age)
  for (age in unique) {df$Vehicle_Age[df$Vehicle_Age == age] <- as.integer(which(unique == age))}
  
  #At this point all values in the data frame are numeric
  
  # remove the id column from the set
  df <- df %>% select(-id)


 return (df)
}      # end of data_prep function

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
show_parameters <- function() {
  parameters <- data.frame()
  parameters <- rbind(parameters, c("number_of_test_sets", params$number_of_test_sets))
  parameters <- rbind(parameters, c("testing_set_percentage", params$testing_set_pwercentage))
  parameters <- rbind(parameters, c("dataset_percentage", params$dataset_))
  colnames(parameters) <- c("Parameter", "Value")
  return (parameters)
}

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

add_formatted_row <- function(model, cm) {
  return ( list(model, round(cm$overall["Accuracy"], 4), 
                       round(cm$byClass["Sensitivity"], 4), 
                       round(cm$byClass["Specificity"], 4)))  
}  

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  
knn_fcn <- function(test_set) {
  df <- as.data.frame(test_set)
  test_set_y <- df$Response
  test_set_x <- test_set %>% select(-Response)
  y_hat <- predict(fit_knn, test_set_x, type = "class")
  cm <- confusionMatrix(y_hat, factor(test_set_y))
  knn_results[nrow(knn_results)+1,] <<- add_formatted_row("knn", cm)
  knn_mean <<- mean(knn_results$Accuracy) 
}

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

rf_fcn <- function(test_set) {
  test_set_y <- as.factor(test_set$Response)
  test_set_x <- test_set %>% select(-Response)
  y_hat <- predict(fit_rf, test_set_x, type = "class")
  cm <- confusionMatrix(y_hat, factor(test_set_y))
  rf_results[nrow(rf_results)+1,] <<- list("rf", cm$overall["Accuracy"], cm$byClass["Sensitivity"], cm$byClass["Specificity"])
}

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

ensemble_fcn <- function(test_set) {
  test_set_y <- as.factor(test_set$Response)
  test_set_x <- test_set %>% select(-Response)
  
  p_rf <- predict(fit_rf, test_set_x, type = "prob")
  p_rf <- p_rf / rowSums(p_rf)
  p_knn <- predict(fit_knn, test_set_x)
  p <- (p_rf + p_knn) / 2
  y_pred <- factor(apply(p, 1, which.max)-1)
  cm <- confusionMatrix(y_pred, test_set_y)
  ensemble_results[nrow(ensemble_results)+1,] <<- list('ensemble', cm$overall["Accuracy"], cm$byClass["Sensitivity"], cm$byClass["Specificity"])
}

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

glm_model <- function(dp, fam = gaussian()) {
model <- glm(Response ~ Previously_Insured + Vehicle_Age + Vehicle_Damage + Vehicle_Age, 
                family = fam, data = dp)
summary(model)
y_hat <- predict(model, dp %>% select(-Response), type="response")
cutoff <- quantile(y_hat, 0.18)
pred_glm <- ifelse(y_hat < 0.0006104226, "1", "0")
pred_glm = as.factor(pred_glm)

cm <- confusionMatrix(as.factor(dp$Response), pred_glm) 
glm_results[nrow(glm_results)+1,]  <<- list("glm", cm$overall["Accuracy"], cm$byClass["Sensitivity"], cm$byClass["Specificity"])
}

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

#
# naive Bayes 
#
n_bayes_fcn <- function(test_set, fit) {
  test_set_y <- as.factor(test_set$Response)
  test_set_x <- test_set%>% select(-Response)
  y_hat <- predict(fit, test_set_x)
  cm <- confusionMatrix(y_hat, factor(test_set_y))
  bayes_results[nrow(bayes_results)+1,] <<-list("naiveBayes", cm$overall["Accuracy"], cm$byClass["Sensitivity"], cm$byClass["Specificity"])
  bayes_mean <<- mean(bayes_results$Accuracy)
}

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Function to build accuracy table for display
#
disp_fcn <- function(rslts, title, include_col_1 = FALSE) {

disp_df <- rslts %>%  select(-1)
if (nrow(rslts) > 1) {
  disp_df["MinMax"] <- ''
  index <- which.min(disp_df$Accuracy)
  disp_df[index, "MinMax"] <- "Minimum"
  index <- which.max(disp_df$Accuracy)
  disp_df[index, "MinMax"] <- "Maximum"
  colnames(disp_df) <- c('Accuracy', 'Sensitivity', 'Specificity', 'MinMax')
}

model_accuracy <- mean(disp_df$Accuracy)
disp_df <- disp_df %>% mutate(across(c('Accuracy','Sensitivity', 'Specificity'), round, 4))
disp_df[nrow(disp_df)+1,] <- c("Average", round(model_accuracy, 4), "", "")


knitr::kable(disp_df, digits = 4, format.args = (list(scientific=FALSE)),  
              table.attr = "style='width:70%;' ")  |> kable_styling(bootstrap_options = c("striped"))
}

disp_fcn_glm <- function(rslts, title) {

disp_df <- rslts
if (nrow(rslts) > 1) {
  disp_df["MinMax"] <- ''
  index <- which.min(disp_df$Accuracy)
  disp_df[index, "MinMax"] <- "Minimum"
  index <- which.max(disp_df$Accuracy)
  disp_df[index, "MinMax"] <- "Maximum"
  colnames(disp_df) <- c('Model', 'Accuracy', 'Sensitivity', 'Specificity', 'MinMax')
}

model_accuracy <- mean(disp_df$Accuracy)
disp_df <- disp_df %>% mutate(across(c('Accuracy','Sensitivity', 'Specificity'), round, 4))
disp_df[nrow(disp_df)+1,] <- c("Model", "Average", round(model_accuracy, 4), "", "")


knitr::kable(disp_df, digits = 5, format.args = (list(scientific=FALSE)),  
              table.attr = "style='width:70%;' ")  |> kable_styling(bootstrap_options = c("striped"))
}
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

find_max_value <- function(dp) {
  index <- which.max(dp$Accurcy)
  return (dp[index, ])
}

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

add_summary_row <- function(dp) {
  summary_table[nrow(summary_table)+1,] <- dp[which.max(dp$Accuracy),]
}

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

build_summary_table <- function() {
  summary_table[nrow(summary_table)+1,] <- add_summary_row(knn_results)
  summary_table[nrow(summary_table)+1,] <- add_summary_row(rf_results)
  summary_table[nrow(summary_table)+1,] <- add_summary_row(ensemble_results)
  summary_table[nrow(summary_table)+1,] <- add_summary_row(bayes_results)
  
  # Add all rows from the glm_results since only one result per test was included
  summary_table <- full_join(summary_table, glm_results)
  # put into descending accuracy order
  summary_table <<- summary_table[order(-summary_table$Accuracy),]
 
}

# -----------------------------------------------------------------------------
# end of functions
# -----------------------------------------------------------------------------


## ----read dataset and data prep, echo = FALSE-----------------------------------------------------------------------------------------------------------------
url <- "https://github.com/ehodgdon/Data_Science/raw/refs/heads/main/insurance-train.csv"
raw_data <- read.csv(url, header=TRUE)

# raw_data <- read.csv("insurance-train.csv")
original_size <- nrow(raw_data)
original_cols <- colnames(raw_data)
if (params$dataset_percentage < 100) {
  size <- round(nrow(raw_data) * params$dataset_percentage / 100)
  raw_data <- slice_sample(raw_data, n = size, replace = FALSE)
}
tbl <- matrix(c(1:36), ncol = 3, byrow= TRUE)
rownames <- c("Identification", "Gender", "Age (in years)", "Driver's License Nbr", "Region Code", "Previously Insured", "Insured Vehicles Age", "Vehicle Damage",
             "Annual Premium", "Policy Sales Channel", "Vintage", "Response")
colnames <- c("Name", "Description", "Class")

col3 <- vector()
for (col in 1:ncol(raw_data)) {col3 <- c(col3, class(raw_data[,col]))}
colnames(tbl) <- colnames
rownames(tbl) <- rownames
tbl[,1] <- as.vector(colnames(raw_data))
tbl[,2] <- c("id", "Gender", "Age", "Driving_License", "Region_Code", "Previously_Insured", "Vehicle's_age", "Vehicle_damage", "Annual Premium", "Policy_Sales_Channel",
          "Vintage", "Success")
tbl[,3] <- col3




## ----data prep - build separate datasets, message=FALSE-------------------------------------------------------------------------------------------------------


raw_data <- data_prep(raw_data)
set.seed(2024)

test_sets <- list()
original_data <- raw_data                                                       # save a copy of the original dataset
total_rows <- nrow(raw_data)
testing_set_rows <- (total_rows * params$testing_set_percentage) %/% 100
for (i in 1:params$number_of_test_sets) {
 tset <- slice_sample(raw_data, n = testing_set_rows, replace = FALSE)   # partition off rows for train testing (could be multiple)
 raw_data <- anti_join(raw_data, tset)                                          # remove testing set rows from the original dataset
 test_sets <- list.append(test_sets, tset)                                      # create list of testing sets
 }

training_set <- raw_data                                                        # what is left over is designated the training set
training_set_x <- training_set %>% select(-Response)
training_set_y <- as.factor(training_set$Response)







## ----display column names, echo=FALSE-------------------------------------------------------------------------------------------------------------------------
knitr::kable(tbl, caption = "Column Names in Dataset", align="c")


## ----display nzv columns, fig.align='left'--------------------------------------------------------------------------------------------------------------------
df <- data.frame(cols = original_cols[train_nzv])
knitr::kable(df, caption = "Columns Removed", col.names = NULL, align="l")




## ----plot spearman matrix, fig.width= 4.5, fig.height = 5, out.width = ".7\\textwidth",  fig.align="center", wrapfigure = list("R", .7), results="asis"-------
cor_matrix <- training_set %>% mutate(Vehicle_Age = as.integer(Vehicle_Age)) %>% cor(method="spearman")
corrplot <-  corrplot(cor_matrix,
             method="color",
             type="full",
             order="original",
             tl.cex = 0.7,
             tl.col= "black",
             tl.srt = 45,
             number.digits = 1,
             cl.pos = "n",
             mar = c(0,0,2,0),
             addCoef.col = "black",
             title="Spearman Correlation Map")


## ----previously_insured, echo=TRUE----------------------------------------------------------------------------------------------------------------------------
num <- sum(training_set$Previously_Insured == 0 & training_set$Response == TRUE)
num


## ----response mean, fig.align="center", echo=FALSE, error=FALSE, fig.width = 7, fig.height= 3.25--------------------------------------------------------------
response_mean <- mean(training_set$Response)
plot_df <- data.frame(c("negative", "positive"), c(100 * (1-response_mean), 100 * response_mean))
colnames(plot_df) <- c("response","percent")

response_plot_1 <- ggplot(data=plot_df) + 
  geom_bar(mapping = aes(x=response, y=percent, fill=response), binwidth= 0.5, show.legend=FALSE, stat="identity") +
    geom_text(color="white", size=3.5, aes(x=response, y=percent, label=round(percent), vjust=1.5)) +

  ylim(c(0, 100)) +  
  ggtitle("Overall Responses") 

temp_df <- training_set[training_set$Previously_Insured == 0,]
plot1_df <- data.frame(c("negative", "positive response"), c(100 * sum(temp_df$Response == 0)/nrow(temp_df), 
                                                             100 * sum(temp_df$Response == 1)/nrow(temp_df)))
colnames(plot1_df) <- c("response", "percent")
response_plot_2 <-ggplot(data=plot1_df) + 
  geom_bar(mapping = aes(x=response, y=percent, fill=response), binwidth= 0.5, show.legend=FALSE, stat="identity") +
  ylim(c(0,100)) +
  geom_text(color="white", size=3.5, aes(x=response, y=percent, label=round(percent), vjust=1.5)) +
  ggtitle("No Pre Insured Responses")
grid.arrange(response_plot_1, response_plot_2, ncol = 2, padding=10)


## ----first set of training, error=TRUE, message = FALSE, include=FALSE----------------------------------------------------------------------------------------
try({
k_hist <- list()
sub_set <- slice_sample(training_set, n = 100000, replace = FALSE) 
sub_set_x <- sub_set %>% select(-Response)
sub_set_y <- sub_set$Response

control <- trainControl(method = "cv", number = 5, p = .9)
train_knn1 <- train(sub_set_x, as.factor(sub_set_y), method="knn", tuneGrid = data.frame(k = c(seq(3,9,2),20)), trControl = control)

df_knn1 <- train_knn1$results     # extract results as a data frame
k_hist <- list.append(k_hist, df_knn1)

max_accuracy <- which.min(df_knn1$Accuracy)
max_k <- df_knn1$k[max_accuracy]

a <- max_k - 2
b <- max_k + 2
if (a < 1) a <- 1

# range <- seq(a,b)

})


## ----second set of training, echo = FALSE, include = FALSE----------------------------------------------------------------------------------------------------

train_knn2 <- train(sub_set_x, as.factor(sub_set_y), method="knn", tuneGrid = data.frame(k = seq(a,b)), trControl = control)

df_knn2 <- train_knn2$results
max_accuracy <- which.max(df_knn2$Accuracy)
max_k <- df_knn2$k[max_accuracy]

df_knn <- merge(df_knn1, df_knn2, by=c("Accuracy","k", "Kappa", "AccuracySD", "KappaSD"), all=TRUE)


## ----graghing value of k, echo=FALSE, include=FALSE, fig.align = "center", fig.height = 4, fig.width = 5------------------------------------------------------
# The first two training sessions are to determine a good value of k and are done with a reduced dataset size and the number of folds set to 5
all_k <- full_join(df_knn1, df_knn2)

## ----display k graph, fig.width = 6, fig.height=3-------------------------------------------------------------------------------------------------------------
sorted_k <- all_k[order(all_k$k),]
ggplot(sorted_k, aes(x=k, y=Accuracy))+ geom_line()
max_k <- which.max(sorted_k$Accuracy)




## ----final train - takes a while, echo=FALSE------------------------------------------------------------------------------------------------------------------
control <- trainControl(method = "cv", number = 10, p = .9)
train_knn <- train(training_set_x, training_set_y, method="knn", tuneGrid = data.frame(k = max_k), trControl = control)

fit_knn <- knn3(training_set_x, training_set_y, k=max_k)



## ----test_sets, echo=FALSE, error=FALSE, include = FALSE------------------------------------------------------------------------------------------------------
lapply(test_sets, knn_fcn)




## ----display knn results, echo=FALSE--------------------------------------------------------------------------------------------------------------------------

disp_fcn(knn_results, "Summary of k-NN testing")
knn_mean <- mean(knn_results$Accuracy)



## ----random forest - takes a while also, echo = FALSE, error = FALSE, include=FALSE---------------------------------------------------------------------------

control <- trainControl(method = "cv", number = 5)
grid <- data.frame(mtry = c(1,2,3))
train_rf <- train(training_set_x, training_set_y, method="rf", 
                  ntree = 500, 
                  sampsize = 100000,
                  metric = "Accuracy",
                  importance = TRUE, 
                  trControl = control, tuneGrid = grid, nSamp = 10000)
fit_rf <- randomForest(training_set_x, training_set_y, mtry = train_rf$bestTune$mtry)
y_hat_rf <- predict(fit_rf, training_set_x, type="class")
cm <- confusionMatrix(y_hat_rf, factor(training_set_y))

lapply(test_sets, rf_fcn)


## ----display rf resultsa,echo = FALSE-------------------------------------------------------------------------------------------------------------------------
disp_fcn(rf_results, "Summary of random forest  testing")
rf_mean <- mean(rf_results$Accuracy)



## ----graph importance, echo=FALSE, fig.align = "center"-------------------------------------------------------------------------------------------------------
importance <- fit_rf$importance
colnames(importance) <- c("Mean")
sorted_importance <- as.data.frame(importance[order(importance[,"Mean"]),]) * 100/ sum(importance)
colnames(sorted_importance) <- c("sorted_importance")


xlabls <- gsub("_", " ", rownames(sorted_importance))
ggplot(data=sorted_importance, aes(x = reorder(xlabls, -sorted_importance), y=sorted_importance, fill=reorder(xlabls, -sorted_importance))) + 
  geom_bar(show.legend=FALSE, stat="identity") +
  xlab("Feature") +
  ylab("Percent") +
  ylim(0, 40) +
  ggtitle("Relative Importance") +
  theme(plot.title = element_text(hjust = 0.5, size = 9)) +
  guides(x = guide_axis(angle = 90))   



## ----ensemble, echo=FALSE, include=FALSE----------------------------------------------------------------------------------------------------------------------
lapply(test_sets, ensemble_fcn)
ensemble_mean <- mean(ensemble_results$Accuracy)


## ----display ensemble results, echo=FALSE---------------------------------------------------------------------------------------------------------------------
disp_fcn(ensemble_results, "Summary of ensemble testing")


## ----naive_bayes model testing, echo = FALSE, include = FALSE-------------------------------------------------------------------------------------------------
control <- trainControl(method = "cv", number = 5)
grid <- data.frame(mtry = c(1,2,3))
train_nb <- train(as.factor(Response)~ Vehicle_Damage + Previously_Insured + Age + Vehicle_Age, method = "naive_bayes", data=training_set, usepoisson = TRUE)
lapply(test_sets, n_bayes_fcn, train_nb)


## ----display bayes results, echo = FALSE----------------------------------------------------------------------------------------------------------------------
disp_fcn(bayes_results, "Summary of naive Bayes mode")


## ----glm test cases, echo = FALSE-----------------------------------------------------------------------------------------------------------------------------
glm_model(training_set, binomial(link=logit))
glm_results[[nrow(glm_results),1]] <- paste(glm_results[[nrow(glm_results),1]],  "(binomial)")

glm_model(training_set, gaussian())
glm_results[[nrow(glm_results),1]] <- paste(glm_results[[nrow(glm_results),1]],  "(gaussian)")

glm_model(training_set, quasi(link = "identity", variance ="constant"))
glm_results[[nrow(glm_results),1]] <- paste(glm_results[[nrow(glm_results),1]],  "(quasi)")

glm_model(training_set, quasibinomial(link = "logit"))
glm_results[[nrow(glm_results),1]] <- paste(glm_results[[nrow(glm_results),1]],  "(quasibinomial)")
disp_fcn_glm(glm_results, "Summary of several glm families model")



## ----creat summary table, echo=FALSE, include = FALSE---------------------------------------------------------------------------------------------------------

summary_table <- build_summary_table()


## ----display summary list, echo = FALSE-----------------------------------------------------------------------------------------------------------------------
knitr::kable(summary_table,  row.names = FALSE, digits = 4)


## ----find best model------------------------------------------------------------------------------------------------------------------------------------------



## ----show parameters, echo=FALSE------------------------------------------------------------------------------------------------------------------------------
show_parameters() %>% 
  kbl(longtable = TRUE) %>%
  kable_styling(full_width = FALSE, position="left", latex_options = "hold_position")


## ----stop time, echo=FALSE------------------------------------------------------------------------------------------------------------------------------------
print(paste('Elapsed time for this analysis:', format(as.numeric(difftime(Sys.time(), start_time, units="mins")), digits=5), "mins"))
print(paste('Finished at:', format(Sys.time(), "%R %p")))

