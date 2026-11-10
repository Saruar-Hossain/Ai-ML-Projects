# Load the BanglaRQA dataset
data <- read.csv("E:/AAA-Summer26/Datascience/BanglaRQA_All_14889_QA_Pairs.csv")


# View the dataset in RStudio
data
View(data)


# Check the number of rows and columns in the dataset
dim(data)


# Keep only answerable QA pairs for hallucination detection
answerable_data <- data[data$is_answerable == 1, ]


# Check the size of the filtered dataset
dim(answerable_data)


# Check the distribution of question types
table(answerable_data$question_type)
# Set seed to make the random sampling reproducible
set.seed(123)


# Select samples from each question type according to the original distribution
factoid_sample <- answerable_data[answerable_data$question_type == "factoid", ][sample(nrow(answerable_data[answerable_data$question_type == "factoid", ]), 3440), ]

list_sample <- answerable_data[answerable_data$question_type == "list", ][sample(nrow(answerable_data[answerable_data$question_type == "list", ]), 585), ]

confirmation_sample <- answerable_data[answerable_data$question_type == "confirmation", ][sample(nrow(answerable_data[answerable_data$question_type == "confirmation", ]), 560), ]

causal_sample <- answerable_data[answerable_data$question_type == "causal", ][sample(nrow(answerable_data[answerable_data$question_type == "causal", ]), 415), ]


# Combine all selected samples into one dataset
sample_5000 <- rbind(
  factoid_sample,
  list_sample,
  confirmation_sample,
  causal_sample
)


# Shuffle the selected samples to mix different question types
sample_5000 <- sample_5000[sample(nrow(sample_5000)), ]


# Check the final number of selected QA pairs
dim(sample_5000)


# Check the question type distribution after sampling
table(sample_5000$question_type)

# Keep only the columns needed for LLM response generation
before_llm_dataset <- sample_5000[, c(
  "context",
  "question_text",
  "answer_text"
)]

dim(before_llm_dataset)

colnames(before_llm_dataset)


View(before_llm_dataset)

data <- read.csv("E:/AAA-Summer26/Datascience/Final Project/BanglaRQA_Final_5000_final_clean_v4_with_pair_id (1).csv")
View(data)
# Check number of rows and columns
dim(data)

# Count how many rows exist for each pair_id
pair_count <- table(data$pair_id)

# View frequency distribution
table(pair_count)

# Find problematic pair_ids
problem_pairs <- names(pair_count[pair_count != 2])

problem_pairs
# Check Faithful vs Hallucinated count
table(data$label)

# Check dataset structure

str(data)
# Check missing values in each column
colSums(is.na(data))



# Check duplicate rows
sum(duplicated(data))

# Remove answers having both Faithful and Hallucinated labels
answer_label_check <- aggregate(
  label ~ llm_answer,
  data = data,
  FUN = function(x) paste(unique(x), collapse=",")
)

#1 1 1 1 0 0
# 1  0
# 1,0



# Identify conflicting answers
bad_answers <- answer_label_check$llm_answer[
  grepl(",", answer_label_check$label)
]

# Remove ambiguous samples
clean_data <- data[
  !(data$llm_answer %in% bad_answers),
]


# Keep only complete QA pairs
pair_count <- table(clean_data$pair_id)

complete_pairs <- names(pair_count[pair_count == 2])

dataset_B <- clean_data[
  clean_data$pair_id %in% complete_pairs,
]


# Check final dataset size (rows and columns)
dim(dataset_B)

# Check class distribution (0 = Faithful, 1 = Hallucinated)
table(dataset_B$label)

# Check whether any complete duplicate rows exist
sum(duplicated(dataset_B))

# Verify that each pair_id contains one Faithful and one Hallucinated response
pair_label_check <- aggregate(
  label ~ pair_id,
  data = dataset_B,
  FUN = function(x) paste(sort(x), collapse=",")
)

# Check  how many pairs has (0,1) both values
table(pair_label_check$label)
colSums(is.na(dataset_B))



# PHASE 3: TEXT PREPROCESSING


# Load required libraries

library(stringr)
library(stringi)


# ==========================================
# Step 1: Combine context + question + LLM answer
# ==========================================

# Create one text input for the ML model
# The model needs:
# Reference context + Question + Generated answer
#
# We exclude answer_text because it is the human reference answer
# and including it may make the task unrealistically easy.

dataset_B$text <- paste(
  dataset_B$context,
  dataset_B$question_text,
  dataset_B$llm_answer,
  sep = " "
)


# Check combined text

head(dataset_B$text)



# ==========================================
# Step 2: Create Bangla text cleaning function
# ==========================================

clean_bangla_text <- function(text){
  
  # Unicode normalization
  # Makes different Unicode forms of the same Bangla word identical
  text <- stringi::stri_trans_nfc(text)
  
  
  # Remove punctuation and symbols
  text <- str_replace_all(
    text,
    "[[:punct:]]",
    " "
  )
  
  
  # Remove extra spaces
  text <- str_replace_all(
    text,
    "\\s+",
    " "
  )
  
  
  # Remove spaces from beginning and end
  text <- str_trim(text)
  
  
  return(text)
  
}



# ==========================================
# Step 3: Apply cleaning function
# ==========================================

dataset_B$clean_text <- sapply(
  dataset_B$text,
  clean_bangla_text
)



# ==========================================
# Step 4: Compare original and cleaned text
# ==========================================

# Original combined text

dataset_B$text[1]


# Cleaned text

dataset_B$clean_text[1]






# ==========================================
# Step 5: Check missing values after preprocessing
# ==========================================

sum(is.na(dataset_B$clean_text))


# ==========================================
# PHASE 4: TRAIN-TEST SPLIT
# ==========================================


# Set seed for reproducible split
set.seed(123)


# Extract unique pair IDs
# Each pair_id contains:
# 1 Faithful response + 1 Hallucinated response

unique_pairs <- unique(dataset_B$pair_id)



# Split 80% pair IDs for training

train_pairs <- sample(
  unique_pairs,
  size = floor(0.8 * length(unique_pairs))
)



# Remaining 20% pair IDs for testing

test_pairs <- setdiff(
  unique_pairs,
  train_pairs
)



# Create training dataset

train_data <- dataset_B[
  dataset_B$pair_id %in% train_pairs,
]



# Create testing dataset

test_data <- dataset_B[
  dataset_B$pair_id %in% test_pairs,
]



# ==========================================
# Verify split
# ==========================================


# Check number of rows and columns

dim(train_data)

dim(test_data)



# Check class distribution
# 0 = Faithful, 1 = Hallucinated

table(train_data$label)

table(test_data$label)



# Check every pair still has two responses

table(table(train_data$pair_id))

table(table(test_data$pair_id))



# Check no pair_id leakage between train and test
# Expected result = 0

length(intersect(
  train_data$pair_id,
  test_data$pair_id
))



# Prepare text and labels for next phase (BoW / TF-IDF)

X_train <- train_data$clean_text

y_train <- train_data$label


X_test <- test_data$clean_text

y_test <- test_data$label



# ==========================================
# PHASE 5: FEATURE REPRESENTATION
# BoW and TF-IDF
# ==========================================


# Load required libraries

library(text2vec)
library(Matrix)



# ==========================================
# Step 1: Prepare input text and labels
# ==========================================

# Input text

X_train_text <- train_data$clean_text

X_test_text <- test_data$clean_text


# Target labels
# 0 = Faithful
# 1 = Hallucinated

y_train <- train_data$label

y_test <- test_data$label



install.packages("text2vec")
install.packages("Matrix")
library(text2vec)
library(Matrix)

# ==========================================
# PART A: BAG OF WORDS (BoW)
# ==========================================


# Create vocabulary ONLY from training data
# Test data is not used to avoid data leakage

vocab_bow <- create_vocabulary(
  itoken(
    X_train_text,
    tokenizer = word_tokenizer
  )
)



# Convert vocabulary into vectorizer

vectorizer_bow <- vocab_vectorizer(
  vocab_bow
)



# Convert training text into BoW vectors

X_train_bow <- create_dtm(
  itoken(
    X_train_text,
    tokenizer = word_tokenizer
  ),
  vectorizer_bow
)



# Convert test text using SAME training vocabulary

X_test_bow <- create_dtm(
  itoken(
    X_test_text,
    tokenizer = word_tokenizer
  ),
  vectorizer_bow
)



# Check BoW dimensions

dim(X_train_bow)

dim(X_test_bow)





# ==========================================
# PART B: TF-IDF
# ==========================================

# ==========================================
# PART B: TF-IDF
# ==========================================


# Create vocabulary from training data only

vocab_tfidf <- create_vocabulary(
  itoken(
    X_train_text,
    tokenizer = word_tokenizer
  )
)


# Convert vocabulary into vectorizer

vectorizer_tfidf <- vocab_vectorizer(
  vocab_tfidf
)


# Create training document-term matrix

train_dtm <- create_dtm(
  itoken(
    X_train_text,
    tokenizer = word_tokenizer
  ),
  vectorizer_tfidf
)


# Create TF-IDF model

tfidf <- TfIdf$new()


# Fit TF-IDF on training data
# and transform training text

X_train_tfidf <- tfidf$fit_transform(
  train_dtm
)



# Create test document-term matrix
# using the same training vocabulary

test_dtm <- create_dtm(
  itoken(
    X_test_text,
    tokenizer = word_tokenizer
  ),
  vectorizer_tfidf
)



# Transform test using fitted TF-IDF

X_test_tfidf <- tfidf$transform(
  test_dtm
)



# Check dimensions

dim(X_train_tfidf)

dim(X_test_tfidf)


# Check labels match the number of rows

length(y_train)

length(y_test)



# ============================================================
# STAGE 6: CLASSIFICATION
# 5-FOLD CROSS VALIDATION + FINAL TEST EVALUATION
# ============================================================

# ============================================================
# STAGE 6: CLASSIFICATION
# 5-FOLD CROSS VALIDATION + FINAL TEST EVALUATION
# ============================================================

# Step 1: Load Required Libraries
library(caret)
library(e1071)
library(glmnet)

# Step 2: Convert Labels into Categorical Factors (0 = Faithful, 1 = Hallucinated)
y_train_factor <- factor(
  y_train,
  levels = c(0, 1),
  labels = c("Faithful", "Hallucinated")
)

y_test_factor <- factor(
  y_test,
  levels = c(0, 1),
  labels = c("Faithful", "Hallucinated")
)

# Step 3: Create 5-Fold Stratified Split for Validation
set.seed(123)
folds <- createFolds(
  y_train_factor,
  k = 5,
  list = TRUE
)

# Helper Function: Calculate F1 Score for Validation Folds
calculate_f1 <- function(pred, actual) {
  cm <- confusionMatrix(
    pred,
    actual,
    positive = "Hallucinated"
  )
  f1 <- cm$byClass["F1"]
  return(as.numeric(f1))
}

# ------------------------------------------------------------
# MODEL 1: BoW + Naive Bayes
# ------------------------------------------------------------
cat("\n--- Training Model 1: BoW + Naive Bayes ---\n")
bow_nb_f1 <- c()

for (i in 1:5) {
  val_index  <- folds[[i]]
  # Coerce dgCMatrix subset to dense matrix for naiveBayes
  X_train_cv <- as.matrix(X_train_bow[-val_index, ])
  y_train_cv <- y_train_factor[-val_index]
  X_val_cv   <- as.matrix(X_train_bow[val_index, ])
  y_val_cv   <- y_train_factor[val_index]
  
  model      <- naiveBayes(X_train_cv, y_train_cv)
  prediction <- predict(model, X_val_cv)
  
  bow_nb_f1[i] <- calculate_f1(prediction, y_val_cv)
}
cat("Model 1 Mean 5-Fold CV F1-Score:", round(mean(bow_nb_f1), 4), "\n")

# Final Training on Full Training Set & Test Set Prediction
bow_nb_model     <- naiveBayes(as.matrix(X_train_bow), y_train_factor)
bow_nb_test_pred <- predict(bow_nb_model, as.matrix(X_test_bow))










# ------------------------------------------------------------
# MODEL 2: TF-IDF + Naive Bayes
# ------------------------------------------------------------
cat("\n--- Training Model 2: TF-IDF + Naive Bayes ---\n")
tfidf_nb_f1 <- c()

for (i in 1:5) {
  val_index  <- folds[[i]]
  # Coerce dgCMatrix subset to dense matrix for naiveBayes
  X_train_cv <- as.matrix(X_train_tfidf[-val_index, ])
  y_train_cv <- y_train_factor[-val_index]
  X_val_cv   <- as.matrix(X_train_tfidf[val_index, ])
  y_val_cv   <- y_train_factor[val_index]
  
  model      <- naiveBayes(X_train_cv, y_train_cv)
  prediction <- predict(model, X_val_cv)
  
  tfidf_nb_f1[i] <- calculate_f1(prediction, y_val_cv)
}
cat("Model 2 Mean 5-Fold CV F1-Score:", round(mean(tfidf_nb_f1), 4), "\n")

# Final Training on Full Training Set & Test Set Prediction
tfidf_nb_model     <- naiveBayes(as.matrix(X_train_tfidf), y_train_factor)
tfidf_nb_test_pred <- predict(tfidf_nb_model, as.matrix(X_test_tfidf))

# ------------------------------------------------------------
# MODEL 3: BoW + Logistic Regression
# ------------------------------------------------------------
cat("\n--- Training Model 3: BoW + Logistic Regression ---\n")
set.seed(123)
# glmnet handles dgCMatrix directly without dense conversion
bow_lr_model <- cv.glmnet(
  X_train_bow,
  y_train,
  family = "binomial",
  nfolds = 5,
  type.measure = "class"
)

bow_lr_raw <- predict(
  bow_lr_model,
  X_test_bow,
  s = "lambda.min",
  type = "class"
)

bow_lr_test_pred <- factor(
  as.numeric(bow_lr_raw),
  levels = c(0, 1),
  labels = c("Faithful", "Hallucinated")
)



# ------------------------------------------------------------
# MODEL 4: TF-IDF + Logistic Regression
# ------------------------------------------------------------
cat("\n--- Training Model 4: TF-IDF + Logistic Regression ---\n")
set.seed(123)
# glmnet handles dgCMatrix directly without dense conversion
tfidf_lr_model <- cv.glmnet(
  X_train_tfidf,
  y_train,
  family = "binomial",
  nfolds = 5,
  type.measure = "class"
)

tfidf_lr_raw <- predict(
  tfidf_lr_model,
  X_test_tfidf,
  s = "lambda.min",
  type = "class"
)

tfidf_lr_test_pred <- factor(
  as.numeric(tfidf_lr_raw),
  levels = c(0, 1),
  labels = c("Faithful", "Hallucinated")
)

# ============================================================
# STEP 4: PRINT CONFUSION MATRICES FIRST, THEN COMPARISON TABLE
# ============================================================

# Define models and predictions in a list for clean iteration
model_list <- list(
  "BoW + Naive Bayes"           = bow_nb_test_pred,
  "TF-IDF + Naive Bayes"        = tfidf_nb_test_pred,
  "BoW + Logistic Regression"   = bow_lr_test_pred,
  "TF-IDF + Logistic Regression"= tfidf_lr_test_pred
)





# 1. Print all 4 Confusion Matrices first
cat("\n============================================================\n")
cat("                MODEL CONFUSION MATRICES                    \n")
cat("============================================================\n")

results_list <- list()

model_list <- list(
  "BoW + Naive Bayes"            = bow_nb_test_pred,
  "TF-IDF + Naive Bayes"         = tfidf_nb_test_pred,
  "BoW + Logistic Regression"    = bow_lr_test_pred,
  "TF-IDF + Logistic Regression" = tfidf_lr_test_pred
)

cat("\n============================================================\n")
cat("                MODEL CONFUSION MATRICES                    \n")
cat("============================================================\n")

for (model_name in names(model_list)) {
  pred <- model_list[[model_name]]
  
  cm <- confusionMatrix(
    pred,
    y_test_factor,
    positive = "Hallucinated"
  )
  
  cat("\n--------------------------------------------------\n")
  cat(" CONFUSION MATRIX:", model_name, "\n")
  cat("--------------------------------------------------\n")
  print(cm$table)
  
  results_list[[model_name]] <- data.frame(
    Model     = model_name,
    Accuracy  = round(unname(cm$overall["Accuracy"]), 4),
    Precision = round(unname(cm$byClass["Pos Pred Value"]), 4),
    Recall    = round(unname(cm$byClass["Sensitivity"]), 4),
    F1_Score  = round(unname(cm$byClass["F1"]), 4),
    stringsAsFactors = FALSE
  )
}

results <- do.call(rbind, results_list)
rownames(results) <- NULL

cat("\n============================================================\n")
cat("                FINAL MODEL RESULTS TABLE                   \n")
cat("============================================================\n")
print(results)
