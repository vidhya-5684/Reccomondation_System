## The code provided by the course material for the generation of the dataset.

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(recosystem)

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")
set.seed(1, sample.kind="Rounding") 

#Splitting data into test and train dataset
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)
dim(edx)
dim(validation)
head(edx)

# Splitting edx dataset for testing diffrent models.
set.seed(1, sample.kind="Rounding") 
test_index_sample <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train_set <- edx[-test_index_sample,]
temp_sample <- edx[test_index_sample,]
dim(train_set)

# Make sure userId and movieId in validation set are also in edx set
test_set <- temp_sample %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")
# Add rows removed from validation set back into edx set
removed_sample <- anti_join(temp_sample, test_set)
train_set <- rbind(train_set, removed_sample)

rm(test_index_sample, temp_sample, removed_sample)

# Defining Loss function
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}
## Data Exploration

edx %>% 
  group_by(rating) %>% 
  summarise(n= n()) %>% 
  as_tibble() ## counting number of ratings for each rating

edx %>% 
  select(rating) %>%
  ggplot(aes(rating)) +
  geom_histogram(bins = 10,color = "white") + 
  ggtitle("Rating Distributions") +
  xlab("Ratings") +
  ylab("Number of Ratings") ## Distribution of rating

edx %>% 
  group_by(movieId) %>%
  summarise(n=n()) %>%
  ggplot(aes(n)) +
  geom_histogram(color = "white") +
  scale_x_log10() + 
  ggtitle("Distribution of Movies")+
  xlab("Number of Ratings") +
  ylab("Number of Movies") ## Distribution of movies

edx %>% group_by(userId) %>%
  summarise(n=n()) %>%
  arrange(n) %>%
  head() %>% 
  as_tibble() ## Summarization of user column

edx %>% group_by(userId) %>%
  summarise(n=n()) %>%
  ggplot(aes(n)) +
  geom_histogram(color = "white") +
  scale_x_log10() + 
  ggtitle("Distribution of Users")+
  xlab("Number of Ratings") +
  ylab("Number of Users") ## Distribution of user Details

edx %>% group_by(genres) %>% 
  summarise(n=n()) %>%
  head() ## Genre column



# First Model
mu_hat <- mean(train_set$rating) #Average of the ratings


#RMSE of the model with mu_hat as the predicted value
mean_RMSE <- RMSE(test_set$rating,mu_hat)
mean_RMSE
#RMSE = 1.059

#making a table for different RMSE of different model for comparing
rmse_results <- tibble(method = "Just the average", RMSE = mean_RMSE)

# A model with movie effect
mu <- mean(train_set$rating)

avg_movies <-  train_set %>%
  group_by(movieId) %>%
  summarise(b_i = mean(rating - mu))

#Plotting b_i to show the variations.
qplot(b_i, data = avg_movies, bins = 10, color = I("black"))

#New RMSE
predicted_ratings <- mu + test_set %>%
  left_join(avg_movies, by = "movieId") %>%
  pull(b_i)

RMSE_movies <- RMSE(predicted_ratings, test_set$rating)
RMSE_movies
#RMSE_movie = 0.94

#Model with movie effect
#Estimating b_u
avg_user<- train_set %>% 
  left_join(avg_movies,by="movieId") %>% 
  group_by(userId) %>% 
  summarise(b_u = mean(rating - mu - b_i))
#Predicting ratings and calculating RMSE
predicted_ratings <- test_set %>% 
  left_join(avg_movies,by="movieId") %>% 
  left_join(avg_user,by="userId") %>% 
  mutate(pred=mu+b_u+b_i) %>% 
  pull(pred)

User_RMSE <- RMSE(predicted_ratings, test_set$rating)
User_RMSE

##Factorization
lambdas <- seq(0, 10, 0.25)
rmsess <- sapply(lambdas, function(l){
  mu <- mean(train_set$rating)

  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))

  b_u <- train_set %>%
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))

  predicted_ratings <-
    test_set %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
   return(RMSE(predicted_ratings, test_set$rating)) })

 qplot(lambdas, rmsess)  
lambda <- lambdas[which.min(rmsess)]
##predicting ratings with lambda
b_i <- train_set %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda))
b_u <- train_set %>%
  left_join(b_i, by="movieId") %>% group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))

predicted_ratings <-
  test_set %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)


regular_RMSE <- RMSE(predicted_ratings, test_set$rating)
regular_RMSE

### Matrix factorization.....

 if(!require(recosystem))
   install.packages("recosystem", repos = "http://cran.us.r-project.org") ##installing recosystem library

 set.seed(1, sample.kind = "Rounding")

# # Converting train and test data to recosystem input format
train <-  with(train_set, data_memory(user_index = userId,
                                           item_index = movieId,
                                           rating     = rating))
test  <-  with(test_set,  data_memory(user_index = userId,
                                           item_index = movieId,
                                           rating     = rating))

# Creating the model object
model <-  recosystem::Reco()

# Select the best tuning parameters
params <- model$tune(train, opts = list(dim = c(10, 20, 30),
                                       lrate = c(0.1, 0.2),
                                       costp_l2 = c(0.01, 0.1),
                                       costq_l2 = c(0.01, 0.1),
                                       nthread  = 4, niter = 10))

# # Training the algorithm
 model$train(train, opts = c(params$min, nthread = 4, niter = 20))

predicted_ratings <-  model$predict(test, out_memory())
head(predicted_ratings, 10)
RMSE_MF <- RMSE(predicted_ratings, test_set$rating)
RMSE_MF
#
# ##############################################
#

train_edx <-  with(edx, data_memory(user_index = userId,
                                      item_index = movieId,
                                      rating     = rating))
test_validation  <-  with(validation,  data_memory(user_index = userId,
                                      item_index = movieId,
                                      rating     = rating))


# Applying the algorithm on edy and validation dat set
model$train(train_edx, opts = c(params$min, nthread = 4, niter = 20))

predicted_ratings <-  model$predict(test_validation, out_memory())
head(predicted_ratings, 10)
RMSE_MF_valid <- RMSE(predicted_ratings, validation$rating)
RMSE_MF_valid




top_movies <- tibble(title = validation$title, rating = predicted_ratings) %>%
  arrange(-rating) %>%
  group_by(title) %>%
  select(title) %>%
  head(10)

bad <- tibble(title = validation$title, rating = predicted_ratings) %>%
  arrange(rating) %>%
  group_by(title) %>%
  select(title) %>%
  head(10)





