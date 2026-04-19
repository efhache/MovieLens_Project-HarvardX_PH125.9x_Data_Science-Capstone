##########################################################
# Part 1: Loading the data and creating the edx and final_holdout_test 
# datasets. Code provided at the start of the course
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 120)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


##########################################################
# Part 2: Preliminary Data Exploration (edx dataset)
# responses to the quizz
##########################################################
# 1. Structure and dimensions of edx
res_dim <- dim(edx)
message(paste("The edx dataset has", res_dim[1], "rows and", res_dim[2], "columns."))

# 2. Preview of the data
head(edx)

# 3. Summary of key statistics
summary(edx)

# 4. Counting unique users and movies, numbers of zeros and threes
n_unique_users <- n_distinct(edx$userId)
n_unique_movies <- n_distinct(edx$movieId)

message(paste("There are", n_unique_users, "unique users and", n_unique_movies, "unique movies in edx."))

# Verification of specific ratings. Using efficient vectorized sums for specific counts
count_0 <- sum(edx$rating == 0)
count_3 <- sum(edx$rating == 3)
# Insight: In the MovieLens 10M dataset, ratings are on a 0.5 to 5.0 scale.
# A count of 0 for the "0" rating confirms the expected range.
message(paste("Number of 0 ratings:", count_0))
message(paste("Number of 3 ratings:", count_3))

# 5. Identifying the most rated genres (simple top 10)
# NOTE ON COMPUTATIONAL LIMITATIONS:
# The standard approach using separate_rows() directly on the 9M row 'edx' dataset
# represents a significant memory overhead. In a Virtual Machine (VM) environment 
# with limited RAM (my setting), this operation triggers a "memory exhaustion" state, 
# leading to system instability (Disk Swapping).


# [ALGORITHM 0: Standard Tidyverse - COMMENTED OUT DUE TO RAM CONSTRAINTS]
# top_genres <- edx %>% 
#   separate_rows(genres, sep = "\\|") %>% 
#   group_by(genres) %>% 
#   summarize(count = n()) %>% 
#   arrange(desc(count))
# head(top_genres, 10)

# To overcome this, we evaluate three alternative methods to ensure 
# results consistency while optimizing memory footprint.

# --- Method 1: Data Sampling (Statistical Approximation) ---
# Useful for quick exploration but introduces a small margin of error.
top_genres_m1 <- edx %>% 
  slice_sample(n = 1000000) %>% 
  separate_rows(genres, sep = "\\|") %>% 
  group_by(genres) %>% 
  summarize(count = n()) %>%
  mutate(method = "Sampling (1M)")

# --- Method 2: stringr::str_count (Efficient counting without duplication) ---
# Accurate for counts, but less flexible for complex multi-genre analysis.
genres_list <- c("Drama", "Comedy", "Thriller", "Romance") # Example list
top_genres_m2 <- data.frame(
  genres = genres_list,
  count = sapply(genres_list, function(g) sum(str_detect(edx$genres, g))),
  method = "String Detect"
)

# --- Method 3: Pre-aggregation (Optimal Engineering approach) ---
# This method reduces the 9M rows to ~800 unique combinations BEFORE expanding.
# It is mathematically identical to Algorithm 0 but runs in seconds.
top_genres_m3 <- edx %>%
  group_by(genres) %>%
  summarize(n = n()) %>% 
  separate_rows(genres, sep = "\\|") %>%
  group_by(genres) %>%
  summarize(count = sum(n)) %>%
  mutate(method = "Pre-aggregation") %>%
  arrange(desc(count))

# Verification: Compare results of Method 1 and Method 3
head(top_genres_m3)

# Movie with the greatest number of ratings
edx %>%
  group_by(movieId, title) %>%
  summarize(count = n()) %>%
  arrange(desc(count)) %>%
  head(1)

# Top 5 most given ratings
edx %>%
  group_by(rating) %>%
  summarize(count = n()) %>%
  arrange(desc(count)) %>%
  head(5)

# Comparison between whole stars and half stars
edx %>%
  group_by(rating) %>%
  summarize(count = n()) %>%
  mutate(half_star = ifelse(rating %% 1 == 0, "Whole", "Half")) %>%
  ggplot(aes(x = factor(rating), y = count, fill = half_star)) +
  geom_bar(stat = "identity") +
  labs(title = "Frequency of Ratings",
       subtitle = "Comparison: Whole stars vs Half stars",
       x = "Rating",
       y = "Count",
       fill = "Type") +
  theme_minimal()

##########################################################
# Part 3: Creating the train_set and test_set
##########################################################
# EN : Creating a training set and a test set from edx
# FR : Création d'un set d'entraînement et d'un set de test à partir de edx
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)
train_set <- edx[-test_index,]
temp <- edx[test_index,]

# EN : Ensure that the users and films in the test_set are also in the train_set
# FR : S'assurer que les utilisateurs et films du test_set sont aussi dans le train_set
test_set <- temp %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# EN : Add the deleted lines to the train_set
# FR : Ajouter les lignes supprimées au train_set
removed <- anti_join(temp, test_set)
train_set <- rbind(train_set, removed)

rm(test_index, temp, removed)


##########################################################
# Part 4: Data exploration & visualisation.
##########################################################



##########################################################
# Part 5: Modelling and calculating the RMSE.
##########################################################



##########################################################
# Part 6: Final calculation on final_holdout_test.
##########################################################