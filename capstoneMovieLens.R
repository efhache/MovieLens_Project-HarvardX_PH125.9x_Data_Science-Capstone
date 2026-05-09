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

# Visualisation of the rate distribution
# This shows whether users are generous or strict.
edx %>% ggplot(aes(rating)) + 
  geom_histogram(binwidth = 0.5, color = "black", fill = "steelblue") + 
  ggtitle("Rating distribution (edx)") +
  xlab("Rates") + ylab("Number of ratings")


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
# Creating a training set and a test set from edx
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)
train_set <- edx[-test_index,]
temp <- edx[test_index,]

# Ensure that the users and films in the test_set are also in the train_set
test_set <- temp %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Add the deleted lines to the train_set
removed <- anti_join(temp, test_set)
train_set <- rbind(train_set, removed)

rm(test_index, temp, removed)


##########################################################
# Part 4: Data exploration & visualisation.
##########################################################

# 1. Distribution of number of ratings per movie
# We want to show that the majority of ratings are concentrated on a few
# blockbusters, whilst thousands of films are hardly ever watched.

edx %>% 
  count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "white", fill = "steelblue") + 
  scale_x_log10() + 
  ggtitle("Movies popularity distribution") +
  xlab("Number of ratings (Log scale)") +
  ylab("Number of movies") +
  theme_minimal()

# It can be seen that the distribution follows a power-law distribution.
# This justifies the use of a film effect (item bias), as popularity is not uniform.

# 2. Average rating by user
# We want to show that some users are consistently harsher or more generous than average.
edx %>% 
  group_by(userId) %>% 
  summarize(avg_rating = mean(rating)) %>% 
  ggplot(aes(avg_rating)) + 
  geom_histogram(bins = 30, color = "white", fill = "darkorange") + 
  ggtitle("Average rating distribution by Users") +
  xlab("Average Rating") +
  ylab("Number of users") +
  theme_minimal()
# There is significant variation among users.
# Some give an average score of 2, others 4.5, which justifies the inclusion of
# the user effect (user bias).


# 3. The danger of small sample sizes
# Identify movies with only a few ratings that have extreme averages
# in other words : we are going to look for the ‘best’ and ‘worst’ films that have only one rating.
movie_titles <- edx %>% select(movieId, title) %>% distinct()

edx %>% 
  group_by(movieId) %>% 
  summarize(n = n(), avg = mean(rating)) %>%
  left_join(movie_titles, by = "movieId") %>%
  filter(n <= 5) %>% 
  arrange(desc(avg)) %>% 
  head(10)
# It is noticeable that the films with the highest ratings are often obscure
# films with very few voters. This introduces noise into the model, justifying
# the use of Regularisation (Penalised Least Squares) to penalise small samples.


# 4. Extracting the year of the rating from the timestamp
# We are trying to answer the questions: Do people rate films more harshly
# over time? Or are older films rated more highly out of nostalgia?
library(lubridate)
edx %>% 
  mutate(date = as_datetime(timestamp)) %>%
  mutate(year = round_date(date, unit = "year")) %>%
  group_by(year) %>%
  summarize(rating = mean(rating)) %>%
  ggplot(aes(year, rating)) +
  geom_point() +
  geom_smooth(method = "loess") +
  ggtitle("Average rating by year of rating") +
  theme_minimal()
# The temporal analysis reveals a downward trend in average ratings from 
# 1995 to 2005, followed by a slight recovery. This suggests a time-dependent bias, 
# where the era of rating influences the score. While this effect is visible, 
# its magnitude (range of ~0.4) is smaller than the movie and user effects, 
# suggesting that while time is a factor, the primary drivers of rating variance
# remain movie quality and individual user behavior.


# 5. Extracting the year of release using a regular expression
# We are trying to determine whether the ‘classics’ receive higher ratings.
# We will therefore extract the year of release for each track.
edx <- edx %>% 
  mutate(release_year = as.numeric(str_extract(str_extract(title, "\\(\\d{4}\\)$"), "\\d{4}")))

edx %>% 
  group_by(release_year) %>%
  summarize(avg_rating = mean(rating)) %>%
  ggplot(aes(release_year, avg_rating)) +
  geom_line(color = "darkred") +
  labs(title = "Average Rating vs Release Year", x = "Release Year", y = "Average Rating") +
  theme_minimal()

# Selection Bias (Survivorship Bias): There is a clear trend showing that 
#    older movies (1930-1980) tend to have higher average ratings compared 
#    to modern movies. This suggests that only the "classics" persist in 
#    the dataset, while mediocre older films have been filtered out by time.
# High Variance in Early Years: The significant oscillations before 1930 
#    are due to low sample sizes (fewer ratings for very old films). 
# Justification for Modelling: This visualization justifies two things:
#    a) Potential inclusion of a release year effect (b_y) in the model.
#    b) The absolute necessity of Regularization to penalize unstable 
#       estimates from years/movies with very few observations


# 6. Compare the ‘Classics’ and ‘Mediocre’ effects
# If this statement is true, we should find that older films have lower variance
# (they are almost all ‘good’), whereas recent films have a much wider distribution
# (there are some that are very good and some that are very bad).
edx %>% 
  mutate(era = ifelse(release_year < 1980, "Old (Before 1980)", "Recent (After 1980)")) %>%
  ggplot(aes(rating, fill = era)) +
  geom_density(alpha = 0.4) +
  ggtitle("Distribution of Ratings: Old vs Recent Movies") +
  theme_minimal()
# The density plot provides a nuanced view of the Survivorship Bias:
# Shape of Distributions: Modern movies (After 1980) show extremely sharp, 
#    tall peaks at whole numbers (3.0, 4.0), indicating a highly concentrated 
#    but also more critical rating behavior.
# Density Mass: Although modern peaks are taller, older movies (Before 1980) 
#    exhibit "fresher" or wider shoulders toward the high end of the scale. 
#    The pink area (Old) is noticeably thicker between 4.5 and 5.0.
# Low-End Presence: Recent movies show a significantly larger density mass 
#    between 1 and 3, whereas older movies have almost no presence in the 
#    very low rating zones.
# Conclusion: This confirms that the higher average for classics isn't just 
#    about a few high scores, but about a global shift of the entire 
#    probability mass toward the right, as mediocre old films are absent 
#    from the dataset.

# 7. Check whether the fluctuations are due to the sample size
# To demonstrate that the ‘jagged lines’ prior to 1930 are due to a lack of
# data, we will correlate the year with the number of entries.
edx %>% 
  group_by(release_year) %>%
  summarize(n_ratings = n()) %>%
  ggplot(aes(release_year, n_ratings)) +
  geom_line() +
  scale_y_log10() +
  ggtitle("Number of Ratings by Release Year") +
  ylab("Count (Log Scale)") +
  theme_minimal()
# The log-scale distribution of ratings per release year explains the high
# variance observed in early 20th-century cinema. With some years having fewer
# than 100 ratings, the mean becomes highly sensitive to outliers. 
# This empirical evidence necessitates Regularization (Penalized Least Squares)
# to shrink these unstable estimates toward the global mean, ensuring that
# movies with low exposure do not unfairly bias the recommendation engine.


# 8. User ‘Strictness’ (Number of ratings vs Average rating)
# We aim to answer the following question : 
# Are users who rate a lot (the ‘critics’) stricter than those who only rate a few films?
edx %>% 
  group_by(userId) %>% 
  summarize(n_user = n(), avg_user = mean(rating)) %>% 
  filter(n_user >= 10) %>% # Those with too few ratings to be meaningful are ignored
  ggplot(aes(n_user, avg_user)) +
  geom_point(alpha = 0.1, color = "darkblue") +
  #geom_smooth(method = "loess", color = "red") +
  # We use ‘gam’ instead of ‘loess’ to handle the volume of data
  geom_smooth(method = "gam", formula = y ~ s(x, bs = "cs"), color = "red") +
  scale_x_log10() +
  labs(title = "User Rating Habit",
       subtitle = "Method: GAM smoothing for large datasets",
       x = "Number of ratings per User (Log Scale)",
       y = "Average Rating given") +
  theme_minimal()
# The scatter plot reveals a clear "funnel effect": 
# Variance vs Volume: Users with fewer ratings (left side) exhibit extreme 
# average ratings (0.5 or 5.0), showing high estimation instability. 
# Convergence: As the number of ratings increases (right side), the averages 
# converge toward the global mean (~3.6). 
# Regularization Necessity: This visualization is a textbook justification 
# for Regularization. We must penalize (shrink) the biases of users with 
# low rating counts to prevent them from skewing the model's predictions.
#
# The GAM smoothing reveals a non-linear relationship between user activity 
# and rating behavior. The downward trend for high-volume users proves that a 
# simple global mean (μ) is insufficient. We must incorporate a user-specific 
# bias to account for this shift in scale, while applying regularization to 
# stabilize the extreme averages seen in low-volume users (left side of the plot).



# 9. Analysis of the Title-Genre Correlation (The keyword ‘Series’)
# Sometimes, certain words in titles indicate a particular characteristic.
# For example, films that are part of a franchise. We can check whether films
# with long titles or specific parentheses receive higher ratings.
# But more simply, we can look at the standard deviation of ratings per user:
edx %>% 
  group_by(userId) %>% 
  filter(n() >= 50) %>%
  summarize(sd_user = sd(rating)) %>%
  ggplot(aes(sd_user)) +
  geom_histogram(bins = 30, fill = "darkgreen", color = "white") +
  labs(title = "Distribution of Rating Variability per User",
       subtitle = "Standard Deviation of ratings for users with >50 ratings",
       x = "Standard Deviation",
       y = "Number of Users") +
  theme_minimal()
# The distribution of standard deviations confirms that users not only haveµ
# different averages, but also radically different rating behaviours
# (some being very consistent, others very inconsistent).

# Number of ratings per user
# Do some users leave few reviews? This might require some adjustment
edx %>% count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black", fill = "forestgreen") + 
  scale_x_log10() +
  ggtitle("Number of ratings per user (log scale)") +
  xlab("Number of ratings") + ylab("Users")

# 10. Impact of genre complexity on ratings
# Some films belong to just one genre (e.g. ‘Drama’), whilst others belong to
# eight (e.g. ‘Action|Adventure|Sci-Fi|...’). So, we’re going to investigate
# whether a film’s complexity (the number of genres) influences its rating.
edx %>% 
  mutate(n_genres = str_count(genres, "\\|") + 1) %>%
  group_by(n_genres) %>%
  summarize(avg_rating = mean(rating), n = n()) %>%
  ggplot(aes(n_genres, avg_rating)) +
  geom_line(color = "purple", size = 1) +
  geom_point(aes(size = n), color = "purple") +
  labs(title = "Average Rating by Number of Genres",
       x = "Number of Genres assigned to a Movie",
       y = "Average Rating",
       size = "Number of ratings") +
  theme_minimal()
# Positive Trend (1-5 genres): Average ratings increase with the number 
# of genres, suggesting that "multi-genre" movies are perceived as 
# richer or appeal to a broader audience.
# Threshold Effect (6+ genres): A sharp drop is observed, but the 
# decreasing size of the data points indicates a "Small Sample Size" issue.
# Statistical Verdict: The volatility for 6+ genres is due to the low 
# number of ratings (low N), similar to the effect seen in very old films.
# Modeling Impact: This reinforces the idea that genre combinations 
# influence ratings, justifying the use of advanced bias modeling or 
# Regularization to handle these low-frequency combinations.


# 11. Age of the movie at the time of rating
# We investigate the "Nostalgia vs. Hype" effect: Does a movie's age when
# it is rated influence the score? This analysis combines the release year 
# and the timestamp to see if older films are judged more leniently.
edx %>% 
  mutate(date = as_datetime(timestamp)) %>%
  mutate(rating_year = year(date)) %>%
  mutate(age_at_rating = rating_year - release_year) %>%
  filter(age_at_rating >= 0) %>%
  group_by(age_at_rating) %>%
  summarize(avg_rating = mean(rating)) %>%
  ggplot(aes(age_at_rating, avg_rating)) +
  geom_line(color = "darkgreen") +
  geom_smooth(method = "gam", color = "red") +
  labs(title = "Average Rating vs Movie Age at Rating Time",
       x = "Age of Movie (Years)",
       y = "Average Rating") +
  theme_minimal()
# Maturation Effect: Ratings significantly increase during the first 25 years
# after release, showing that time filters out initial hype or general criticism. 
# Golden Era: Movies aged 30 to 60 years maintain the highest average ratings 
# (~3.9), confirming a strong survivorship bias and "classic" status. 
# Old Age Volatility: Beyond 80 years, the sharp decline and wide confidence 
# interval (grey area) reflect the scarcity of data for very old cinema. 
# Conclusion: Age is a powerful predictor. This justifies using it as a 
# feature or, at the very least, using Regularization to handle the 
# unstable predictions for the oldest films.
#
# Interestingly, the confidence interval narrows significantly around age 
# 85-90 despite the rating drop. This suggests a high concentration of ratings 
# for a very small number of specific silent-era masterpieces, which are rated 
# more harshly by contemporary users than the 'Golden Age' classics of the 1950s.


# Average rating by year of release
edx %>% group_by(release_year) %>%
  summarize(avg_rating = mean(rating)) %>%
  ggplot(aes(release_year, avg_rating)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "loess", color = "red") +
  ggtitle("Effect of the year of release on the average rating")

# 12. Day of the week effect
# We test if users are more generous on weekends vs weekdays.
edx %>%
  mutate(date = as_datetime(timestamp),
         day_of_week = wday(date, label = TRUE, abbr = FALSE)) %>%
  group_by(day_of_week) %>%
  summarize(avg_rating = mean(rating)) %>%
  ggplot(aes(day_of_week, avg_rating, fill = day_of_week)) +
  geom_bar(stat = "identity") +
  coord_cartesian(ylim = c(3.4, 3.7)) +
  labs(title = "Average Rating by Day of the Week",
       x = "Day of the Week", y = "Average Rating") +
  theme_minimal() +
  theme(legend.position = "none")

# Interpretation: If a difference exists, it shows that the context of 
# the rating (leisure time vs work week) is a subtle bias to consider.
# Conclusion: The visualization reveals that the average rating remains 
# remarkably stable across the week, with variations of less than 0.05 
# points between workdays and weekends. 
# While original as a hypothesis, the "Day of the Week" does not appear 
# to be a significant predictor for this dataset. 
# To maintain model parsimony (Occam's Razor), this feature will likely 
# be excluded from the final algorithm as it would add complexity 
# without significantly improving the RMSE.


# 13. Title Length Bias
# Does the length of a movie title correlate with its rating?
edx %>%
  mutate(title_length = nchar(title)) %>%
  group_by(title_length) %>%
  summarize(avg_rating = mean(rating), n = n()) %>%
  filter(n >= 50) %>% # We ignore lengths that are too short
  ggplot(aes(title_length, avg_rating)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "gam", color = "darkblue") +
  labs(title = "Average Rating vs Title Length",
       x = "Number of characters in title",
       y = "Average Rating") +
  theme_minimal()

# Interpretation: This explores if "short & punchy" titles perform 
# differently than "long & descriptive" ones.
# Conclusion: The GAM smoothing line is nearly flat, indicating no 
# significant correlation between title length and movie ratings. 
# While certain studies suggest title length can influence engagement, 
# it does not act as a bias in the MovieLens 10M dataset. 
# This experiment allows us to discard this variable and focus on 
# more impactful predictors like movie and user effects.


# 14. Genre Loyalty / Specialization
# Do "genre specialists" rate differently than casual viewers?
# We focus on the most common genre: Drama
drama_fans <- edx %>%
  group_by(userId) %>%
  summarize(drama_ratio = mean(str_detect(genres, "Drama")),
            avg_user_rating = mean(rating)) %>%
  filter(drama_ratio > 0)

ggplot(drama_fans, aes(drama_ratio, avg_user_rating)) +
  geom_point(alpha = 0.1, color = "darkred") +
  geom_smooth(method = "gam", color = "black") +
  labs(title = "Drama Specialization vs Average Rating",
       x = "Proportion of Drama movies rated",
       y = "User Average Rating") +
  theme_minimal()

# Interpretation: If the line goes down, it means that the more a user 
# watches a specific genre, the more "expert" and critical they become.

##update 14. to zoom on part between 3.4 and 3.8
# 14. Genre Loyalty / Specialization (REVISITED)
drama_fans <- edx %>%
  group_by(userId) %>%
  summarize(drama_ratio = mean(str_detect(genres, "Drama")),
            avg_user_rating = mean(rating)) %>%
  filter(drama_ratio > 0)

ggplot(drama_fans, aes(drama_ratio, avg_user_rating)) +
  # On réduit l'opacité des points au minimum pour ne voir que la masse
  geom_point(alpha = 0.02, color = "gray") + 
  # On garde la ligne noire bien visible
  geom_smooth(method = "gam", color = "black", size = 1.2) +
  # THIS IS WHERE IT ALL HAPPENS: Let’s zoom in to see the differences
  coord_cartesian(ylim = c(3.5, 3.7)) + 
  labs(title = "Focus: Drama Specialization vs Average Rating",
       subtitle = "Zoomed Y-axis to highlight the non-linear trend",
       x = "Proportion of Drama movies rated",
       y = "User Average Rating") +
  theme_minimal()
# Conclusion: The analysis reveals that "Drama specialists" (users with a 
# high drama ratio) tend to give significantly higher ratings than average. 
# While the global mean is around 3.5, users specialized in Drama often 
# exceed 3.65. This confirms a strong interaction between user preference 
# and this specific genre, justifying why a simple movie-effect model 
# isn't enough and why we need to account for these "loyal" user profiles.

# 14b. Cross-Genre Loyalty Comparison (Version Corrigée)
genres_to_test <- c("Drama", "Comedy", "Action", "Thriller", "Sci-Fi", "Horror")

# Fonction corrigée (on s'assure que les noms de colonnes matchent le ggplot)
check_genre_loyalty <- function(genre_name) {
  edx %>%
    group_by(userId) %>%
    summarize(ratio = mean(str_detect(genres, genre_name)),
              avg_rating = mean(rating), # Nom de colonne : avg_rating
              .groups = "drop") %>%
    filter(ratio > 0) %>%
    mutate(genre = genre_name)
}

# Applying the function to the selected genres
loyalty_data <- map_df(genres_to_test, check_genre_loyalty)

v
ggplot(loyalty_data, aes(x = ratio, y = avg_rating)) + # Using avg_rating
  # We use geom_bin2d or a very low alpha value to avoid overloading the RAM when displaying the image
  geom_smooth(method = "gam", color = "darkblue", size = 1) +
  facet_wrap(~genre, scales = "free") +
  coord_cartesian(ylim = c(3.2, 4.0)) + # Zoom pour bien voir les pentes
  labs(title = "Specialization Effect across Multiple Genres",
       subtitle = "Comparison of rating trends by user loyalty",
       x = "Proportion of genre in user's history",
       y = "User Average Rating") +
  theme_minimal()
# Conclusion: This multi-genre comparison reveals a fascinating "Specialization Paradox". 
# While "Drama" shows a positive loyalty effect (ratings increase with exposure), 
# "Horror" and "Sci-Fi" exhibit a strong negative loyalty effect: the more 
# users watch these genres, the more critical they become, significantly 
# lowering their average ratings. 
# "Comedy" and "Action" show a neutral trend after an initial stabilization. 
# This diversity of behaviors proves that a global "User Effect" or "Genre Effect" 
# is insufficient; we need a model that captures these complex interactions.

# 15. Visualizing Matrix Sparsity and Latent Structure
# Note: Instead of a descriptive heatmap, I use an abstract 'Sparsity Matrix'. 
#
# WHY THIS APPROACH?
# 1. STRUCTURAL OVERVIEW: By removing movie titles and user IDs, we shift the focus 
#    from individual data points to the global architecture of the dataset.
# 2. SPARSITY DEMONSTRATION: By using a RANDOM sample, the 'white space' visually 
#    proves the fundamental challenge: most users haven't seen most movies.
# 3. PATTERN RECOGNITION: The rare vertical and horizontal streaks reveal the few 
#    prolific users and popular movies that provide the 'anchors' for our models.
# 4. MATH TRANSITION: This visual justifies the use of Matrix Factorization to 
#    predict the missing values (the white areas) mathematically.

set.seed(42)

# We take 200 users and 200 films at random to see the actual rarity
random_users <- sample(unique(edx$userId), 200)
random_movies <- sample(unique(edx$movieId), 200)

heatmap_data <- edx %>% 
  filter(userId %in% random_users & movieId %in% random_movies)

# Visualisation to bring out streaks and voids
ggplot(heatmap_data, aes(x = as.factor(movieId), y = as.factor(userId), fill = rating)) +
  geom_tile() +
  scale_fill_viridis_c(option = "magma", na.value = "white") +
  # Force the full grid to be displayed even where there is no data
  scale_x_discrete(drop = FALSE) + 
  scale_y_discrete(drop = FALSE) +
  theme_minimal() +
  theme(axis.text = element_blank(),
        panel.grid = element_blank()) +
  labs(title = "True Matrix Sparsity (Random 100x100 Sample)",
       subtitle = "White areas represent missing ratings (unseen movies)",
       x = "Movies (Random Sample)",
       y = "Users (Random Sample)")


##########################################################
# Part 5: Modelling and calculating the RMSE.
##########################################################

# DATA REFRESH 
# IMPORTANT: We re-run the split from Part 3 because we added new features 
# to 'edx' (release_year, age_at_rating, rating_year) during exploration. 
# Re-splitting ensures that 'train_set' and 'test_set' contain these variables 
# for the modeling phase.

set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)
train_set <- edx[-test_index,]
temp <- edx[test_index,]

# Ensure userId and movieId in test_set are also in train_set
test_set <- temp %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Add back rows removed from test_set to train_set to maintain data integrity
removed <- anti_join(temp, test_set)
train_set <- rbind(train_set, removed)

rm(test_index, temp, removed)
# END DATA REFRESH

# 1. Define the RMSE function
# This function will be our judge for all subsequent models.
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# 2. Model 1: Naive Baseline (Just the average)
# We assume the best prediction for any movie/user combination is the 
# average rating of the entire training set.
mu_hat <- mean(train_set$rating)

# Calculate RMSE on the test set
naive_rmse <- RMSE(test_set$rating, mu_hat)

# Create a results table to track our progress
# We will add every new model result to this data frame.
rmse_results <- data.frame(method = "Baseline: Global Average (mu)", 
                           RMSE = naive_rmse)

# Display the result
print(rmse_results)

# 3. Model 2: Movie Effect (b_i)
# We calculate the average deviation for each movie from the global mean mu.
movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu_hat))

# Predict ratings by adding the movie effect to mu
predicted_ratings <- mu_hat + test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)

# Calculate and store RMSE
model_1_rmse <- RMSE(test_set$rating, predicted_ratings)
rmse_results <- rbind(rmse_results,
                      data.frame(method="Movie Effect Model (mu + b_i)",  
                                 RMSE = model_1_rmse))

print(rmse_results)

# 4. Model 3: Movie + User Effect (b_i + b_u)
# We calculate the average deviation of each user, 
# taking into account the movie effect already calculated.
user_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu_hat - b_i))

# Predict ratings
predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu_hat + b_i + b_u) %>%
  pull(pred)

# Calculate and store RMSE
model_2_rmse <- RMSE(test_set$rating, predicted_ratings)
rmse_results <- rbind(rmse_results,
                      data.frame(method="Movie + User Effects Model (mu + b_i + b_u)",  
                                 RMSE = model_2_rmse))

print(rmse_results)

# 5. Model 4: Regularized Movie + User Effect
# We use cross-validation to find the optimal lambda (tuning parameter)

lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas, function(l){
  
  mu <- mean(train_set$rating)
  
  b_i <- train_set %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  predicted_ratings <- test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, test_set$rating))
})

# Plot lambdas to visualize the minimum
#qplot(lambdas, rmses)

# Find the best lambda
#lambda <- lambdas[which.min(rmses)]

# Find the best lambda and the corresponding minimum RMSE
min_rmse <- min(rmses)
best_lambda <- lambdas[which.min(rmses)]

# Update the results table
rmse_results <- rbind(rmse_results,
                      data.frame(method = "Regularized Movie + User Effect",  
                                 RMSE = min_rmse))

# Display the results and the best lambda
print(rmse_results)
cat("The best lambda is:", best_lambda)

# Replacement for qplot to avoid the warning
ggplot(data.frame(lambdas = lambdas, rmses = rmses), aes(x = lambdas, y = rmses)) +
  geom_point() +
  theme_minimal() +
  labs(title = "RMSE Optimization",
       subtitle = "Finding the optimal lambda for regularization",
       x = "Lambda",
       y = "RMSE")


# 6. Model 5: Regularized Movie + User + Genre + Year Effect
# We include the release_year and genres effects found during EDA 5)
# [DRY NOTE]: This section is kept for pedagogical validation alongside the 
# optimized loop below. (DRY is for Don't Repeat Yourself)

# Retrieve the release year if it is missing (format "Title (Year)")
if(!"release_year" %in% colnames(train_set)){
  train_set <- train_set %>% 
    mutate(release_year = as.numeric(str_extract(title, "(?<=\\()\\d{4}(?=\\))")))
  test_set <- test_set %>% 
    mutate(release_year = as.numeric(str_extract(title, "(?<=\\()\\d{4}(?=\\))")))
}

# One approach is to use the `sapply` loop, but this is very memory-intensive. 
# At each iteration of the lambda function, there are 4 left_join operations on 
# millions of rows. This risks causing R to crash or taking hours on my setup
# (Virtual Machine, 6GB max)
# To test whether the Genre + Year effect works, we can first calculate the biases
# just once using the previous best_lambda (4.75) instead of running a full 
# loop again.

l <- 4.75 # We use the pre-optimised lambda to save time

mu <- mean(train_set$rating)

# We calculate the biases one by one
b_i <- train_set %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+l))

b_u <- train_set %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+l))

b_g <- train_set %>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by="userId") %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - b_i - b_u - mu)/(n()+l))

b_y <- train_set %>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by="userId") %>%
  left_join(b_g, by="genres") %>%
  group_by(release_year) %>%
  summarize(b_y = sum(rating - b_i - b_u - b_g - mu)/(n()+l))

# Prediction
predicted_ratings <- test_set %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_g, by = "genres") %>%
  left_join(b_y, by = "release_year") %>%
  mutate(pred = mu + b_i + b_u + b_g + b_y) %>%
  pull(pred)

final_rmse_complex <- RMSE(predicted_ratings, test_set$rating)

# Update results table
rmse_results <- rbind(rmse_results,
                      data.frame(method = "Reg. Movie+User+Genre+Year Effect",  
                                 RMSE = final_rmse_complex))

print(rmse_results)



# 5. Optimized Regularization (Genre + Year included)

# Running a sapply with four successive left_join operations on 10 million rows
# in a 6 GB VM will cause a ‘Memory Exhaustion’ (R to crash). To optimise this,
# I will pre-calculate the sums and counts. Instead of performing join operations
# at each iteration of the loop, we work with vectors of residuals. This is much
# lighter on the RAM.

# SCIENTIFIC INSIGHT: 
# You will notice that the RMSE and best_lambda remain identical between 
# the single test above and this systematic loop. 
# 
# WHY?
# 1. CONVERGENCE: The optimal lambda (4.75) discovered for Movie/User 
#    effects remains the anchor for the global model.
# 2. CARDINALITY: Genre and Year have much fewer levels (lower cardinality) 
#    than Users or Movies. Therefore, they introduce less 'noise' that 
#    requires additional shrinkage (regularization).
# 3. STABILITY: This consistency proves the robustness of the 
#    regularization parameter across different feature dimensions.

gc() #garbage collector

lambdas <- seq(0, 10, 0.25)

# Pre-calculate components to avoid joins in the loop
# Calculate the residuals just once for the base
mu <- mean(train_set$rating)

# We pre-aggregate the data by film and user to speed things up
movie_stats <- train_set %>%
  group_by(movieId) %>%
  summarize(s = sum(rating - mu), n = n())

user_stats <- train_set %>%
  group_by(userId) %>%
  summarize(s = sum(rating - mu), n = n())

# Note: For gender and year, we stick to a simple calculation 
# as they have fewer levels (lower cardinality)

rmses <- sapply(lambdas, function(l){
  
  # Movie effect smoothed
  b_i <- movie_stats %>%
    mutate(b_i = s / (n + l)) %>%
    select(movieId, b_i)
  
  # Regularised user effect (approximated for velocity)
  b_u <- train_set %>%
    left_join(b_i, by = "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu) / (n() + l))
  
  # We stop at b_i and b_u for the tuning loop
  # because optimising lambda for four variables simultaneously is likely to be too computationally intensive.
  
  predicted_ratings <- test_set %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, test_set$rating))
})

# Final result with the best lambda
best_lambda <- lambdas[which.min(rmses)]

# We apply this lambda function to the FULL model (including Genre and Year) just once
l <- best_lambda

b_i <- train_set %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+l))

b_u <- train_set %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+l))

b_g <- train_set %>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by="userId") %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - b_i - b_u - mu)/(n()+l))

b_y <- train_set %>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by="userId") %>%
  left_join(b_g, by="genres") %>%
  group_by(release_year) %>%
  summarize(b_y = sum(rating - b_i - b_u - b_g - mu)/(n()+l))

# Score final
final_predicted_ratings <- test_set %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_g, by = "genres") %>%
  left_join(b_y, by = "release_year") %>%
  mutate(pred = mu + b_i + b_u + b_g + b_y) %>%
  pull(pred)

final_rmse <- RMSE(final_predicted_ratings, test_set$rating)

# Update results
rmse_results <- rbind(rmse_results,
                      data.frame(method = "Final Regularized Model (All Effects)",  
                                 RMSE = final_rmse))
print(rmse_results)

# 7. Model 6: Adding Time Effect (Week-based) - Optimized for RAM
# One initial approach would be to use the round_any function (which is part of 
# the plyr package), but as I have a 6 GB RAM limit, loading the entire plyr 
# package could lead to excessive memory usage, and it conflicts with dplyr (tidyverse).
# The preferred option is therefore to use a simple mathematical formula in basic
# R to do exactly the same thing (round the timestamp to the nearest week).

# 604800 seconds = 7 days * 24h * 3600s

# Feature engineering without external packages
train_set <- train_set %>% 
  mutate(date = floor(timestamp / 604800) * 604800)

test_set <- test_set %>% 
  mutate(date = floor(timestamp / 604800) * 604800)

# We use the best lambda found previously
l <- 4.75

# Calculation of the time effect b_t
# We subtract all previous effects to isolate the variation due to time
b_t <- train_set %>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by="userId") %>%
  left_join(b_g, by="genres") %>%
  left_join(b_y, by="release_year") %>%
  group_by(date) %>%
  summarize(b_t = sum(rating - mu - b_i - b_u - b_g - b_y) / (n() + l))

# Interim test on the test_set
predicted_ratings_t <- test_set %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_g, by = "genres") %>%
  left_join(b_y, by = "release_year") %>%
  left_join(b_t, by = "date") %>%
  mutate(pred = mu + b_i + b_u + b_g + b_y + b_t) %>%
  pull(pred)

# Handling unknown dates (NAs)
predicted_ratings_t[is.na(predicted_ratings_t)] <- mu

# Displaying the new test RMSE
current_rmse <- RMSE(predicted_ratings_t, test_set$rating)
cat("Current RMSE with time-varying effects:", current_rmse)


# 8. Model 7: Backfitting + Capping - OPTIMISED FOR VM 6GO
# what trying to do:
# 1. BIAS CONVERGENCE (Backfitting): In previous models, b_i was calculated 
# independently of b_u. However, movie and user effects are interconnected. 
# Recalculating b_i after estimating b_u (Backfitting) allows the biases to 
# account for each other, leading to more stable and accurate estimates. 
# 2. DATA BOUNDARY LOGIC (Capping): Linear models can predict values > 5 or < 0.5. 

# 1. Standardising the name of the variable mu
mu <- mu_hat

# 2. Préparation des colonnes nécessaires (Feature Engineering)
# Pour le train_set
train_set <- train_set %>% 
  mutate(release_year = as.numeric(str_extract(str_extract(title, "\\(\\d{4}\\)$"), "\\d{4}")),
         date = floor(timestamp / 604800) * 604800)

# For the test_set
test_set <- test_set %>% 
  mutate(release_year = as.numeric(str_extract(str_extract(title, "\\(\\d{4}\\)$"), "\\d{4}")),
         date = floor(timestamp / 604800) * 604800)

lambdas_finetune <- seq(3.5, 5.5, 0.25)

rmses_model7 <- sapply(lambdas_finetune, function(l){
  
  # PASSE 1
  b_i <- train_set %>% 
    group_by(movieId) %>% 
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>% 
    group_by(userId) %>% 
    summarize(b_u = sum(rating - mu - b_i)/(n()+l))
  
  # STEP 2 (Refinement) – We overwrite b_i so as not to store two versions
  b_i <- train_set %>% 
    left_join(b_u, by="userId") %>% 
    group_by(movieId) %>% 
    summarize(b_i = sum(rating - mu - b_u)/(n()+l))
  
  # Other effects calculated sequentially
  b_g <- train_set %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - mu - b_i - b_u)/(n()+l))
  
  b_y <- train_set %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    left_join(b_g, by="genres") %>%
    group_by(release_year) %>%
    summarize(b_y = sum(rating - mu - b_i - b_u - b_g)/(n()+l))
  
  b_t <- train_set %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    left_join(b_g, by="genres") %>%
    left_join(b_y, by="release_year") %>%
    group_by(date) %>%
    summarize(b_t = sum(rating - mu - b_i - b_u - b_g - b_y)/(n()+l))
  
  # Prediction with capping (pmin/pmax is very efficient in memory)
  preds <- test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
    left_join(b_y, by = "release_year") %>%
    left_join(b_t, by = "date") %>%
    mutate(pred = mu + b_i + b_u + b_g + b_y + b_t) %>%
    mutate(pred = pmin(5, pmax(0.5, pred))) %>% 
    pull(pred)
  
  preds[is.na(preds)] <- mu
  rmse_val <- RMSE(preds, test_set$rating)
  
  # CRITICAL CLEAN-UP FOR 6GB VM
  rm(b_i, b_u, b_g, b_y, b_t, preds)
  gc() # Forces RAM to be released before the next lambda
  
  return(rmse_val)
})

best_l_m7 <- lambdas_finetune[which.min(rmses_model7)]
best_rmse_m7 <- min(rmses_model7)

# Displaying results
cat("--- MODEL 7 RESULTS ---")
cat("Optimal Lambda (Test Set):", best_l_m7)
cat("Improved Test RMSE:", best_rmse_m7)

# Update to the results tracking table
rmse_results <- rbind(rmse_results,
                      data.frame(method = "Model 7: Backfitting + Capping + All Effects",  
                                 RMSE = best_rmse_m7))

# Visualisation of the optimisation
plot(lambdas_finetune, rmses_model7, type = "b", 
     main = "Optimization of Lambda for Model 7",
     xlab = "Lambda", ylab = "RMSE")

##########################################################
# Part 6: Final calculation on final_holdout_test.
##########################################################
# Having found the optimal lambda (best_l_m7) using the train/test split, 
# we now retrain the model on the ENTIRE 'edx' dataset. 
# Mathematically, this maximizes the information used to estimate biases, 
# leading to more stable predictors. The 'final_holdout_test' remains 
# untouched until this very last step to ensure a true "blind" evaluation.

# NOTE: 
# The final_holdout_test provided by HarvardX does not contain a 'release_year' 
# column by default. However, our best performing model uses this feature.
# To ensure consistency, we perform feature extraction from the 'title' column 
# (which is provided). This is NOT an alteration of the data points themselves, 
# but a transformation of existing information to match our model's requirements.


# 1. Prepare the final hold-out set (Feature Extraction)
final_holdout_test <- final_holdout_test %>% 
  mutate(
    # Extract release year from title
    release_year = as.numeric(str_extract(str_extract(title, "\\(\\d{4}\\)$"), "\\d{4}")),
    # Convert timestamp to week units
    date = floor(timestamp / 604800) * 604800
  )


# 2. Final Parameter Assignment
l_final <- best_l_m7
mu_edx <- mean(edx$rating) # Global average of the full edx set

# Adding the columns required by EDX for the final model
edx <- edx %>% 
  mutate(release_year = as.numeric(str_extract(str_extract(title, "\\(\\d{4}\\)$"), "\\d{4}")),
         date = floor(timestamp / 604800) * 604800)

# Cleaning up to free up RAM after the migration
gc()

# 3. Final Bias Calculation on EDX (using Backfitting logic)
# PASSE 1
b_i <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu_edx)/(n() + l_final))

b_u <- edx %>% 
  left_join(b_i, by="movieId") %>% 
  group_by(userId) %>% 
  summarize(b_u = sum(rating - mu_edx - b_i)/(n() + l_final))

# PASSE 2 (Refinement)
b_i <- edx %>% 
  left_join(b_u, by="userId") %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu_edx - b_u)/(n() + l_final))

# Sequential calculation for other effects
b_g <- edx %>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by="userId") %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - mu_edx - b_i - b_u)/(n() + l_final))

b_y <- edx %>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by="userId") %>%
  left_join(b_g, by="genres") %>%
  group_by(release_year) %>%
  summarize(b_y = sum(rating - mu_edx - b_i - b_u - b_g)/(n() + l_final))

b_t <- edx %>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by="userId") %>%
  left_join(b_g, by="genres") %>%
  left_join(b_y, by="release_year") %>%
  group_by(date) %>%
  summarize(b_t = sum(rating - mu_edx - b_i - b_u - b_g - b_y)/(n() + l_final))

# 4. FINAL PREDICTION with Capping
final_predictions <- final_holdout_test %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_g, by = "genres") %>%
  left_join(b_y, by = "release_year") %>%
  left_join(b_t, by = "date") %>%
  mutate(pred = mu_edx + b_i + b_u + b_g + b_y + b_t) %>%
  mutate(pred = pmin(5, pmax(0.5, pred))) %>% 
  pull(pred)

# Handle potential NAs for items/users not in edx
final_predictions[is.na(final_predictions)] <- mu_edx

# 5. THE VERDICT: FINAL RMSE
final_holdout_rmse <- RMSE(final_holdout_test$rating, final_predictions)

# 6. Final Results Table
rmse_results <- rbind(rmse_results,
                      data.frame(method = "FINAL MODEL: Backfitting + Capping (Full edx set)",  
                                 RMSE = final_holdout_rmse))

print(rmse_results)
cat(">>> OFFICIAL FINAL SCORE (Hold-out):", final_holdout_rmse)

# Final result (formatted)
final_table <- data.frame(
  "Threshold_Target" = 0.86490,
  "RMSE_Final" = final_holdout_rmse,
  "Performance" = ifelse(final_holdout_rmse < 0.86490, "Target Achieved", "Target Not Met")
)
print(final_table)

# 7. Cleanup to free memory
rm(b_i, b_u, b_g, b_y, b_t, final_predictions)
gc()
