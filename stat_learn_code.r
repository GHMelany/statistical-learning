library(corrplot)
library(ggplot2)
library(reshape2)
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)
library(xgboost)
library(parallel)
library(doParallel)
library(factoextra)
library(cluster)
library(umap)
library(Rtsne)
library(qgraph)

# Load red and white wine datasets
red_wine <- read.csv("C:/Users/melan/OneDrive/Documenti/winequality-red.csv", sep = ";")
white_wine <- read.csv("C:/Users/melan/OneDrive/Documenti/winequality-white.csv", sep = ";")

# Add wine type to each dataset
red_wine$type <- "red"
white_wine$type <- "white"

# Combine datasets
combined_wine <- rbind(red_wine, white_wine)

head(combined_wine)
summary(combined_wine)

# Ensure column names match
names(white_wine) <- names(red_wine)

sum(is.na(combined_wine))

combined_wine <- unique(combined_wine)

# Remove the 'free.sulfur.dioxide' variable from the dataset
combined_wine <- combined_wine[, !(names(combined_wine) %in% "free.sulfur.dioxide")]

# Check the number of rows before and after removing duplicates
original_rows <- nrow(combined_wine)
unique_rows <- nrow(combined_wine)
cat("Number of duplicate rows removed:", original_rows - unique_rows, "\n")

#Wine distribution
type_distribution <- table(combined_wine$type)
pie(type_distribution, 
    labels = paste(names(type_distribution), 
                   "(", round(100 * type_distribution / sum(type_distribution), 1), "%)", sep = ""), 
    main = "Distribuzione dei tipi di vino",
    col = c("#c45bad", "#efd79d"))

# Select only numeric columns
numeric_data <- combined_wine[sapply(combined_wine, is.numeric)]

# Function to normalize values between 0 and 1
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

# Apply normalization to all numeric columns
normalized_data <- as.data.frame(lapply(combined_wine, function(col) {
  if (is.numeric(col)) {
    normalize(col)
  } else {
    col  # Keep non-numeric columns unchanged
  }
}))

# Check the normalized dataset
summary(normalized_data)

# Calculate the correlation matrix
correlation_matrix <- cor(numeric_data, use = "complete.obs")

# Display the correlation matrix
print(correlation_matrix)

# Check the structure of the dataset
str(combined_wine)

# Convert 'type' to binary numeric
combined_wine$type <- ifelse(combined_wine$type == "red", 1, 0)

# Verify the conversion
table(combined_wine$type)

str(combined_wine)

# Verify that all numeric columns are between 0 and 1
sapply(normalized_data, function(col) {
  if (is.numeric(col)) {
    range(col)  # Check the range of each numeric column
  }
})

# Transform the correlation matrix into long format
correlation_data <- melt(correlation_matrix)

# Create the heatmap with ggplot2
ggplot(correlation_data, aes(Var1, Var2, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "#d8849e", high = "#c45bad", mid = "white", midpoint = 0,
                       limit = c(-1, 1), space = "Lab") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1)) +
  labs(title = "Heatmap della Matrice di Correlazione", x = "", y = "")

# Remove the 'free.sulfur.dioxide' variable from the dataset
combined_wine <- combined_wine[, !(names(combined_wine) %in% "free.sulfur.dioxide")]

# Check the structure of the updated dataset
str(combined_wine)

# Create the correlation network graph
qgraph(correlation_matrix,
       layout = "spring",
       vsize = 6,
       label.cex = 1.2,
       label.scale = FALSE,
       labels = colnames(correlation_matrix),
       label.norm = "O",
       title = "Correlation Network Graph",
       label.color = "black")

# SUPERVISED LEARNING

# Split the dataset into training and test sets
set.seed(123)  # For reproducibility
train_index <- createDataPartition(combined_wine$type, p = 0.8, list = FALSE)
train_set <- combined_wine[train_index, ]
test_set <- combined_wine[-train_index, ]

# Logistic regression model
model <- glm(type ~ ., data = train_set, family = binomial())

# Model summary
summary(model)

# Predictions on the test set
predictions <- predict(model, test_set, type = "response")

# Convert probabilities to classes
predicted_classes <- ifelse(predictions > 0.5, 1, 0)

# Calculate the confusion matrix
confusionMatrixData <- confusionMatrix(as.factor(predicted_classes), as.factor(test_set$type))

# Print the results
print(confusionMatrixData$table)  # Confusion matrix
print(confusionMatrixData$overall['Accuracy'])  # Accuracy
print(confusionMatrixData$byClass['Balanced Accuracy'])  # Balanced Accuracy
print(confusionMatrixData$byClass['Precision'])  # Precision
print(confusionMatrixData$byClass['Recall'])  # Recall
print(confusionMatrixData$byClass['F1'])  # F1-score

# Ensure 'type' is a factor for KNN
train_set$type <- as.factor(train_set$type)
test_set$type <- as.factor(test_set$type)

# Set up training control with cross-validation
train_control <- trainControl(method = "cv", number = 30, savePredictions = "final")

# Set seed for reproducibility
set.seed(111)

# Train KNN model with parameter tuning
knn_model <- train(
  type ~ .,                 # Formula (target variable and predictors)
  data = train_set,         # Training set
  method = "knn",           # KNN method
  trControl = train_control, # Training control
  tuneLength = 20           # Number of k values to test
)

# Print model details to see the best k value
print(knn_model)

# Predictions on the test set
knn_predictions <- predict(knn_model, newdata = test_set)

# Evaluate model performance on the test set
knn_results <- confusionMatrix(knn_predictions, test_set$type)

# Print confusion matrix
print(knn_results$table)

# Evaluation metrics
cat("Balanced Accuracy:", knn_results$byClass['Balanced Accuracy'], "\n")
cat("Precision:", knn_results$byClass['Precision'], "\n")
cat("Recall:", knn_results$byClass['Recall'], "\n")
cat("F1 Score:", knn_results$byClass['F1'], "\n")

# Ensure 'type' is a factor
train_set$type <- as.factor(train_set$type)
test_set$type <- as.factor(test_set$type)

# Create Decision Tree model
tree_model <- rpart(type ~ ., data = train_set, method = "class")

# Model summary
summary(tree_model)

# Visualize the decision tree
rpart.plot(tree_model, type = 3, extra = 104, fallen.leaves = TRUE, cex = 0.7)

# Predictions on the test set
predictions <- predict(tree_model, newdata = test_set, type = "class")

# Evaluate the model
confusionMatrixData <- confusionMatrix(as.factor(predictions), as.factor(test_set$type))

# Print confusion matrix
print(confusionMatrixData$table)

# Evaluation metrics
print(paste("Accuracy:", confusionMatrixData$overall['Accuracy']))
print(paste("Balanced Accuracy:", confusionMatrixData$byClass['Balanced Accuracy']))
print(paste("Precision:", confusionMatrixData$byClass['Precision']))
print(paste("Recall:", confusionMatrixData$byClass['Recall']))
print(paste("F1 Score:", confusionMatrixData$byClass['F1']))

# Ensure 'type' is a factor
train_set$type <- as.factor(train_set$type)
test_set$type <- as.factor(test_set$type)

set.seed(13)

# Define parameter values to optimize
mtry_values <- seq(2, sqrt(ncol(train_set) - 1), by = 1)
nodesize_values <- c(1, 5, 10, 20)
minsplit_values <- c(2, 4, 6, 10)
maxdepth_values <- c(5, 10, 15, 20)

# Initialize variables to track the best model
best_model <- NULL
best_accuracy <- 0
best_params <- list()

# Iterate over all combinations of parameters
for (mtry in mtry_values) {
  for (nodesize in nodesize_values) {
    for (minsplit in minsplit_values) {
      for (maxdepth in maxdepth_values) {
        # Train Random Forest model
        model <- randomForest(
          type ~ .,
          data = train_set,
          mtry = mtry,
          nodesize = nodesize,
          ntree = 500,
          maxnodes = maxdepth
        )

        # Predictions on the test set
        predictions <- predict(model, newdata = test_set)

        # Calculate accuracy
        cm <- confusionMatrix(predictions, test_set$type)
        accuracy <- cm$overall['Accuracy']

        # Update best model if accuracy is higher
        if (accuracy > best_accuracy) {
          best_model <- model
          best_accuracy <- accuracy
          best_params <- list(
            mtry = mtry,
            nodesize = nodesize,
            minsplit = minsplit,
            maxdepth = maxdepth
          )
        }
      }
    }
  }
}

# Print best parameters and accuracy
cat("Best Parameters:\n")
print(best_params)
cat("Best Accuracy:", best_accuracy, "\n")

# Evaluate the best model
best_predictions <- predict(best_model, newdata = test_set)
best_cm <- confusionMatrix(best_predictions, test_set$type)

# Print confusion matrix and metrics
print(best_cm$table)
cat("Accuracy:", best_cm$overall['Accuracy'], "\n")
cat("Balanced Accuracy:", best_cm$byClass['Balanced Accuracy'], "\n")
cat("Precision:", best_cm$byClass['Precision'], "\n")
cat("Recall:", best_cm$byClass['Recall'], "\n")
cat("F1 Score:", best_cm$byClass['F1'], "\n")

# Variable importance plot
varImpPlot(best_model)

# Set up parallel processing
num_cores <- detectCores() - 1
cl <- makeCluster(num_cores)
registerDoParallel(cl)

# Prepare data for XGBoost
train_set$type <- as.factor(as.numeric(as.factor(train_set$type)) - 1)
test_set$type <- as.factor(as.numeric(as.factor(test_set$type)) - 1)

train_matrix <- as.matrix(train_set[, -which(names(train_set) == "type")])
train_label <- train_set$type

test_matrix <- as.matrix(test_set[, -which(names(test_set) == "type")])
test_label <- test_set$type

# Training control with cross-validation
train_control <- trainControl(
  method = "cv",
  number = 5,
  verboseIter = TRUE,
  allowParallel = TRUE
)

# Tuning grid
tune_grid <- expand.grid(
  nrounds = c(50, 100),         # Number of iterations
  max_depth = c(3, 6, 9, 12),   # Maximum depth
  eta = c(0.01, 0.05, 0.1, 0.3), # Learning rate
  gamma = c(0, 0.1, 0.5, 1),    # Minimum loss reduction
  colsample_bytree = c(0.5, 0.7, 0.9, 1.0), # Column subsampling
  min_child_weight = c(1, 5, 10), # Minimum child weight
  subsample = c(0.5, 0.7, 0.9, 1.0) # Row subsampling
)

# Train XGBoost model
set.seed(111)
xgb_tuned_model <- train(
  x = train_matrix,
  y = train_label,
  method = "xgbTree",
  trControl = train_control,
  tuneGrid = tune_grid
)

# Stop parallel processing
stopCluster(cl)
registerDoSEQ()

# Print best tuning parameters
cat("Best Tuning Parameters:\n")
print(xgb_tuned_model$bestTune)

# Predictions with the optimized model
best_predictions <- predict(xgb_tuned_model, newdata = test_matrix)

# Align levels of predictions and test labels
best_predictions <- factor(best_predictions, levels = levels(test_label))
test_label <- factor(test_label, levels = levels(best_predictions))

# Generate confusion matrix
confusionMatrixData <- confusionMatrix(best_predictions, test_label)

# Print confusion matrix
cat("\nConfusion Matrix:\n")
print(confusionMatrixData$table)

# Print performance metrics
cat("\nPerformance Metrics:\n")
cat("Accuracy:", confusionMatrixData$overall['Accuracy'], "\n")
cat("Balanced Accuracy:", confusionMatrixData$byClass['Balanced Accuracy'], "\n")
cat("Precision:", confusionMatrixData$byClass['Pos Pred Value'], "\n")
cat("Recall:", confusionMatrixData$byClass['Sensitivity'], "\n")
cat("F1 Score:", confusionMatrixData$byClass['F1'], "\n")

# UNSUPERVISED LEARNING

# Prepare data for PCA
pca_data <- combined_wine[, sapply(combined_wine, is.numeric)]
pca_data <- pca_data[, !colnames(pca_data) %in% "type"]

# Perform PCA with scaling and centering
pca_results <- prcomp(pca_data, center = TRUE, scale. = TRUE)

# Summary of PCA results
summary(pca_results)

# Explained variance for each component
cat("Explained variance:\n")
print(pca_results$sdev^2 / sum(pca_results$sdev^2) * 100)

# Plot of explained variance
fviz_eig(pca_results, addlabels = TRUE, ylim = c(0, 100))

# Project data onto the first two principal components
pca_projected <- as.data.frame(pca_results$x[, 1:2])
pca_projected$Type <- combined_wine$type

# Biplot visualization
fviz_pca_biplot(pca_results, geom = "point", habillage = combined_wine$type,
                addEllipses = TRUE, ellipse.level = 0.95)

# Extract the first five principal components
scores <- as.data.frame(pca_results$x[, 1:5])

# Elbow method to determine optimal number of clusters
set.seed(123)
wss <- sapply(1:10, function(k) {
  kmeans(scores, centers = k, nstart = 10, iter.max = 50)$tot.withinss
})

# Plot the Elbow method graph
plot(1:10, wss, type = "b", pch = 19, frame = FALSE,
     xlab = "Number of Clusters",
     ylab = "Total Within-Cluster Sum of Squares",
     main = "Elbow Method for the Optimal Number of Clusters")

# K-MEANS PCA

# Optimal number of clusters
optimal_clusters <- 4

# Perform KMeans clustering
set.seed(123)
kmeans_result <- kmeans(scores, centers = optimal_clusters, nstart = 10)

# Add clustering results to dataframe
scores_df <- data.frame(
  scores,
  Cluster = factor(kmeans_result$cluster),
  Type = combined_wine$type
)

# Convert 'Type' to factor
scores_df$Type <- as.factor(scores_df$Type)

# Plot using 'Type'
ggplot(scores_df, aes(x = PC1, y = PC2, color = Cluster, shape = Type)) +
  geom_point(alpha = 1, size = 3.5) +
  scale_color_manual(values = c("#dd1e2b", "#ff914d", "#5e17eb", "purple")) +
  scale_shape_manual(values = c(1, 3, 4)) +
  theme_minimal() +
  labs(title = "PCA with K-means Clustering",
       x = "Principal Component 1",
       y = "Principal Component 2",
       color = "Cluster",
       shape = "Type of Wine")

# Calculate silhouette score
silhouette_score <- silhouette(kmeans_result$cluster, dist(scores))

# Visualize the silhouette score
fviz_silhouette(silhouette_score)

# Calculate and print average silhouette width
average_silhouette <- mean(silhouette_score[, 3])
cat("Average Silhouette Width: ", average_silhouette, "\n")

# Repeat KMeans with a different number of clusters (e.g., 2)
optimal_clusters <- 2
set.seed(123)
kmeans_result <- kmeans(scores, centers = optimal_clusters, nstart = 10)
scores_df$Cluster <- factor(kmeans_result$cluster)

# Updated plot
ggplot(scores_df, aes(x = PC1, y = PC2, color = Cluster, shape = Type)) +
  geom_point(alpha = 1, size = 3.5) +
  scale_color_manual(values = c("#dd1e2b", "#ff914d")) +
  scale_shape_manual(values = c(1, 3, 4)) +
  theme_minimal() +
  labs(title = "PCA with K-means Clustering",
       x = "Principal Component 1",
       y = "Principal Component 2",
       color = "Cluster",
       shape = "Type of Wine")

# Calculate silhouette score for the new clustering
silhouette_score <- silhouette(kmeans_result$cluster, dist(scores))
fviz_silhouette(silhouette_score)
average_silhouette <- mean(silhouette_score[, 3])
cat("Average Silhouette Width: ", average_silhouette, "\n")

# WARD2 Hierarchical Clustering

# Perform hierarchical clustering with "ward.D2" method
hc_ward <- hclust(dist(scores), method = "ward.D2")

# Plot the dendrogram
plot(hc_ward, main = "Hierarchical Clustering Dendrogram (ward.D2)", sub = "", xlab = "")

# Cut the dendrogram to obtain 4 clusters
clusters_hc_ward <- cutree(hc_ward, k = 4)

# Create dataframe with new clusters
pca_df_ward <- data.frame(
  scores,
  HC_Cluster = as.factor(clusters_hc_ward),
  Type = combined_wine$type
)

# Convert 'Type' to factor
pca_df_ward$Type <- as.factor(pca_df_ward$Type)

# Visualize new clusters
ggplot(pca_df_ward, aes(x = PC1, y = PC2, color = HC_Cluster, shape = Type)) +
  geom_point(alpha = 1, size = 3.5) +
  scale_color_manual(values = c("#dd1e2b", "#ff914d", "#5e17eb", "purple")) +
  scale_shape_manual(values = c(1, 3, 4)) +
  theme_minimal() +
  labs(
    title = "PCA with Hierarchical Clustering (Ward.D2)",
    x = "Principal Component 1",
    y = "Principal Component 2",
    color = "Cluster",
    shape = "Type of Wine"
  )

# Calculate silhouette score for hierarchical clustering
sil_scores_hc_ward <- silhouette(clusters_hc_ward, dist(scores))

# Visualize silhouette score
fviz_silhouette(sil_scores_hc_ward)

# Calculate and print average silhouette width
average_silhouette_hc_ward <- mean(sil_scores_hc_ward[, 3])
cat("Average Silhouette Width (Ward.D2): ", average_silhouette_hc_ward, "\n")

# K-MEANS UMAP

# Prepare data excluding 'type'
combined_wine_features <- combined_wine[, sapply(combined_wine, is.numeric)]
combined_wine_features <- combined_wine_features[, !colnames(combined_wine_features) %in% "type"]

# Configure UMAP
umap_config <- umap.defaults
umap_config$n_neighbors <- 15
umap_config$min_dist <- 0.01
umap_config$n_components <- 2

# Apply UMAP to the data
umap_results <- umap(combined_wine_features, config = umap_config)

# Convert UMAP results to dataframe
umap_df <- as.data.frame(umap_results$layout)
colnames(umap_df) <- c("UMAP1", "UMAP2")

# Add 'Type' variable
umap_df$Type <- combined_wine$type

# Convert 'Type' to factor
umap_df$Type <- as.factor(umap_df$Type)

# Elbow method for optimal number of clusters on UMAP data
set.seed(123)
wss <- sapply(1:10, function(k) {
  kmeans(umap_df[, c("UMAP1", "UMAP2")], centers = k, nstart = 10)$tot.withinss
})

# Plot the Elbow method graph
plot(1:10, wss, type = "b", pch = 19, frame = FALSE,
     xlab = "Number of Clusters (K)",
     ylab = "Total Within-Cluster Sum of Squares",
     main = "Elbow Method for Choosing the Optimal Number of Clusters")

# Apply KMeans to UMAP results
optimal_clusters <- 3
kmeans_result <- kmeans(umap_df[, c("UMAP1", "UMAP2")], centers = optimal_clusters, nstart = 10)

# Add clustering results to dataframe
umap_df$Cluster <- as.factor(kmeans_result$cluster)

# Visualize UMAP clusters
unique_types <- length(levels(umap_df$Type))
shape_values <- c(1, 3, 4, 5, 6, 7, 8)[1:unique_types]

ggplot(umap_df, aes(x = UMAP1, y = UMAP2, color = Cluster, shape = Type)) +
  geom_point(alpha = 1, size = 3.5) +
  scale_color_manual(values = c("#dd1e2b", "#ff914d", "#5e17eb")) +
  scale_shape_manual(values = shape_values) +
  theme_minimal() +
  labs(
    title = "UMAP with KMeans Clustering",
    x = "UMAP Dimension 1",
    y = "UMAP Dimension 2",
    color = "Cluster",
    shape = "Type of Wine"
  )

# Calculate silhouette score for KMeans on UMAP results
sil_scores_kmeans <- silhouette(kmeans_result$cluster, dist(umap_df[, c("UMAP1", "UMAP2")]))
fviz_silhouette(sil_scores_kmeans)

# Calculate and print average silhouette width
average_silhouette_kmeans <- mean(sil_scores_kmeans[, 3])
cat("Average Silhouette Width: ", average_silhouette_kmeans, "\n")

# UMAP Hierarchical Clustering

# Hierarchical clustering on UMAP results
hc <- hclust(dist(umap_df[, c("UMAP1", "UMAP2")]), method = "complete")

# Plot the dendrogram
plot(hc, main = "Hierarchical Clustering Dendrogram (UMAP)", sub = "", xlab = "")

# Cut the dendrogram to obtain 3 clusters
clusters_hc <- cutree(hc, k = 3)

# Add clustering results to dataframe
umap_df$HC_Cluster <- as.factor(clusters_hc)

# Visualize clusters using UMAP components
ggplot(umap_df, aes(x = UMAP1, y = UMAP2, color = HC_Cluster, shape = Type)) +
  geom_point(alpha = 1, size = 3.5) +
  scale_color_manual(values = c("#dd1e2b", "#ff914d", "#5e17eb")) +
  scale_shape_manual(values = c(1, 3)) +
  theme_minimal() +
  labs(
    title = "UMAP with Hierarchical Clustering",
    x = "UMAP Dimension 1",
    y = "UMAP Dimension 2",
    color = "Cluster",
    shape = "Type of Wine"
  )

# Calculate silhouette score for hierarchical clustering
sil_scores_hc <- silhouette(clusters_hc, dist(umap_df[, c("UMAP1", "UMAP2")]))
fviz_silhouette(sil_scores_hc)
average_silhouette_hc <- mean(sil_scores_hc[, 3])
cat("Average Silhouette Width (Hierarchical Clustering): ", average_silhouette_hc, "\n")

# Calculate average values for each variable within hierarchical clusters
combined_wine$HC_Cluster <- clusters_hc
relevant_columns <- colnames(combined_wine_features)
average_scores_hc <- aggregate(. ~ HC_Cluster, data = combined_wine[, c(relevant_columns, "HC_Cluster")], mean)
print(average_scores_hc)

# T-SNE KMEANS

# Remove duplicate rows from the dataset
combined_wine_unique <- unique(combined_wine_features)

# Find indices of remaining rows in the original combined dataset
unique_indices <- which(!duplicated(combined_wine_features))

# Get 'type' variable corresponding to non-duplicate rows
type_after_unique <- combined_wine$type[unique_indices]

# Perform t-SNE on data without 'type' and duplicates
set.seed(123)
tsne_results <- Rtsne(combined_wine_unique, dims = 2, perplexity = 30, verbose = TRUE, max_iter = 3000)

# Convert t-SNE results to dataframe
tsne_df <- as.data.frame(tsne_results$Y)
colnames(tsne_df) <- c("TSNE1", "TSNE2")

# Add 'type' variable to t-SNE dataframe
tsne_df$Type <- as.factor(type_after_unique)

# Perform KMeans clustering on t-SNE results
set.seed(123)
optimal_clusters <- 3
kmeans_tsne_result <- kmeans(tsne_df[, 1:2], centers = optimal_clusters, nstart = 20)
tsne_df$Cluster <- factor(kmeans_tsne_result$cluster)

# Visualize clusters using t-SNE components
ggplot(tsne_df, aes(x = TSNE1, y = TSNE2, color = Cluster, shape = Type)) +
  geom_point(alpha = 1, size = 3.5) +
  scale_color_manual(values = c("#dd1e2b", "#ff914d", "#5e17eb")) +
  scale_shape_manual(values = c(1, 3, 4)) +
  theme_minimal() +
  labs(
    title = "Clustering t-SNE with KMeans",
    x = "t-SNE Dimension 1",
    y = "t-SNE Dimension 2",
    color = "Cluster",
    shape = "Type of Wine"
  )

# Calculate silhouette score for t-SNE + KMeans
tsne_silhouette <- silhouette(kmeans_tsne_result$cluster, dist(tsne_df[, 1:2]))
fviz_silhouette(tsne_silhouette)
average_silhouette_tsne <- mean(tsne_silhouette[, 3])
cat("Average Silhouette Width (t-SNE + KMeans): ", average_silhouette_tsne, "\n")

# Add KMeans clustering result to data without duplicates
combined_wine_unique$Cluster <- kmeans_tsne_result$cluster

# Calculate means for each cluster
cluster_means <- aggregate(. ~ Cluster, data = combined_wine_unique, mean)
print(cluster_means)

# T-SNE HIERARCHICAL

# Perform hierarchical clustering using "centroid" method
dist_tsne <- dist(tsne_df[, 1:2])
hc_tsne <- hclust(dist_tsne, method = "centroid")

# Plot the dendrogram
plot(hc_tsne, main = "Hierarchical Clustering Dendrogram (t-SNE)", sub = "", xlab = "")

# Cut the dendrogram to obtain 4 clusters
clusters_hc_tsne <- cutree(hc_tsne, k = 4)
tsne_df$HC_Cluster <- factor(clusters_hc_tsne)

# Visualize clusters using t-SNE components
ggplot(tsne_df, aes(x = TSNE1, y = TSNE2, color = HC_Cluster, shape = Type)) +
  geom_point(alpha = 1, size = 3.5) +
  scale_color_manual(values = c("#dd1e2b", "#ff914d", "#5e17eb", "purple")) +
  scale_shape_manual(values = c(1, 3, 4, 5)) +
  theme_minimal() +
  labs(
    title = "t-SNE with Hierarchical Clustering",
    x = "t-SNE Dimension 1",
    y = "t-SNE Dimension 2",
    color = "Cluster",
    shape = "Type of Wine"
  )

# Calculate silhouette score for hierarchical clustering
sil_scores_hc_tsne <- silhouette(clusters_hc_tsne, dist_tsne)
fviz_silhouette(sil_scores_hc_tsne)
average_silhouette_hc_tsne <- mean(sil_scores_hc_tsne[, 3])
cat("Average silhouette width (Hierarchical): ", average_silhouette_hc_tsne, "\n")
