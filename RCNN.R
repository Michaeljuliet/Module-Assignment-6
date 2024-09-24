library(keras)

# Load the dataset
fashion_mnist <- dataset_fashion_mnist()
train_images <- fashion_mnist$train$x / 255
train_labels <- fashion_mnist$train$y
test_images <- fashion_mnist$test$x / 255
test_labels <- fashion_mnist$test$y


model <- keras_model_sequential() %>%
  # First Convolutional Layer
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = 'relu', input_shape = c(28, 28, 1)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  
  # Second Convolutional Layer
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  
  # Third Convolutional Layer
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = 'relu') %>%
  
  # Flatten
  layer_flatten() %>%
  
  # Dense (Fully Connected) Layer
  layer_dense(units = 64, activation = 'relu') %>%
  
  # Output Layer
  layer_dense(units = 10, activation = 'softmax')

# Compile the model
model %>% compile(
  optimizer = 'adam',
  loss = 'sparse_categorical_crossentropy',
  metrics = c('accuracy')
)

# Fit the model
model %>% fit(
  train_images, train_labels, epochs = 5, validation_data = list(test_images, test_labels)
)


predictions <- model %>% predict(test_images)

# Show predictions for two images
cat("Prediction for first image:", which.max(predictions[1, ]) - 1, "\n")
cat("Prediction for second image:", which.max(predictions[2, ]) - 1, "\n")


