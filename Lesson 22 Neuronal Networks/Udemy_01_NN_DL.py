import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

df = pd.read_csv(r"D:\Documentos\Trabajo\Udemy\TensorFlow_FILES\DATA\fake_reg.csv")

print(df.head())
print('\n')

sns.set_style('whitegrid')
sns.pairplot(df, height=3.5, aspect=1.5)
plt.show()

from sklearn.model_selection import train_test_split

# Because the way that TensorFlow it actually works, we've to pass numpy arrays instead of pandas data frames or pd
# series.
# We can do it by the .values and it will return it back as numpy array

X = df[['feature1', 'feature2']].values

y = df['price'].values

print('X in numpy arrays')
print(X)
print('\n')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(X_train.shape)
print('\n')
print(X_test.shape)

# We have to normalize our data, if we've very large values inside our data, that could cost errors with the weight

from sklearn.preprocessing import MinMaxScaler

# Transform features by scaling each feature to a given range.
# This estimator scales and translates each feature individually such
# that it is in the given range on the training set, e.g. between
# zero and one.

# We don't need to scale the label

# help(MinMaxScaler)

scaler = MinMaxScaler()

scaler.fit(X_train)  # We calculate what it's need it to make the standardization

# We only run it in the training set, because we want to prevent what it's known as data leakage from the test set:
# We don't want to assume that we have prior information of test set.

X_train = scaler.transform(X_train)  # We make the transformation

X_test = scaler.transform(X_test)

from tensorflow.keras import Sequential  # Model
from tensorflow.keras.layers import Dense  # Layers

model1 = Sequential([Dense(4, activation='relu'),
                    Dense(2, activation='relu'),
                    Dense(1)])  # We pass a list of the actual layer we want, regular
# densely-connected NN layer; each
# neuron it's going to be connected to every other neuron in the next layer.
# Parameters: Units= the number of neurons that it's going to be in this layer; Activation= takes in a string call for
#  what function these neurons should be using (sigmoid activation, rectified linear unit ('relu'...)


# Way two for build a neuronal network

model = Sequential()

model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='relu'))

model.add(Dense(1))  # Very determinant, as it only has to predict the price, we only really want one output

model.compile(optimizer='rmsprop', loss='mse')
# The optimizer parameter; It's essentially asking how do we want to perform this gradient descent, if we want to use
# "rmsprop" or the "adam" The loss parameter; it's going to change dependent in what actually you're trying to
# accomplish, so, if we want to determine the loss parameter for multi class classification, we would use
# 'categorical_crossentropy", for binary, "binary_crossentropy", and for regression, "mse"


model.fit(x=X_train, y=y_train, epochs=250)


# Sample: one element of a dataset.
#  Example: one image is a sample in a convolutional network
#  Example: one audio file is a sample for a speech recognition model

# Batch: a set of N samples. The samples in a batch are processed independently, in parallel. If training, a batch
# results in only one update to the model. A batch generally approximates the distribution of the input data better than
# a single input. The larger the batch, the better the approximation; however, it is also true that the batch will take
# longer to process and will still result in only one update. For inference (evaluate/predict), it is recommended to
# pick a batch size that is as large as you can afford without going out of memory (since larger batches will usually
# result in faster evaluation/prediction).

# Epoch: an arbitrary cutoff, generally defined as "one pass over the entire dataset", used to separate training into
# distinct phases, which is useful for logging and periodic evaluation.

# When using validation_data or validation_split with the fit method of Keras models, evaluation will be run at the end
# of every epoch.
# Within Keras, there is the ability to add callbacks specifically designed to be run at the end of an epoch. Examples
# of these are learning rate changes and model checkpointing (saving).


# Verbose parameter, print the report in the run console. if it's 0, it doesn't print anything

print(model.history.history)
# We call all the loss history of the model fit, it's going be dictionary

sns.set_style('whitegrid')
loss_df = pd.DataFrame(model.history.history)
loss_df.plot()
plt.show()

# Evaluation the model
# How well perform in our test data

print('Training loss error')
print(model.evaluate(X_train, y_train, verbose=0))
print('\n Test loss error')
print(model.evaluate(X_test, y_test, verbose=0))

# It calculate the mean squared error

# It return back our model loss in the test data, the number represents the metric
# loss of our model, in this case mean square error

test_predictions = model.predict(X_test)

print(test_predictions)
print('\n')
print(test_predictions.shape)
print('\n')
print(type(test_predictions))
print('\n')

test_predictions = pd.Series(test_predictions.reshape(300,))  # We've to reshape the numpy array, to match the
# expectations of pandas series expects
print('Pandas Data Frame')
print(test_predictions.head())
print('\n')

print(type(y_test))
print('\n')
pred_df = pd.DataFrame(y_test, columns=[' Test True Y'])

pred_df = pd.concat([pred_df, test_predictions], axis=1)
pred_df.columns = ['Test True Y', 'Model Predictions']
print(pred_df.head())
print(' ')
sns.scatterplot(x='Test True Y', y='Model Predictions', data=pred_df)
plt.show()


from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(pred_df['Test True Y'], pred_df['Model Predictions'])
print(mae)
print('\n')
mse = mean_squared_error(pred_df['Test True Y'], pred_df['Model Predictions'])
print(mse)
print('\n')
RMSE = np.sqrt(mse)
print(RMSE)
print('\n')
# Root mean squared error
print(df.describe())

# Predicting in brand new data

new_gem = [[998, 1000]]

# First of all, we 've to remember that our model it's trained with scales features, so I have to scale this new gem

new_gem = scaler.transform(new_gem)

print(model.predict(new_gem))

from tensorflow.keras.models import load_model

# We can save the model by importing the top function

model.save('my_gem_model.h5')

# To run it in an other file, we import the function load model.

later_model = load_model('my_gem_model.h5')
