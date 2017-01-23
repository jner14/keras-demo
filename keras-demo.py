ver = 6
print("Starting keras-demo.py V0.{}".format(ver))


from keras.models import Sequential
from keras.layers import Dense
from pandas import DataFrame, read_csv
import numpy as np


# PARAMETERS
data_dir      = "data/"
data_filename = "pima-indians-diabetes.csv"


# fix random seed for reproducibility
seed = 8
np.random.seed(seed)

# load pima indians dataset
dataset = read_csv(data_dir + data_filename, header=None)

# split into input (X) and output (Y) training and test variables
msk = np.random.rand(len(dataset)) < 0.8
trainX = dataset[msk].loc[:, :7]
trainY = dataset[msk].loc[:, 8]
testX = dataset[~msk].loc[:, :7]
testY = dataset[~msk].loc[:, 8]

# create model
model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(trainX.as_matrix(), trainY.as_matrix(), nb_epoch=150, batch_size=10)

# evaluate the model
scores = model.evaluate(testX.as_matrix(), testY.as_matrix())
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# calculate predictions
# predictions = model.predict(X)
# rounded = [round(x) for x in predictions]
# print(rounded)

print("Finished V0.{}".format(ver))
