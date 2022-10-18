import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

CURR = os.getcwd()
df = pd.read_csv(os.path.join(CURR, 'dataset', 'concrete_data_edit.csv'))

X = df.drop('strength', axis = 1) # features
y = df['strength'] # outcome

from scipy.stats import zscore
Xscaled = X.apply(zscore)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Xscaled, y, test_size = 0.20, random_state= 42)# default = 0.25
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.20, random_state= 42)# default = 0.25

# ======================
## Without batch
# ======================
keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

run_index = 1 # increment every time you train the model

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2, #1e-2, 1e-3
    decay_steps=10000,
    decay_rate=0.9)

model = keras.models.Sequential()
model.add(keras.layers.Dense(100, activation='relu', input_shape=X_train.shape[1:]))
for _ in range(20):
    model.add(keras.layers.Dense(100, kernel_initializer="lecun_normal"))
    model.add(keras.layers.Activation("relu"))

model.add(keras.layers.AlphaDropout(rate=0.1)) # try rate 5%, 10%, 20% and 40%    
model.add(keras.layers.Dense(1))

optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
model.compile(loss='mean_squared_error', optimizer=optimizer)

early_stopping_cb = keras.callbacks.EarlyStopping(patience=20)
model_checkpoint_cb = keras.callbacks.ModelCheckpoint("mix_design.h5", save_best_only=True)
run_logdir = os.path.join(os.curdir, "mix_design_logs", "run_alpha_dropout_{:03d}".format(run_index))
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
callbacks = [early_stopping_cb, model_checkpoint_cb, tensorboard_cb]

history = model.fit(X_train, y_train, epochs=100,
    validation_data=(X_valid, y_valid),
    callbacks=callbacks)

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 500)
plt.show()

model.evaluate(X_test, y_test)

model.predict(X_test[1:10])
print(y_test[1:10])

# ======================
## With batch --> long time training
# ======================
# save dataset to multiple csv file
keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

from sklearn.preprocessing import StandardScaler
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42)

scaler = StandardScaler()
scaler.fit(X_train)
mean = scaler.mean_
std = scaler.scale_

## gathering 8-features and 1-outcome by concatinate 
train_data = np.c_[X_train, y_train] 
valid_data = np.c_[X_valid, y_valid]
test_data = np.c_[X_test, y_test]
header_cols = df.columns
header = ",".join(header_cols)

## split dataset to multiple file & save --> "new_data" directory
from multi_csv import save_to_multiple_csv_files

train_filepaths = save_to_multiple_csv_files(train_data, "train", header, n_parts=20)
valid_filepaths = save_to_multiple_csv_files(valid_data, "valid", header, n_parts=10)
test_filepaths = save_to_multiple_csv_files(test_data, "test", header, n_parts=10)

# ------------
# read csv
from read_csv import readCSV

instance = readCSV(mean=mean, std=std, n_inputs=8) # features column = 8
train_set = instance.csv_reader_dataset(train_filepaths, repeat=None)
valid_set = instance.csv_reader_dataset(valid_filepaths)
test_set = instance.csv_reader_dataset(test_filepaths)

# ------------
keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

run_index = 1 # increment every time you train the model

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2, #1e-2, 1e-3, 1e-4, 1e-5
    decay_steps=10000,
    decay_rate=0.9)

model = keras.models.Sequential()
model.add(keras.layers.Dense(30, activation='relu', input_shape=X_train.shape[1:]))
for _ in range(4):
    # model.add(keras.layers.Dense(30, kernel_initializer="lecun_normal"))
    model.add(keras.layers.Dense(30))
    model.add(keras.layers.Activation("relu"))

model.add(keras.layers.AlphaDropout(rate=0.2)) # try rate 5%, 10%, 20% and 40%    
model.add(keras.layers.Dense(1))

model.summary()

optimizer = keras.optimizers.RMSprop(learning_rate=lr_schedule)
model.compile(loss='mean_squared_error', optimizer=optimizer)

early_stopping_cb = keras.callbacks.EarlyStopping(patience=20)
model_checkpoint_cb = keras.callbacks.ModelCheckpoint("mix_design_batch.h5", save_best_only=True)
run_logdir = os.path.join(os.curdir, "mix_design_batch_logs", "run_alpha_dropout_{:03d}".format(run_index))
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
callbacks = [early_stopping_cb, model_checkpoint_cb, tensorboard_cb]

batch_size = 32
history = model.fit(train_set, epochs=1,
    validation_data=(valid_set),
    callbacks=callbacks)

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 100)
plt.show()

model.evaluate(test_set, steps=len(X_test) // batch_size)

new_set = test_set.map(lambda X, y: X) # select only features
model.predict(new_set, steps=len(X_test) // batch_size)

