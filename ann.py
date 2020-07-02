import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout

from tensorflow.keras.callbacks import TensorBoard

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

from scipy import stats
from data import get_data


# Import testing and training dataset
def importData():
    X_train, y_train, X_test = get_data()
    X_train['Target'] = y_train

    train_dataset = X_train.sample(frac=0.8)
    train_dataset.to_csv('data/train.csv', index=False)
    print(len(train_dataset.index))
    test_dataset = X_train.drop(train_dataset.index)
    test_dataset.to_csv('data/test.csv', index=False)

    print(train_dataset.info())
    print(test_dataset.info())

    return train_dataset, test_dataset


# Seperate training and testing label from training and testing dataset
def prepareLabels(train_dataset, test_dataset):
    train_labels = train_dataset.pop('Target')
    test_labels = test_dataset.pop('Target')

    print('===Labels===')
    print(train_labels)
    print(test_labels)

    return train_labels, test_labels


# Returns the stats of the dataset
def inspectDataSet(dataset):
    stats = dataset.describe()
    stats = stats.transpose()
    print('===Stats===')
    print(stats)
    return stats


# Normalize the data
def normalize(x, train_stats):
    return (x - train_stats['mean']) / train_stats['std']


# Build and compile model and returns it
def build_model(train_dataset, layers, dropouts):
    model = keras.Sequential()

    # Input layer
    model.add(Dropout(dropouts[0], input_shape=[len(train_dataset.keys())]))

    # 1st hidden layer, we pop the 1st element from the layers list
    model.add(Dense(layers.pop(0), activation='relu'))
    # model.add(Dense(layers.pop(0), activation='elu'))
    model.add(Dropout(dropouts[1]))

    # other hidden layers
    for layer in layers:
        model.add(Dense(layer, activation='relu'))
        # model.add(Dense(layer, activation = 'elu'))
        model.add(Dropout(dropouts[1]))

    # Output layer
    # model.add(Dense(1, activation = 'relu'))#No really necessary since if x > 0 return x else return 0
    model.add(Dense(1))  # Not a single negative price data in our dataset

    # RMSprop with default values
    '''
    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=0.001, rho=0.9, momentum=0.0, epsilon=1e-07, centered=False,
        name='RMSprop'
        )
    '''

    # Adam Optimizer
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
        name='Adam'
    )

    # Adamax
    '''
    optimizer = tf.keras.optimizers.Adamax(
        learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name='Adamax'
        )
    '''

    model.compile(loss='mse',  # Recommended for gaussian data is mse and gaussian like with outliers is mae
                  optimizer=optimizer,
                  metrics=['mae', 'mse', 'mape', tf.keras.metrics.RootMeanSquaredError()])

    model.summary()

    return model


# Trains individual model and returns it
def training(model, normed_train_data, train_labels, normed_test_data, test_labels, name):
    tensorboard = TensorBoard(log_dir='ann_logs/' + name)
    EPOCHS = 15000  # Max epochs
    # early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)
    history = model.fit(
        normed_train_data,
        train_labels,
        epochs=EPOCHS,
        # validation_split = 0.2,
        validation_data=(normed_test_data, test_labels),
        verbose=0,
        use_multiprocessing=True,
        callbacks=[early_stop, tfdocs.modeling.EpochDots(), tensorboard])
    return history, model


# Displays training stats
def trainingStat(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist.tail()

    plotter = tfdocs.plots.HistoryPlotter(smoothing_std=0)
    plotter.plot({'Basic': history}, metric="mae")
    plt.ylabel('MAE [Price_Total]')
    plt.show()

    plotter = tfdocs.plots.HistoryPlotter(smoothing_std=0)
    plotter.plot({'Basic': history}, metric="mape")
    plt.ylabel('MPAE [Price_Total]')
    plt.show()

    plotter.plot({'Basic': history}, metric="mse")
    plt.ylabel('MSE [Price_Total^2]')
    plt.show()

    try:
        plotter.plot({'Basic': history}, metric=tf.keras.metrics.RootMeanSquaredError())
        plt.ylabel('RMSE [Price_Total]')
        plt.show()
    except:
        pass


# Build and train models -> This is actually deep learning
def buildAndTrain(normed_train_data, train_labels, normed_test_data, test_labels):
    layers = [448, 448]  # Hidden layers
    dropouts = [0.0, 0.1]  # Input, hidden
    name = '224-448-448-1-dropouts(0.0,0.1)-relu-adam-unnormalized'

    print('===Hidden Layers===')
    print(layers)

    # Building the model
    model = build_model(normed_train_data, layers, dropouts)

    # Test the model
    print('Test the model')
    example_batch = normed_train_data[:10]
    print(example_batch)
    example_result = model.predict(example_batch)
    print(example_result)

    # Train the model
    history, model = training(model, normed_train_data, train_labels, normed_test_data, test_labels, name)
    model.save("ann_models/" + name + ".model")

    # Training stat
    trainingStat(history)

    # Evaluate Model
    finalEvaluation(normed_test_data, test_labels, "ann_models/" + name + ".model")


# Evaluate individual model using the seperate test data and test label
def finalEvaluation(normed_test_data, test_labels, model):
    # Load the model
    model = tf.keras.models.load_model(model)

    # Evaluate the final model
    loss, mae, mse, mape, rmse = model.evaluate(normed_test_data, test_labels, verbose=0)
    print('===FINAL EVALUATION===')
    print("Testing set Mean Abs Error: {:5.2f} Price_Total".format(mae))
    print("Testing set Mean Abs Percentage Error: {}% Price_Total".format(mape))
    print("Testing set Root Mean Squared Error: {} Price_Total".format(rmse))
    print("Testing set Mean Squared Error: {} Price_Total".format(mse))

    # Make predictions
    # print('===Test Predictions===')
    test_predictions = model.predict(normed_test_data)
    # print(test_predictions)
    test_predictions = test_predictions.flatten()  # Convert to 1D array
    # print(test_predictions)

    # Scatter plot
    a = plt.axes(aspect='equal')
    plt.scatter(test_labels, test_predictions)
    plt.xlabel('True Values [Price_Total]')
    plt.ylabel('Predictions [Price_Total]')
    plt.show()

    # Prediction error histogram
    error = test_predictions - test_labels
    plt.hist(error, bins=100)
    plt.xlabel("Prediction Error [Price_Total]")
    _ = plt.ylabel("Count")
    plt.show()

    # R Square
    def r2(x, y):
        return stats.pearsonr(x, y)[0] ** 2

    sns.jointplot(test_labels, test_predictions, kind="reg", stat_func=r2)
    plt.show()


def main():
    # Import Data
    train_dataset, test_dataset = importData()

    # Get labels
    train_labels, test_labels = prepareLabels(train_dataset, test_dataset)

    # Get stats
    train_stats = inspectDataSet(train_dataset)

    # Normalize Data
    '''
    train_dataset = normalize(train_dataset, train_stats)
    test_dataset = normalize(test_dataset, train_stats)
    print('Normalized Data')
    print(normed_train_data.tail())
    '''

    # Build and train model
    buildAndTrain(train_dataset, train_labels, test_dataset, test_labels)

    # Final Evaluation
    # finalEvaluation(normed_test_data, test_labels, "Input-512-256-1-Drop(0.2,0.2).model")


if __name__ == "__main__":
    main()