import json
import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

DATA_PATH = "data.json"
SAVED_MODEL_PATH = "../server/model.h5"
EPOCHS = 80
BATCH_SIZE = 2
PATIENCE = 5
LEARNING_RATE = 0.0001


def load_data(data_path):
    """
    Function to load dataset and extract x and y from data.
    :param data_path: Path of the dataset
    :return x and y: From data
    """
    # Open dataset
    with open(data_path, "r") as fp:
        data = json.load(fp)

    # Get x and y from data
    X = np.array(data["MFCCs"])
    y = np.array(data["labels"])

    print("Training sets loaded!")

    return X, y


def prepare_dataset(data_path, test_size=0.1, validation_size=0.1):
    """
    Function to create train, validation, test split and add an axis to nd array.
    :param data_path: Path of the dataset
    :param test_size: Size of the test set
    :param validation_size: Size of the validation set
    :return X_train, X_validation, X_test, y_train, y_validation and y_test: Train, validation and test splits
    """
    # Load dataset
    X, y = load_data(data_path)

    # Create train, validation, test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    # Add an axis to nd array
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test


def build_model(input_shape, learning_rate=0.0001, loss="sparse_categorical_crossentropy"):
    """
    Build neural network using keras.
    :param input_shape (tuple): Shape of array representing a sample train. E.g.: (44, 13, 1)
    :param loss (str): Loss function to use
    :param learning_rate (float):
    :return model: TensorFlow model
    """

    # Build network architecture using convolutional layers
    model = keras.Sequential()

    # 1st conv layer
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape,
                                     kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2,2), padding='same'))

    # 2nd conv layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu',
                                     kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2,2), padding='same'))

    # 3rd conv layer
    model.add(keras.layers.Conv2D(32, (2, 2), activation='relu',
                                     kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2,2), padding='same'))

    # Flatten output and feed into dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    # Softmax output layer
    model.add(keras.layers.Dense(3, activation='softmax'))

    optimiser = keras.optimizers.Adam(learning_rate=learning_rate)

    # Compile model
    model.compile(optimizer=optimiser,
                  loss=loss,
                  metrics=["accuracy"])

    # Print model parameters on console
    model.summary()

    return model

def main():
    """
    Main function to train network, evaluate network on test set and save model.
    """
    # generate train, validation and test sets
    X_train,  X_validation, X_test,y_train,y_validation, y_test = prepare_dataset(DATA_PATH)

    # create network
    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    model = build_model(input_shape, LEARNING_RATE)

    # train network
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_validation, y_validation))

    # evaluate network on test set
    test_error, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test error: {test_error}, test accuracy: {test_accuracy}")

    # save model
    model.save(SAVED_MODEL_PATH)

# Run the script
if __name__ == "__main__":
    main()