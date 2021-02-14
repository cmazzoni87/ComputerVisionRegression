import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import *
from data_preprocessing import ensemble_data
import datetime as dt
import os


#  Ensemble CNN network to train a CNN model on GAF images labeled Long and Short
PATH = os.path.dirname(__file__)
IMAGES_PATH = os.path.join(PATH, 'GramianAngularFields/TRAIN')
REPO = os.path.join(PATH, 'Models')
PATH_DOC = os.path.join(os.path.dirname(__file__), 'Documents')
PATH_OUT = os.path.join(os.path.dirname(__file__), 'Output')
EPOCHS = 5
SPLIT = 0.30
LR = 0.001
TIMESTAMP = dt.datetime.now().strftime("%Y%m%d%H%M%S")

cnn_networks = 3
model = []
for j in range(cnn_networks):
    model.append(
        tf.keras.models.Sequential([
            #  First Convolution
            Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(255, 255, 3)),
            BatchNormalization(),
            Conv2D(32, kernel_size=(3, 3), activation='relu'),
            BatchNormalization(),
            Conv2D(32, kernel_size=(3, 3), strides=2, padding='same', activation='relu'),
            BatchNormalization(),
            Dropout(0.4),
            # Second Convolution
            Conv2D(64, kernel_size=(3, 3), activation='relu'),
            BatchNormalization(),
            Conv2D(64, kernel_size=(3, 3), activation='relu'),
            BatchNormalization(),
            Conv2D(64, kernel_size=(3, 3), strides=2, padding='same', activation='relu'),
            BatchNormalization(),
            Dropout(0.4),
            # Third Convolution
            Conv2D(128, kernel_size=4, activation='relu'),
            BatchNormalization(),
            Flatten(),
            Dropout(0.4),
            # Output layer
            Dense(1, activation='sigmoid')]
        ))
    # Compile each model
    model[j].compile(optimizer=Adam(lr=LR), loss='binary_crossentropy', metrics=['acc'])

# All images will be rescaled by 1./255
train_validate_datagen = ImageDataGenerator(rescale=1/255, validation_split=SPLIT)  # set validation split
test_datagen = ImageDataGenerator(rescale=1/255)
data_chunks = ensemble_data(cnn_networks, IMAGES_PATH)
for j in range(cnn_networks):
    print('Net : {}'.format(j+1))
    df_train = data_chunks[j].iloc[:-60]
    df_test = data_chunks[j].iloc[-60:]
    train_generator = train_validate_datagen.flow_from_dataframe(
        dataframe=df_train,
        directory=IMAGES_PATH,
        target_size=(255, 255),
        x_col='Images',
        y_col='Labels',
        batch_size=32,
        class_mode='binary',
        subset='training')

    validation_generator = train_validate_datagen.flow_from_dataframe(
        dataframe=df_train,
        directory=IMAGES_PATH,
        target_size=(255, 255),
        x_col='Images',
        y_col='Labels',
        batch_size=32,
        class_mode='binary',
        subset='validation')

    test_generator = test_datagen.flow_from_dataframe(
        dataframe=df_test,
        x_col='Images',
        y_col='Labels',
        directory=IMAGES_PATH,
        target_size=(255, 255),
        class_mode='binary')

    steps_per_epoch = train_generator.n // train_generator.batch_size
    validation_steps = validation_generator.n // validation_generator.batch_size
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=0, factor=0.5, min_lr=0.00001)
    history = model[j].fit_generator(train_generator,
                                     epochs=EPOCHS,
                                     steps_per_epoch=steps_per_epoch,
                                     validation_data=validation_generator,
                                     callbacks=[learning_rate_reduction],
                                     verbose=0)
    print('CNN Model {0:d}: '
          'Epochs={1:d}, '
          'Training Accuracy={2:.5f}, '
          'Validation Accuracy={3:.5f}'.format(j + 1,
                                               EPOCHS,
                                               max(history.history['acc']),
                                               max(history.history['val_acc'])))

    scores = model[j].evaluate_generator(test_generator, steps=5)
    print("{0}s: {1:.2f}%".format(model[j].metrics_names[1], scores[1]*100))
    string_list = []
    model[j].summary(print_fn=lambda x: string_list.append(x))
    summary = "\n".join(string_list)
    logging = ['{0}: {1}'.format(key, val[-1]) for key, val in history.history.items()]
    log = 'Results:\n' + '\n'.join(logging)
    model[j].save(os.path.join(REPO, 'computer_vision_model_{0}_{1}_of_{2}.h5'.format(TIMESTAMP, j+1, cnn_networks)))
    f = open(os.path.join(REPO, 'computer_vision_summary_{0}_{1}_of_{2}.h5'.format(TIMESTAMP, j+1, cnn_networks)), 'w')
    f.write("EPOCHS: {0}\nSteps per epoch: {1}\nValidation steps: {2}\nVal Split:{3}\nLearning RT:{5}\n\n\n{4}"
            "\n\n=========TRAINING LOG========\n{6}".format(EPOCHS, steps_per_epoch, validation_steps,  SPLIT, summary,
                                                            LR, log))
    f.close()
