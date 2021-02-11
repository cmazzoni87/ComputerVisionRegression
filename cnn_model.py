import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import *
import datetime as dt
import numpy as np
import os

PATH = os.path.dirname(__file__)
IMAGES_PATH = os.path.join(PATH , 'GramianAnagularFields/TRAIN')
REPO = os.path.join(PATH , 'Models')
TESTING = os.path.join(PATH , 'GramianAnagularFields/TEST')
PATH_DOC = os.path.join(os.path.dirname(__file__), 'Documents')
PATH_OUT = os.path.join(os.path.dirname(__file__), 'Output')

EPOCHS = 5
SPLIT = 0.30
LR = 0.001

cnn_netwoks = 3
model = []
for j in range(cnn_netwoks):
    model.append(
        tf.keras.models.Sequential([
            #  First Convolution
            Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(300, 300, 3)),
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
            #Output layer
            Dense(1, activation='sigmoid')]
            ))
    # Compile each model
    model[j].compile(optimizer=Adam(lr=LR), loss="binary_crossentropy", metrics=["acc"])

# All images will be rescaled by 1./255
train_validate_datagen = ImageDataGenerator(rescale=1/255, validation_split=SPLIT) # set validation split
for j in range(cnn_netwoks):
    import glob
    import os

    files = glob.glob("*cycle*.log")
    files.sort(key=os.path.getmtime)
    print("\n".join(files))


train_generator = train_validate_datagen.flow_from_directory(
    IMAGES_PATH,
    target_size=(300, 300),
    batch_size= 20,
    class_mode='binary',
    subset='training') # set as training data

validation_generator = train_validate_datagen.flow_from_directory(
    IMAGES_PATH, # same directory as training data
    target_size=(300, 300),
    batch_size=20,
    class_mode='binary',
    subset='validation') # set as validation data

test_generator = train_validate_datagen.flow_from_directory(
    TESTING, # same directory as training data
    target_size=(300, 300),
    class_mode='binary') # set as validation data

steps_per_epoch = train_generator.n // train_generator.batch_size
validation_steps = validation_generator.n // validation_generator.batch_size
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=0, factor=0.5, min_lr=0.00001)

for j in range(cnn_netwoks):
    print(f'Individual Net : {j+1}')
    history = model[j].fit_generator(train_generator,
                                epochs = EPOCHS,
                                steps_per_epoch = steps_per_epoch,
                                validation_data = validation_generator,
                                callbacks=[learning_rate_reduction],
                                verbose=0)
    print("CNN Model {0:d}: "
          "Epochs={1:d}, "
          "Training Accuracy={2:.5f}, "
          "Validation Accuracy={3:.5f}".format(j + 1,
                                               EPOCHS,
                                               max(history.history['acc']),
                                               max(history.history['val_acc'])))
results = np.zeros((60, 4))
for j in range(cnn_netwoks):
    results = results + model[j].predict(test_generator)
    scores = model[j].evaluate_generator(test_generator, steps=5)
    print("%s: %.2f%%" %(model[j].metrics_names[1], scores[1]*100))

results = np.argmax(results, axis=1)
print(results)


# #  Stop training after one non improving Epoch
# callback  = tf.keras.callbacks.EarlyStopping(
#     monitor='val_loss', min_delta=0, patience=50, verbose=0, mode='auto',
#     baseline=None, restore_best_weights=False)
#
# history = model.fit_generator(
#       train_generator,
#       steps_per_epoch=steps_per_epoch,
#       epochs=EPOCHS,
#       verbose=1,
#       validation_data=validation_generator,
#       validation_steps=validation_steps
#       ,callbacks=[callback]
#         )

# timestamp = dt.datetime.now().strftime("%Y%m%d%H%M%S")
# stringlist = []
# model.summary(print_fn=lambda x: stringlist.append(x))
# summary = "\n".join(stringlist)
# logging = ['{0}: {1}'.format(key, val[-1]) for key, val in history.history.items()]
# log = 'Results:\n' + '\n'.join(logging)
#
# scores = model.evaluate_generator(
#     test_generator,
#     steps = 5)
# print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))
#
# model.save_weights(os.path.join(REPO, 'computer_vision_regression_weights_{}.h5'.format(timestamp)))
# model.save(os.path.join(REPO, 'computer_vision_regression_model_{}.h5'.format(timestamp)))
# f = open(os.path.join(REPO, 'computer_vision_regression_summary_{}.txt'.format(timestamp)), 'w')
# f.write("EPOCHS: {0}\nSteps per epoch: {1}\nValidation steps: {2}\nVal Split:{3}\nLearning RT:{5}\n\n\n{4}\n\n"
#         "=========TRAINING LOG========\n{6}".format(EPOCHS, steps_per_epoch, validation_steps, SPLIT, summary,LR, log))
# f.close()



