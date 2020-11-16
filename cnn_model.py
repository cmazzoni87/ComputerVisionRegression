import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np
from tensorflow.keras.preprocessing import image
import glob
import datetime as dt


callback  = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=1, verbose=0, mode='auto',
    baseline=None, restore_best_weights=False
)

PATH = 'C:\\\\Users\\cmazz\\PycharmProjects\\ComputerVisionRegression\\'
IMAGES_PATH = PATH + 'GramianAnagularFields\\'
EPOCHS = 10
SPLIT = 0.2
LR = 0.0005
steps_per_epoch = 8 # 2000 // train_generator.batch_size
validation_steps = 8 # 800 // validation_generator.batch_size

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 300x300 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fifth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    # # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')
    tf.keras.layers.Dense(1, activation='sigmoid')])
model.compile(optimizer=Adam(lr=LR), loss='binary_crossentropy', metrics=['acc'])

# All images will be rescaled by 1./255
train_validate_datagen = ImageDataGenerator(rescale=1/255, validation_split=SPLIT) # set validation split
train_generator = train_validate_datagen.flow_from_directory(
    IMAGES_PATH,
    target_size=(300, 300),
    batch_size=128,
    class_mode='binary',
    subset='training') # set as training data

validation_generator = train_validate_datagen.flow_from_directory(
    IMAGES_PATH, # same directory as training data
    target_size=(300, 300),
    batch_size=32,
    class_mode='binary',
    subset='validation') # set as validation data

history = model.fit_generator(
      train_generator,
      steps_per_epoch=steps_per_epoch,
      epochs=EPOCHS,
      verbose=1,
      validation_data=validation_generator,
      validation_steps=validation_steps
      # ,callbacks=[callback]
        )

timestamp = dt.datetime.now().strftime("%Y%m%d%H%M%S")
stringlist = []
model.summary(print_fn=lambda x: stringlist.append(x))
summary = "\n".join(stringlist)
logging = ['{0}: {1}'.format(key, val[-1]) for key, val in history.history.items()]
log = 'Results:\n' + '\n'.join(logging)

model.save_weights(PATH + 'Models\\computer_vision_regression_{}.h5'.format(timestamp))
f = open(PATH + 'Models\\computer_vision_regression_summary_{}.txt'.format(timestamp), 'w')
f.write("EPOCHS: {0}\nSteps per epoch: {1}\nValidation steps: {2}\nVal Split:{3}\nLearning RT:{5}\n\n\n{4}\n\n"
        "=========TRAINING LOG========\n{6}".format(EPOCHS, steps_per_epoch, validation_steps, SPLIT, summary,LR, log))
f.close()

uploaded = glob.glob(PATH + "Testing\\*.png")
for fn in uploaded:
    img = image.load_img(fn, target_size=(300, 300))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)
    if classes[0] > 0.5:
        print("{0} is BUY with {1}".format(fn, classes[0]))
    else:
        print("{0} is SELL with {1}".format(fn, classes[0]))
