from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping



img_width, img_height = 128, 128    # input image has size (3, 128, 128)
train_data_dir = "data/train"    # Location of training data
validation_data_dir = "data/val"    # Location of validation data
nb_train_samples = 244       # Total Number of Training samples
nb_validation_samples = 153       # Total Number of Validations samples
batch_size = 16
epochs = 50


model = applications.MobileNet(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))


model.summary()

#freez
# Freeze the first five layers which you don't want to train. 
for layer in model.layers:  
    layer.trainable = False


#Adding custom Layers 
x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(512, activation="relu")(x)
predictions = Dense(2, activation="softmax")(x)



# creating the final model 
model_final = Model(inputs = model.input, outputs = predictions)

model_final.summary()

model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])



# Initiate the train and test generators with data Augumentation 
train_datagen = ImageDataGenerator(rescale = 1./255,
                    horizontal_flip = True,
                    fill_mode = "nearest",
                    zoom_range = 0.3,
                    width_shift_range = 0.3,
                    height_shift_range=0.3,
                    rotation_range=30)


# This an augmentator for test dataset
test_datagen = ImageDataGenerator(rescale = 1./255,
                    horizontal_flip = True,
                    fill_mode = "nearest",
                    zoom_range = 0.3,
                    width_shift_range = 0.3,
                    height_shift_range=0.3,
                    rotation_range=30)




train_generator = train_datagen.flow_from_directory(train_data_dir,
                        target_size = (img_height, img_width),
                        batch_size = batch_size, 
                        class_mode = "categorical")

validation_generator = test_datagen.flow_from_directory(validation_data_dir,
                        target_size = (img_height, img_width),
                        class_mode = "categorical")

# Save the model according to the conditions  
checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')



# Train the model 
model_final.fit_generator(train_generator,
        samples_per_epoch = nb_train_samples,
        epochs = epochs,
        validation_data = validation_generator,
        nb_val_samples = nb_validation_samples,
        callbacks = [checkpoint, early])






