import tensorflow as tf

def get_data_generators(data_root, image_shape=(64, 64), batch_size=32):
    """
    Creates training and validation data generators from a directory.
    """
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        validation_split=0.3, 
        rescale=1/255
    )

    train_gen = image_generator.flow_from_directory(
        str(data_root),
        subset='training',
        target_size=image_shape,
        shuffle=True,
        batch_size=batch_size,
        class_mode='categorical'
    )

    val_gen = image_generator.flow_from_directory(
        str(data_root),
        subset='validation',
        target_size=image_shape,
        shuffle=True,
        batch_size=batch_size,
        class_mode='categorical'
    )

    return train_gen, val_gen