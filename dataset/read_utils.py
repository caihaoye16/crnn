import tensorflow as tf
from utils import HEIGHT, WIDTH

def read_and_decode(filenames, num_epochs, preprocess=False):  # read iris_contact.tfrecords
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs)
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)  # return file_name and file

    if not preprocess:    
        features = tf.parse_single_example(serialized_example,
                                           features={
                                               'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
                                               'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
                                               'label/value': tf.VarLenFeature(tf.int64),
                                               'label/length': tf.FixedLenFeature([1], tf.int64)
                                           })  # return image and label

        # Preprocessing Here

        img = tf.decode_raw(features['image/encoded'], tf.uint8)
        img = tf.reshape(img, [HEIGHT, WIDTH, 3])  
        # img = tf.image.rgb_to_grayscale(img)
        img = tf.cast(img, tf.float32) * (1. / 255) - 0.5  # throw img tensor
        label = features['label/value']  # throw label tensor
        label = tf.cast(label, tf.int32)
        length = features["label/length"]
    else:
        features = tf.parse_single_example(serialized_example,
                                           features={
                                               'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
                                               'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
                                               'label/value': tf.VarLenFeature(tf.int64),
                                               'label/length': tf.FixedLenFeature([1], tf.int64),
                                               'image/width': tf.FixedLenFeature([1], tf.int64),
                                               'image/height': tf.FixedLenFeature([1], tf.int64)
                                           })  # return image and label        
        img = tf.decode_raw(features['image/encoded'], tf.uint8)
        width = features['image/width']
        height = features['image/height']
        shape = tf.concat([height, width, [3]], 0)
        img = tf.reshape(img, shape)

        # Process to HEIGHT and WIDTH  
        ratio = HEIGHT / height

        img = tf.cond(tf.squeeze(ratio * width <= WIDTH),
                      lambda: tf.image.resize_image_with_crop_or_pad(tf.image.resize_images(img, tf.cast(tf.concat([[HEIGHT], ratio*width], 0), tf.int32)), HEIGHT, WIDTH),
                      lambda: tf.image.resize_images(img, tf.cast(tf.stack([HEIGHT, WIDTH], 0), tf.int32))
                      )


        img = tf.cast(img, tf.float32) * (1. / 255) - 0.5  # throw img tensor
        label = features['label/value']  # throw label tensor
        label = tf.cast(label, tf.int32)
        length = features["label/length"]       

    return img, label, length


def inputs(batch_size, num_epochs, filename, preprocess=False):
    if not num_epochs: 
        num_epochs = None
    with tf.name_scope('input'):
        # Even when reading in multiple threads, share the filename
        # queue.
        img, label, length = read_and_decode(filename, num_epochs, preprocess)

        # Shuffle the examples and collect them into batch_size batches.
        # (Internally uses a RandomShuffleQueue.)
        # We run this in two threads to avoid being a bottleneck.
        sh_images, sh_labels, sh_length = tf.train.shuffle_batch(
            [img, label, length], batch_size=batch_size, num_threads=4,
            capacity=5000,
            # Ensures a minimum amount of shuffling of examples.
            min_after_dequeue=1000)

        return sh_images, sh_labels, sh_length

def preprocess_for_train(image,label ,scope='crnn_preprocessing_train'):
    """Preprocesses the given image for training.
    """
    with tf.name_scope(scope, 'ssd_preprocessing_train', [image]):
        if image.get_shape().ndims != 3:
            raise ValueError('Input must be of size [height, width, C>0]')
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)# convert image as a tf.float32 tensor
            image_s = tf.expand_dims(image, 0)
            tf.summary.image("image",image_s)

        image = tf.image.rgb_to_grayscale(image)
        tf.summary.image("gray",image)
        return image, label
