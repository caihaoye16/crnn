import tensorflow as tf
import vgg_preprocessing
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

        # random resize
        ratio = tf.random_uniform([1], maxval=0.9)
        width = tf.cast(tf.cast(width, tf.float32) * (1 - tf.pow(ratio, 3)), tf.int32)
        width = tf.cond(tf.squeeze(width) < 2,
                        lambda: tf.constant([2]),
                        lambda: width)
        height = tf.cast(height, tf.int32)
        img = tf.image.resize_images(img, tf.concat([height, width], 0))

        # Process to HEIGHT and WIDTH  
        ratio = tf.cast(HEIGHT, tf.float32) / tf.cast(height, tf.float32)
        actual_width = tf.cast(tf.cast(width, tf.float32) * ratio, tf.int32) 
        # img = tf.Print(img, [tf.shape(img), height, width, ratio, actual_width])

        img, img_width = tf.cond(tf.squeeze(actual_width <= WIDTH),
                      lambda: [tf.image.pad_to_bounding_box(tf.image.resize_images(img, tf.cast(tf.concat([[HEIGHT], actual_width], 0), tf.int32)), 0, 0, HEIGHT, WIDTH),
                               tf.squeeze(actual_width)],
                      lambda: [tf.image.resize_images(img, [HEIGHT, WIDTH]),
                               WIDTH]
                      )
        # img = tf.image.resize_image_with_crop_or_pad(tf.image.resize_images(img, [HEIGHT, WIDTH/2]), HEIGHT, WIDTH)

        # Vallina
        # img = tf.cast(img, tf.float32) * (1. / 255.) - 0.5  # throw img tensor

        # ResNet
        img = tf.cast(img, tf.float32) / 255.
        img = vgg_preprocessing.preprocess_image(
            image=img,
            output_height=HEIGHT,
            output_width=WIDTH)

        label = features['label/value']  # throw label tensor
        label = tf.cast(label, tf.int32)
        length = features["label/length"]       

    return img, label, length, img_width


def inputs(batch_size, num_epochs, filename, preprocess=False):
    if not num_epochs: 
        num_epochs = None
    with tf.name_scope('input'):
        # Even when reading in multiple threads, share the filename
        # queue.
        img, label, length, width = read_and_decode(filename, num_epochs, preprocess)

        # Shuffle the examples and collect them into batch_size batches.
        # (Internally uses a RandomShuffleQueue.)
        # We run this in two threads to avoid being a bottleneck.
        sh_images, sh_labels, sh_length, sh_width = tf.train.shuffle_batch(
            [img, label, length, width], batch_size=batch_size, num_threads=4,
            capacity=5000,
            # Ensures a minimum amount of shuffling of examples.
            min_after_dequeue=1000)

        return sh_images, sh_labels, sh_length, sh_width

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
