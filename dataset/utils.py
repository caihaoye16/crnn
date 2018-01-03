import tensorflow as tf
import os
from PIL import Image
import numpy as np

char_list = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz "
WIDTH = 300
HEIGHT = 32

def int64_feature(value):
    """Wrapper for inserting int64 features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    value_tmp = []
    is_int = True
    for val in value:
        if not isinstance(val,int):
            is_int = False
            value_tmp.append(int(float(val)))
    if is_int is False:
        value = value_tmp
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value):
    """Wrapper for inserting float features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    value_tmp = []
    is_float = True
    for val in value:
        if not isinstance(val, int):
            is_float = False
            value_tmp.append(float(val))
    if is_float is False:
        value = value_tmp
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto.
    """
    if not isinstance(value,bytes):
        if not isinstance(value,list):
            value = value.encode('utf-8')
        else:
            value = [val.encode('utf-8') for val in value]
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _get_output_filename(output_dir, name):
    return '%s/%s.tfrecords' % (output_dir, name)


def load_label_from_imglist(imgLists):
    # get label from img name
    labels = []
    for img_dir in imgLists:
        img_basename = os.path.basename(img_dir)
        (img_name, postfix) = os.path.splitext(img_basename)
        str = img_name.split("_")
        labels.append(str[1])
    return labels


def load_label_from_img_dir(img_dir):
    # get label from img name
    img_basename = os.path.basename(img_dir)
    (img_name, postfix) = os.path.splitext(img_basename)
    str = img_name.split("_")
    if len(str)>1:
        return str[1]
    else:
        return str[0]


def load_image(img_dir, width = WIDTH, height = HEIGHT):
    """
    :param img_dir:
    :return:img_data
     load image and resize it
    """
    data = Image.open(img_dir)
    w, h = data.size
    # print(w, h)
    ratio = height / float(h)
    if int(ratio*w) > width:
        data = data.resize([width, height])
    else:
        data = data.resize([int(ratio*w), height])
        # print(data.size)
        container = Image.new('RGB', (width, height))
        container.paste(data)
        data = container
    # print(data.size)
    data = data.tobytes()
    return data


def char_to_int(char):
    temp = ord(char) - 32
    if temp > 94 or temp < 0:
        temp = 0
    # if temp>=97 and temp<=122:
    #     temp = temp-97+10+26
    # else:
    #     if temp >= 65 and temp <= 90:
    #         temp -= 55
    #     else:
    #         if temp>=48 and temp<=57:
    #             temp -= 48
    #         else:
    #             temp = 62
    return temp


def int_to_char(number):
    return chr(number)



def encode_labels(labels):
    """
    :param labels:
    :return:
    """
    encord_labels = []
    lengths = []
    for label in labels:
        encord_label = [char_to_int(char) for char in label]
        encord_labels.append(encord_label)
        lengths.append(len(label))
    return encord_labels,lengths


def encode_label(label):
    """
    :param labels:
    :return:
    """
    encord_label = [char_to_int(char) for char in label]
    length = len(label)
    return encord_label,length


def sparse_tensor_to_str(spares_tensor):
    """
    :param spares_tensor:
    :return: a str
    """
    indices= spares_tensor[0][0]
    values = spares_tensor[0][1]
    dense_shape = spares_tensor[0][2]

    number_lists = np.ones(dense_shape,dtype=values.dtype)
    str_lists = []
    str=''
    for i,index in enumerate(indices):
        number_lists[index[0],index[1]] = values[i]
    for number_list in number_lists:
        str_lists.append([int_to_char(val) for val in number_list])
    for str_list in str_lists:
        str += ''.join(str_list)
    return str




