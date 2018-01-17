from __future__ import print_function
import tensorflow as tf
# from deployment import model_deploy
# from net import model_save as model
import os
slim = tf.contrib.slim
import time
from net import model
from dataset import read_utils
from tensorflow.python import debug as tf_debug
from PIL import Image
import numpy as np
from dataset.utils import char_list, HEIGHT, WIDTH



flags = tf.app.flags

flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("sample_size", 100, "The size of samples [64]")

flags.DEFINE_string("data_dir", "data/", "Evaluation dataset directory.")
flags.DEFINE_string("dataset", "coco", "Evaluation dataset.")
flags.DEFINE_string("gt_file", "data/file.txt", "Ground truth file.")
flags.DEFINE_string("logdir", "logs", "Directory to save log")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("ckpt_file", "model.ckpt", "Checkpoint file")
flags.DEFINE_string("mode", "new", "Decode mode")
flags.DEFINE_boolean("load", False, "Load existing model")
flags.DEFINE_boolean("case_insensitive", False, "Lower case")
flags.DEFINE_boolean("val_save", False, "Save images and videos when validating")
flags.DEFINE_boolean("debug", False, "Whether to turn on debug mode")
flags.DEFINE_boolean("verbose", False, "Whether to store verbose logs")

FLAGS = flags.FLAGS

# if not os.path.exists(FLAGS.checkpoint_dir):
#     os.makedirs(FLAGS.checkpoint_dir)
# if not os.path.exists(FLAGS.sample_dir):
#     os.makedirs(FLAGS.sample_dir)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True



ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'length': 'label length',
    'label': 'A list of labels, one per each object.',
}


def code2str(code_list):
    output = ''
    for c in code_list:
        if FLAGS.mode == 'old':
            output += char_list[c]
        elif FLAGS.mode == 'new':
            output += chr(c+32)

    return output

def str2code(string):
    output = []
    for c in string:
        if FLAGS.mode == 'old':
            output.append(char_list.find(c))
        elif FLAGS.mode == 'new':
            output.append(ord(c)-32)

    return output

# =========================================================================== #
# Main
# =========================================================================== #
def main(_):


    checkpoint_dir = FLAGS.checkpoint_dir

    with tf.Graph().as_default():

        # deploy_config = model_deploy.DeploymentConfig()
        # Create global_step.
        
        val_images = tf.placeholder(tf.float32, shape=[1, HEIGHT, WIDTH, 3], name='input_img')
        val_labels = tf.sparse_placeholder(tf.int32, name='input_labels')
        val_width = tf.placeholder(tf.int32, shape=[1], name='input_width')
        #indices = tf.placeholder(tf.int32, [None, 2])
        #values = tf.placeholder(tf.int32, [None])
        #shape = tf.placeholder(tf.int32, [2])

        #val_labels = tf.SparseTensor(indices, values, shape)


        # Build Model
        crnn = model.CRNNNet()
        with tf.variable_scope('crnn'):
            val_logits, val_seq_len = crnn.net(val_images, val_width, is_training=False, kp=1.0)


        val_loss = crnn.losses(val_labels, val_logits, val_seq_len)
        # TODO: BK-tree NN search
        decoded, log_prob = tf.nn.ctc_beam_search_decoder(tf.transpose(val_logits, perm=[1, 0, 2]), val_seq_len, merge_repeated=False)

        acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), val_labels, normalize=False))
        acc_norm = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), val_labels))



        # Start Training
        with tf.Session(config=config) as sess:
            save = tf.train.Saver(max_to_keep=50)

            assert FLAGS.load
            if not FLAGS.load:
                init_op = tf.group(tf.global_variables_initializer(),
                                   tf.local_variables_initializer())

                sess.run(init_op)            # Start input enqueue threads.
            else:

                # ckpt_file = 'model.ckpt-' + FLAGS.ckpt_step
                ckpt_path = os.path.join(FLAGS.checkpoint_dir, FLAGS.ckpt_file)
                save.restore(sess, ckpt_path)
                # sess.run(tf.local_variables_initializer())


            with open(FLAGS.gt_file, 'r') as f:

                val_loss_s, val_acc_s, val_acc_norm_s = 0, 0, 0
                counter = 0
                hit = 0
                for line in f:
                    # print(line)
                    if FLAGS.dataset == 'ch4':
                        line = line.replace('\xef\xbb\xbf','')
                        line = line.replace('\r\n','')
                        # parse each line
                        img_file = line.split(', ')[0]
                        img_label = line.split(', ')[1][1:-1]
                        print(img_file, img_label)
                    if FLAGS.dataset == 'coco':
                        line = line.replace('\r\n','')
                        line = line.replace('\n','')
                        img_file = line.split(',')[0]+'.jpg'
                        if len(line) < 10:
                            continue
                        start = line.find(',')
                        img_label = line[start+1:]
                        print(img_file, img_label)
                        # labels.append(label)
                        # print(img_file, label)
                        # imgLists.append(os.path.join(data_prefix, img_file))        
                    if FLAGS.dataset == 'IC13':
                        line = line.replace('\xef\xbb\xbf','')
                        line = line.replace('\r\n','')
                        # parse each line
                        img_file = line.split(', ')[0]
                        img_label = line.split(', ')[1][1:-1]
                        # print(img_file, img_label)


                    img = Image.open(os.path.join(FLAGS.data_dir, img_file))
                    # w, h = img.size
                    # # print(w, h)
                    # ratio = 32 / float(h)
                    # data = data.resize([int(ratio*w), 32])
                    # # print(data.size)
                    # container = Image.new('RGB', (32, 100))
                    # container.paste(img)
                    # img = container
                    w, h = img.size
                    if w < h:
                        img = img.rotate(-90, expand=True)
                    w, h = img.size
                    # print(w, h)
                    ratio = HEIGHT / float(h)
                    if int(ratio*w) > WIDTH:
                        img = img.resize([WIDTH, HEIGHT])
                        actual_width = [WIDTH]
                    else:
                        img = img.resize([int(ratio*w), HEIGHT])
                        actual_width = [int(ratio*w)]
                    # print(data.size)
                    container = Image.new('RGB', (WIDTH, HEIGHT))
                    container.paste(img)
                    img = container

                    img = np.asarray(img, np.float32)
                    img = np.expand_dims(img, axis=0)

                    img = img * (1. / 255) - 0.5


                    str_label = img_label
                    if FLAGS.case_insensitive:
                        str_label = str_label.lower()
                    img_label = str2code(img_label)
                    if -1 in img_label:
                        continue
                    print(img_file, str_label)

                    indices = [(0, i) for i in range(len(img_label))]
                    values = [c for c in img_label]
                    shape = [1, len(img_label)]




                    output_label, te_acc, te_acc_norm = sess.run([decoded, acc, acc_norm], feed_dict={
                        val_images: img,
                        val_labels: (indices, values, shape),
                        val_width: actual_width
                        })
                    val_loss_s += 0
                    val_acc_s += te_acc
                    val_acc_norm_s += te_acc_norm
                    counter += 1
                    output_str = code2str(output_label[0].values)

                    print(img_file, output_str)
                    print(te_acc)
                     
                    if FLAGS.case_insensitive:
                        output_str = output_str.lower()
                    if output_str == str_label:
                        hit += 1
                        print(hit)

                val_loss_s /= counter
                val_acc_s /= counter
                val_acc_norm_s /= counter
                pred_acc = hit / float(counter)

                print(hit, counter)
                        
                print('loss %.3f edit dist %.3f %.3f acc %.3f' % (val_loss_s, val_acc_s, val_acc_norm_s, pred_acc))



if __name__ == '__main__':
    tf.app.run()

