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
import numpy as np



flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate of for adam [0.0002]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("sample_size", 100, "The size of samples [64]")

flags.DEFINE_string("dataset", "data/h36m/", "Dataset directory.")
flags.DEFINE_string("logdir", "logs", "Directory to save log")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("ckpt_file", "model.ckpt", "Checkpoint file")

flags.DEFINE_string("mode", "gan", "Mode to use")

flags.DEFINE_boolean("check_input", False, "Whether to check the input of D")
flags.DEFINE_boolean("load", False, "Load existing model")
flags.DEFINE_boolean("val_save", False, "Save images and videos when validating")
flags.DEFINE_boolean("debug", False, "Whether to turn on debug mode")
flags.DEFINE_boolean("verbose", False, "Whether to store verbose logs")

FLAGS = flags.FLAGS

if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
# if not os.path.exists(FLAGS.sample_dir):
#     os.makedirs(FLAGS.sample_dir)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True



ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'length': 'label length',
    'label': 'A list of labels, one per each object.',
}


# =========================================================================== #
# Main
# =========================================================================== #
def main(_):

    batch_size = FLAGS.batch_size
    # num_readers = 4
    num_epochs = FLAGS.epoch
    checkpoint_dir = FLAGS.checkpoint_dir

    with tf.Graph().as_default():

        # deploy_config = model_deploy.DeploymentConfig()
        # Create global_step.
        global_step = tf.Variable(0, name='global_step', trainable=False)

        tr_file_name = os.path.join("/mnt/sdb/mark/mjsyth", "mjtrain_train.tfrecords")
        te_file_name = os.path.join("/mnt/sdb/mark/mjsyth", "mjtrain_test.tfrecords")

        sh_images, sh_labels, sh_length= read_utils.inputs( filename=[tr_file_name], batch_size=batch_size, num_epochs=num_epochs)
        val_images, val_labels, val_length= read_utils.inputs( filename=[te_file_name], batch_size=batch_size, num_epochs=1000)


        # Build Model
        crnn = model.CRNNNet()
        with tf.variable_scope('crnn'):
            logits, seq_len = crnn.net(sh_images, is_training=True)
            tf.get_variable_scope().reuse_variables()
            val_logits, val_seq_len = crnn.net(val_images, is_training=False)

        loss = crnn.losses(sh_labels, logits, seq_len)
        tf.summary.scalar("train/loss", loss)

        val_loss = crnn.losses(val_labels, val_logits, val_seq_len)
        # TODO: BK-tree NN search
        decoded, log_prob = tf.nn.ctc_beam_search_decoder(tf.transpose(val_logits, perm=[1, 0, 2]), val_seq_len, merge_repeated=False)

        acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), val_labels))

        val_loss_sum = tf.placeholder(tf.float32, name='val_loss_sum')
        val_acc_sum = tf.placeholder(tf.float32, name='val_acc_sum')

        tf.summary.scalar("test/val_loss", val_loss_sum)
        tf.summary.scalar("test/edit_distance", val_acc_sum)



        starter_learning_rate = FLAGS.learning_rate
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                   100000, 0.96, staircase=True)
        tf.summary.scalar("train/learning_rate",learning_rate)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):        
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=loss, global_step=global_step)

        # Start Training
        with tf.Session(config=config) as sess:
            save = tf.train.Saver(max_to_keep=10)

            if not FLAGS.load:
                init_op = tf.group(tf.global_variables_initializer(),
                                   tf.local_variables_initializer())

                sess.run(init_op)            # Start input enqueue threads.
            else:

                # ckpt_file = 'model.ckpt-' + FLAGS.ckpt_step
                ckpt_path = os.path.join(FLAGS.checkpoint_dir, FLAGS.ckpt_file)
                save.restore(sess, ckpt_path)
                sess.run(tf.local_variables_initializer())


            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            merged = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, scope='train/*'))
            val_merged = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, scope='test/*'))

            file_writer = tf.summary.FileWriter(FLAGS.logdir, sess.graph)

            try:

                while not coord.should_stop():
                    start_time = time.time()

                    _, merged_t, tr_loss, lr, step, db_lables, db_images, db_logits = sess.run([optimizer, merged, loss, learning_rate, global_step, sh_labels, sh_images, logits])

                    duration = time.time() - start_time

                    print("loss", tr_loss, "time", duration)
                    file_writer.add_summary(merged_t, step)

                    # Print an overview fairly often.
                    if step % 5000 == 0:
                        #######################################################

                        val_loss_s, val_acc_s = 0, 0
                        for ite in range(FLAGS.sample_size):
                            te_loss, te_acc = sess.run([val_loss, acc])
                            val_loss_s += te_loss
                            val_acc_s += te_acc
                        val_loss_s /= FLAGS.sample_size
                        val_acc_s /= FLAGS.sample_size

                        print('Step %d: loss %.3f acc %.3f (%.3f sec)' % (step, val_loss_s, val_acc_s, duration))

                        # Add summary
                        val_sum = sess.run(val_merged, feed_dict={val_loss_sum: val_loss_s, val_acc_sum: val_acc_s})
                        file_writer.add_summary(val_sum, step)

                        save.save(sess, os.path.join(FLAGS.checkpoint_dir, 'model.ckpt'), global_step=step)
                    
            except tf.errors.OutOfRangeError:
                print('Done training for %d epochs, %d steps.' % (num_epochs, step))
            finally:
                # When done, ask the threads to stop.
                coord.request_stop()

                # Wait for threads to finish.
            coord.join(threads)

if __name__ == '__main__':
    tf.app.run()

