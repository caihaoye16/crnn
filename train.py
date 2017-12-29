import os
import scipy.misc
import numpy as np

from model import DCGAN
from vae import VAE
from pose import *
from wgan import WGAN
from wgan_vae import WGANVAE
from sk2im import *
from test_sk2im import *
from sk2im_npy import *
from forward import *
# from model_lsgan import DCGAN

from utils import pp, visualize, to_json


import tensorflow as tf
from tensorflow.python import debug as tf_debug


def my_has_inf_or_nan(datum, tensor):




    _ = datum  # Datum metadata is unused in this predicte.
    if tensor is None:
        # Uninitialized tensor doesn't have bad numerical values.
        return False
    elif (np.issubdtype(tensor.dtype, np.float) or
          np.issubdtype(tensor.dtype, np.complex) or
          np.issubdtype(tensor.dtype, np.integer)):
        return np.any(np.isnan(tensor)) or np.any(np.isinf(tensor))
    else:
        return False




flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_float("momentum", 0.9, "Momentum for GD in completion")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("vec_length", 30, "The size of image to use")
flags.DEFINE_integer("sample_size", 64, "The sample size for testing")
flags.DEFINE_integer("pool_size", 64, "The sample pool size for choosing initial point for completion")
flags.DEFINE_integer("seq_length", 50, "The length of video")
flags.DEFINE_integer("global_counter", 0, "Global counter to save model")
flags.DEFINE_integer("classes", 5, "Number of classes")
flags.DEFINE_integer("n_iter", 500, "Number of iterations for completion")
flags.DEFINE_string("dataset", "data/h36m/", "Dataset directory.")
flags.DEFINE_string("data_video_dir", "/home/mark/Data/Human3.6m", "Video data directory.")
flags.DEFINE_string("cameras_path","data/h36m/cameras.h5","Directory to load camera parameters")
flags.DEFINE_string("tmp_dir","tmp/","Directory to load camera parameters")
flags.DEFINE_string("video_dir","video/","Directory to save videos")
flags.DEFINE_string("img_dir","img/","Where to save images")
flags.DEFINE_string("blend_dir","video_comp","blend")
flags.DEFINE_string("logdir", "logs", "Directory to save log")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sk2im_ckpt_dir", "checkpoint", "Directory name to load the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_string("pose_dir", "pose", "Directory to save generated pose")
flags.DEFINE_string("actions", "all", "Actions to train on")
flags.DEFINE_string("loader", "saver", "Loader to use")
flags.DEFINE_string("mode", "gan", "Mode to use")
flags.DEFINE_boolean("complete", False, "Whether to enter completion mode")
flags.DEFINE_boolean("check_input", False, "Whether to check the input of D")
flags.DEFINE_boolean("load", False, "Load existing model")
flags.DEFINE_boolean("val_save", False, "Save images and videos when validating")
flags.DEFINE_boolean("debug", False, "Whether to turn on debug mode")
flags.DEFINE_boolean("verbose", False, "Whether to store verbose logs")
flags.DEFINE_string("check_grad", None, "Whether to check grad")
flags.DEFINE_string("algo", "sgd", "algo to complete")
flags.DEFINE_boolean("viz", False, "visualization in completion")
flags.DEFINE_boolean("prediction", False, "whether to do prediction")
FLAGS = flags.FLAGS

if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
with tf.Session(config=config) as sess:
    if FLAGS.debug:
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess.add_tensor_filter("has_inf_or_nan", my_has_inf_or_nan)

    if FLAGS.mode == 'gan':
        if not FLAGS.complete and not FLAGS.prediction:
            dcgan = DCGAN(sess, vec_length=FLAGS.vec_length, seq_length = FLAGS.seq_length, batch_size=FLAGS.batch_size, actions = FLAGS.actions, first_z_dim=12, shift_z_dim=64,
                        is_crop=False, checkpoint_dir=FLAGS.checkpoint_dir, loader=FLAGS.loader, classes=FLAGS.classes, pred=FLAGS.prediction)
        else:
            dcgan = DCGAN(sess, vec_length=FLAGS.vec_length, seq_length = FLAGS.seq_length, batch_size=FLAGS.batch_size, actions = FLAGS.actions, first_z_dim=8, shift_z_dim=8,
                        is_crop=False, checkpoint_dir=FLAGS.checkpoint_dir, loader=FLAGS.loader, classes=FLAGS.classes, pred=FLAGS.prediction)

        if not FLAGS.complete and not FLAGS.prediction:
            dcgan.train(FLAGS)
        else:
            # dcgan.complete_single(FLAGS)
            dcgan.complete(FLAGS)

    if FLAGS.mode == 'forward':
        model = Sampler(sess, FLAGS.data_video_dir, vec_length=FLAGS.vec_length, seq_length = FLAGS.seq_length, batch_size=FLAGS.batch_size, actions = FLAGS.actions,
                        is_crop=False, checkpoint_dir=FLAGS.checkpoint_dir, loader=FLAGS.loader, classes=FLAGS.classes)

        model.forward(FLAGS)

    if FLAGS.mode == 'sk2im_npy':
        model = sk2im_npy(sess, FLAGS.data_video_dir, vec_length=FLAGS.vec_length, seq_length = FLAGS.seq_length, batch_size=FLAGS.batch_size, actions = FLAGS.actions,
                        is_crop=False, checkpoint_dir=FLAGS.checkpoint_dir, loader=FLAGS.loader, classes=FLAGS.classes)

        model.forward(FLAGS)
    if FLAGS.mode == 'wgan':
        wgan = WGAN(sess, vec_length=FLAGS.vec_length, seq_length = FLAGS.seq_length, batch_size=FLAGS.batch_size, actions = FLAGS.actions,
                      is_crop=False, checkpoint_dir=FLAGS.checkpoint_dir, loader=FLAGS.loader)

        wgan.train(FLAGS)

    if FLAGS.mode == 'wganvae':
        wgan = WGANVAE(sess, vec_length=FLAGS.vec_length, seq_length = FLAGS.seq_length, batch_size=FLAGS.batch_size, actions = FLAGS.actions,
                      is_crop=False, checkpoint_dir=FLAGS.checkpoint_dir, loader=FLAGS.loader)

        wgan.train(FLAGS)

    if FLAGS.mode == 'vae':
        vae = VAE(sess, vec_length=FLAGS.vec_length, seq_length = FLAGS.seq_length, batch_size=FLAGS.batch_size, actions = FLAGS.actions,
                      is_crop=False, checkpoint_dir=FLAGS.checkpoint_dir, loader=FLAGS.loader, data_video_dir=FLAGS.data_video_dir)

        vae.train(FLAGS)
    if FLAGS.mode == 'pose_gan':
        pose = poseGAN(sess, vec_length=FLAGS.vec_length, seq_length = FLAGS.seq_length, batch_size=FLAGS.batch_size, actions = FLAGS.actions,
                      is_crop=False, checkpoint_dir=FLAGS.checkpoint_dir, loader=FLAGS.loader)
        pose.train(FLAGS)
    if FLAGS.mode == 'pose_wgan':
        pose = poseWGAN(sess, vec_length=FLAGS.vec_length, seq_length = FLAGS.seq_length, batch_size=FLAGS.batch_size, actions = FLAGS.actions,
                      is_crop=False, checkpoint_dir=FLAGS.checkpoint_dir, loader=FLAGS.loader, classes=FLAGS.classes)
        pose.train(FLAGS)
    if FLAGS.mode == 'sk2im':
        model = sk2im(sess, FLAGS.data_video_dir, vec_length=FLAGS.vec_length, seq_length = FLAGS.seq_length, batch_size=FLAGS.batch_size, actions = FLAGS.actions,
                      is_crop=False, checkpoint_dir=FLAGS.checkpoint_dir, loader=FLAGS.loader)
        model.train(FLAGS)
    if FLAGS.mode == 'test_sk2im':
        model = test_sk2im(sess, FLAGS.data_video_dir, vec_length=FLAGS.vec_length, seq_length = FLAGS.seq_length, batch_size=FLAGS.batch_size, actions = FLAGS.actions,
                      is_crop=False, checkpoint_dir=FLAGS.checkpoint_dir, loader=FLAGS.loader)
        model.test(FLAGS)