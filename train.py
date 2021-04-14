"""
For training the network
Modified from
https://github.com/microsoft/Recursive-Cascaded-Networks/blob/master/train.py

"""


import argparse
import numpy as np
import os
import json
import h5py
import copy
import collections
import re
import datetime
import hashlib
import time
from timeit import default_timer
from math import *

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--base_network', type=str, default='LDR',
                    help='Specifies the base network')
parser.add_argument('-n', '--n_cascades', type=int, default=1,
                    help='Number of cascades')
parser.add_argument('-r', '--rep', type=int, default=1,
                    help='Number of times of shared-weight cascading')
parser.add_argument('-g', '--gpu', type=str, default='1',
                    help='Specifies gpu device(s)')
parser.add_argument('-c', '--checkpoint', type=str, default=None,
                    help='Specifies a previous checkpoint to start with')
parser.add_argument('-d', '--dataset', type=str, default="datasets/liver.json",
                    help='Specifies a data config')
parser.add_argument('--batch', type=int, default=4,
                    help='Number of image pairs per batch')
parser.add_argument('--round', type=int, default=20000,
                    help='Number of batches per epoch')
parser.add_argument('--epochs', type=float, default=5,
                    help='Number of epochs')
parser.add_argument('--output', type=str, default='model',
                    help='Number of epochs')
parser.add_argument('--fast_reconstruction', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--val_steps', type=int, default=1)
parser.add_argument('--net_args', type=str, default='')
parser.add_argument('--data_args', type=str, default='')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--clear_steps', action='store_true')
parser.add_argument('--finetune', type=str, default=None)
parser.add_argument('--name', type=str, default=None)
parser.add_argument('--logs', type=str, default='')
parser.add_argument('--image_size', type=int, default=128)
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--use_pretrained_flow', action='store_true')
parser.add_argument('--aldk', action='store_true')
parser.add_argument('--pretrained_flow_path', type=str, default='./Teacher_deformations')
parser.add_argument('--beta', type=float, default=0.1,
                    help='Beta hyper parameter in aldk module')
parser.add_argument('--gamma', type=float, default=0.3,
                    help='Gamma hyper parameter in aldk module')
parser.add_argument('--lamb', type=float, default=0.1,
                    help='Lambda hyper parameter in aldk module')

# Learning rate params
parser.add_argument('--lr_set1', action='store_true')

# Learning scheme changing or not
parser.add_argument('--aldk_learning_scheme', action='store_true')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import tensorflow as tf
import tflearn

import network
from data_util.data import Split

if args.dataset.find('liver') != -1:
    from data_util.liver import Dataset
else:
    from data_util.brain import Dataset


def main():
    repoRoot = os.path.dirname(os.path.realpath(__file__))

    if args.finetune is not None:
        args.clear_steps = True

    batchSize = args.batch
    iterationSize = args.round
    args.lr_set1 = True if args.base_network == "LDR" and args.aldk == False \
                        else False

    gpus = 0 if args.gpu == '-1' else len(args.gpu.split(','))

    Framework = network.FrameworkUnsupervised
    Framework.net_args['aldk'] = args.aldk
    Framework.net_args['base_network'] = args.base_network
    Framework.net_args['n_cascades'] = args.n_cascades
    Framework.net_args['dataset'] = args.dataset
    Framework.net_args['rep'] = args.rep
    Framework.net_args['beta'] = args.beta
    Framework.net_args['gamma'] = args.gamma
    Framework.net_args['lamb'] = args.lamb
    Framework.net_args.update(eval('dict({})'.format(args.net_args)))
    with open(os.path.join(args.dataset), 'r') as f:
        cfg = json.load(f)
        image_size = cfg.get('image_size', [128, 128, 128])
        image_type = cfg.get('image_type')
    framework = Framework(devices=gpus, image_size=args.image_size,
                          segmentation_class_value=cfg.get(
                              'segmentation_class_value', None),
                          fast_reconstruction=args.fast_reconstruction)
    # Dataset = eval('data_util.{}.Dataset'.format(image_type))
    print('Graph built.')

    # load training set and validation set

    def set_tf_keys(feed_dict, **kwargs):
        ret = dict([(k + ':0', v) for k, v in feed_dict.items()])
        ret.update([(k + ':0', v) for k, v in kwargs.items()])
        return ret

    session_config = tf.ConfigProto(allow_soft_placement=True)
    session_config.gpu_options.allow_growth = True

    with tf.Session(config=session_config) as sess:
        saver = tf.train.Saver(tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES), max_to_keep=5,
            keep_checkpoint_every_n_hours=5)
        if args.checkpoint is None:
            steps = 0
            tf.global_variables_initializer().run()
        else:
            if '\\' not in args.checkpoint and '/' not in args.checkpoint:
                args.checkpoint = os.path.join(
                    repoRoot, 'weights', args.checkpoint)
            if os.path.isdir(args.checkpoint):
                args.checkpoint = tf.train.latest_checkpoint(args.checkpoint)

            tf.global_variables_initializer().run(session=sess)
            checkpoints = args.checkpoint.split(';')
            if args.clear_steps:
                steps = 0
            else:
                steps = int(re.search('model-(\d+)', checkpoints[0]).group(1))
            for cp in checkpoints:
                saver = tf.train.Saver()
                saver.restore(sess, cp)

        data_args = eval('dict({})'.format(args.data_args))
        data_args.update(framework.data_args)
        print('data_args', data_args)
        dataset = Dataset(args, args.dataset, args.image_size, **data_args,
                    discriminator=args.aldk,
                    pretrained_flow_path=args.pretrained_flow_path)
        if args.finetune is not None:
            if 'finetune-train-%s' % args.finetune in dataset.schemes:
                dataset.schemes[Split.TRAIN] = dataset.schemes['finetune-train-%s' %
                                                               args.finetune]
            if 'finetune-val-%s' % args.finetune in dataset.schemes:
                dataset.schemes[Split.VALID] = dataset.schemes['finetune-val-%s' %
                                                               args.finetune]
            print('train', dataset.schemes[Split.TRAIN])
            print('val', dataset.schemes[Split.VALID])
        generator = dataset.generator(Split.TRAIN, batch_size=batchSize, loop=True)

        if not args.debug:
            if args.finetune is not None:
                run_id = os.path.basename(os.path.dirname(args.checkpoint))
                if not run_id.endswith('_ft' + args.finetune):
                    run_id = run_id + '_ft' + args.finetune
            else:
                pad = ''
                retry = 1
                run_id = args.output
                """
                while True:
                    dt = datetime.datetime.now(
                        tz=datetime.timezone(datetime.timedelta(hours=8)))
                    run_id = dt.strftime('%b%d-%H%M') + pad
                    modelPrefix = args.output#os.path.join(repoRoot, 'weights', run_id)
                    try:
                        os.makedirs(modelPrefix)
                        break
                    except Exception as e:
                        print('Conflict with {}! Retry...'.format(run_id))
                        pad = '_{}'.format(retry)
                        retry += 1
                """
            modelPrefix = os.path.join(repoRoot, 'weights', run_id)
            if not os.path.exists(modelPrefix):
                os.makedirs(modelPrefix)
            if args.name is not None:
                run_id += '_' + args.name
            if args.logs is None:
                log_dir = 'logs'
            else:
                log_dir = os.path.join('logs', args.logs)
            summary_path = os.path.join(repoRoot, log_dir, run_id)
            if not os.path.exists(summary_path):
                os.makedirs(summary_path)
            summaryWriter = tf.summary.FileWriter(summary_path, sess.graph)
            with open(os.path.join(modelPrefix, 'args.json'), 'w') as fo:
                json.dump(vars(args), fo)

        if args.finetune is not None:
            learningRates = [1e-5 / 2, 1e-5 / 2, 1e-5 / 2, 1e-5 / 4, 1e-5 / 8]
            #args.epochs = 1
        elif args.lr_set1:
            learningRates = [1e-4, 1e-4 / 8, 1e-4 / 16, 1e-4 / 32, 1e-4 / 64, 
                             1e-4 / 128]
        else:
            learningRates = [1e-4, 1e-4, 1e-4, 1e-4 / 2, 1e-4 / 4,
                             1e-4 / 8, 1e-4 / 16, 1e-4 / 32, 1e-4 / 64]

        def get_lr(steps):
            m = args.lr / learningRates[0]
            return m * learningRates[steps // iterationSize]

        def cosine_annealing(step, n_iters, n_cycles, lrate_max):
            iter_per_cycle = n_iters / n_cycles
            cos_inner = (pi * (step % iter_per_cycle)) / (iter_per_cycle)
            lr = lrate_max / 2 * (cos(cos_inner) + 1)
            return lr

        last_save_stamp = time.time()

        total_iter = args.round * args.epochs
        n_cycles = 5

        while True:
            if hasattr(framework, 'get_lr'):
                lr = framework.get_lr(steps, batchSize)
            else:
                if args.lr_set1:
                    decay_step = steps // iterationSize
                    lr = learningRates[decay_step if decay_step < len(learningRates) \
                            else (len(learningRates) - 1)]
                    lr = cosine_annealing(steps, total_iter, n_cycles, lr)
                else:
                    lr = get_lr(steps)
            t0 = default_timer()
            fd = next(generator)
            if args.aldk_learning_scheme:
                if (steps + 1) % 3 != 0 or steps >= 8000:
                    fd['gamma'] = np.ones([batchSize, 1], dtype=np.float32)
                else:
                    fd['gamma'] = np.ndarray([batchSize, 1], dtype=np.float32)
                    fd['gamma'][0] = args.gamma
            else:
                fd['gamma'] = np.ndarray([batchSize, 1], dtype=np.float32)
                fd['gamma'][0] = args.gamma

            fd.pop('mask', [])
            fd.pop('id1', [])
            fd.pop('id2', [])
            t1 = default_timer()
            tflearn.is_training(True, session=sess)
            summ, _ = sess.run([framework.summaryExtra, framework.adamOpt],
                               set_tf_keys(fd, learningRate=lr))

            for v in tf.Summary().FromString(summ).value:
                if v.tag == 'loss':
                    loss = v.simple_value

            steps += 1
            if args.debug or steps % 10 == 0:
                if steps > args.epochs * args.round:
                    break

                if not args.debug:
                    summaryWriter.add_summary(summ, steps)

                if steps % 1000 == 0:
                    if hasattr(framework, 'summaryImages'):
                        summ, = sess.run([framework.summaryImages],
                                         set_tf_keys(fd))
                        summaryWriter.add_summary(summ, steps)

                if steps < 500 or steps % 500 == 0:
                    print('*%s* ' % run_id,
                          time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                          'Steps %d, Total time %.2f, data %.2f%%. Loss %.3e lr %.3e' \
                              % (steps, default_timer() - t0, 
                                  (t1 - t0) / (default_timer() - t0), loss,lr), end='\n')

                # if time.time() - last_save_stamp > 3600 \
                #       or steps % iterationSize == iterationSize - 500:
                #     last_save_stamp = time.time()
                #     saver.save(sess, os.path.join(modelPrefix, 'model'),
                #                global_step=steps, write_meta_graph=False)
                if steps % 1000 == 0:
                    saver.save(sess, os.path.join(modelPrefix, 'model'),
                               global_step=steps, write_meta_graph=False)

                if args.debug or steps % args.val_steps == 0:
                    try:
                        val_gen = dataset.generator(
                            Split.VALID, loop=False, batch_size=batchSize)
                        metrics = framework.validate(
                            sess, val_gen, summary=True)
                        val_summ = tf.Summary(value=[
                            tf.Summary.Value(tag='val_' + k, simple_value=v) \
                                    for k, v in metrics.items()
                        ])
                        summaryWriter.add_summary(val_summ, steps)
                    except:
                        if steps == args.val_steps:
                            print('Step {}, validation failed!'.format(steps))
    print('Finished.')


if __name__ == '__main__':
    main()
