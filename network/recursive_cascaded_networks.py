"""
Recursive Cascaded Networks
modified from https://github.com/microsoft/Recursive-Cascaded-Networks/blob/master/network/recursive_cascaded_networks.py

"""

import tensorflow as tf
import tflearn
import numpy as np

from .utils import Network
from .base_networks import VTN, VTNAffineStem, VoxelMorph, \
        LDR, UNet, MobileNetAffine, LDRUtilStem
from .spatial_transformer import Dense3DSpatialTransformer, Fast3DTransformer
from .trilinear_sampler import TrilinearSampler
import tensorflow_utils as tf_utils


def mask_metrics(seg1, seg2):
    ''' Given two segmentation seg1, seg2, 0 for background 255 for foreground.
    Calculate the Dice score 
    $ 2 * | seg1 \cap seg2 | / (|seg1| + |seg2|) $
    and the Jacc score
    $ | seg1 \cap seg2 | / (|seg1 \cup seg2|) $
    '''
    sizes = np.prod(seg1.shape.as_list()[1:])
    seg1 = tf.reshape(seg1, [-1, sizes])
    seg2 = tf.reshape(seg2, [-1, sizes])
    seg1 = tf.cast(seg1 > 128, tf.float32)
    seg2 = tf.cast(seg2 > 128, tf.float32)
    dice_score = 2.0 * tf.reduce_sum(seg1 * seg2, axis=-1) / (
        tf.reduce_sum(seg1, axis=-1) + tf.reduce_sum(seg2, axis=-1))
    union = tf.reduce_sum(tf.maximum(seg1, seg2), axis=-1)
    return (dice_score, tf.reduce_sum(
        tf.minimum(seg1, seg2), axis=-1) / tf.maximum(0.01, union))

def check_discriminator_loss(flow, d_loss):
    if np.sum(flow) != 0:
        return d_loss
    else:
        return np.zeros(1, dtype=np.float32)


class RecursiveCascadedNetworks(Network):
    default_params = {
        'weight': 1,
        'raw_weight': 1,
        'reg_weight': 1,
    }

    def __init__(self, name, framework,
                 base_network, n_cascades, rep=1,
                 det_factor=0.1, ortho_factor=0.1, reg_factor=1.0,
                 extra_losses={}, warp_gradient=True,
                 fast_reconstruction=False, warp_padding=False,
                 aldk=False, dataset="",
                 flow_multiplier=1.0, beta=0.1, lamb=0.1, gamma=0.5,
                 **kwargs):
        super().__init__(name)
        self.det_factor = det_factor
        self.ortho_factor = ortho_factor
        self.reg_factor = reg_factor
        self.aldk = aldk
        self.flow_multiplier = flow_multiplier
        self.gamma = gamma
        self.lamb = lamb
        self.beta = beta
        self.b_name = base_network
        self.base_network = eval(base_network)
        self.stems = None
        self.dataset = dataset

        if self.b_name == 'LDR':
            if self.dataset.find('liver') != -1:
                self.stems = sum(
                    [[(self.base_network("ldr_util", flow_multiplier=flow_multiplier / (n_cascades + 1)),
                                  {'raw_weight': 0})] * rep +
                        [(self.base_network("deform_stem_" + str(i),
                                        flow_multiplier=flow_multiplier / (n_cascades + 1)),
                      {'raw_weight': 0})] * rep for i in range(n_cascades)], [])
            else:
                self.stems = [(LDRUtilStem('ldr_util', trainable=True),
                               {'raw_weight': 0, 'reg_weight': 0})] \
                             + sum([[(self.base_network("deform_stem_" + str(i),
                                                        flow_multiplier=flow_multiplier/ n_cascades),
                                      {'raw_weight': 0})] * rep \
                                    for i in range(n_cascades)], [])
        else:
            self.stems = [(VTNAffineStem('affine_stem', trainable=True),
                          {'raw_weight': 0, 'reg_weight': 0})] \
                          + sum([[(self.base_network("deform_stem_" + str(i),
                            flow_multiplier=flow_multiplier/ n_cascades),
                            {'raw_weight': 0})] * rep for i in range(n_cascades)], [])

        self.stems[-1][1]['raw_weight'] = 1
        for _, param in self.stems:
            for k, v in self.default_params.items():
                if k not in param:
                    param[k] = v
        print(self.stems)

        self.framework = framework
        self.warp_gradient = warp_gradient
        self.fast_reconstruction = fast_reconstruction

        self.reconstruction = Fast3DTransformer(
            warp_padding) if fast_reconstruction \
                    else Dense3DSpatialTransformer(warp_padding)
        self.trilinear_sampler = TrilinearSampler()
        self.dis_c = [16, 32, 64, 128, 256]

    @property
    def trainable_variables(self):
        return list(set(sum(
            [stem.trainable_variables for stem, params in self.stems], [])))

    @property
    def data_args(self):
        return dict()

    def basicDiscriminator(self, data, name='d_', is_reuse=False):
        with tf.variable_scope(name) as scope:
            if is_reuse is True:
                scope.reuse_variables()
            tf_utils.print_activations(data)

            # from (N, 128, 128, 128, 3) to (N, 64, 64, 64, 16)
            h0_conv = tf_utils.conv3d(data, self.dis_c[0],
                    k_h=5, k_w=5, k_d=5, name='h0_conv2d')
            h0_lrelu = tf_utils.lrelu(h0_conv, name='h0_lrelu')

            # from (N, 64, 64, 64, 16) to (N, 32, 32, 32, 32)
            h1_conv = tf_utils.conv3d(h0_lrelu, self.dis_c[1],
                    k_h=5, k_w=5, k_d=5, name='h1_conv2d')
            h1_lrelu = tf_utils.lrelu(h1_conv, name='h1_lrelu')

            # from (N, 32, 32, 32, 32) to (N, 16, 16, 16, 64)
            h2_conv = tf_utils.conv3d(h1_lrelu, self.dis_c[2],
                    k_h=5, k_w=5, k_d=5, name='h2_conv2d')
            h2_lrelu = tf_utils.lrelu(h2_conv, name='h2_lrelu')

            # from (N, 16, 16, 16, 64) to (N, 8, 8, 8, 128)
            h3_conv = tf_utils.conv3d(h2_lrelu, self.dis_c[3],
                    k_h=5, k_w=5, k_d=5, name='h3_conv2d')
            h3_lrelu = tf_utils.lrelu(h3_conv, name='h2_lrelu')

            # from (N, 8, 8, 8, 128) to (N, 4, 4, 4, 256)
            h4_conv = tf_utils.conv3d(h3_lrelu, self.dis_c[4],
                    k_h=5, k_w=5, k_d=5, name='h4_conv2d')
            h4_lrelu = tf_utils.lrelu(h4_conv, name='h2_lrelu')

            # from (N, 4, 4, 4, 256) to (N, 4096) and to (N, 1)
            h4_flatten = tf.reshape(h4_lrelu, (-1, h4_lrelu.shape[1] \
                    * h4_lrelu.shape[2] * h4_lrelu.shape[3] * h4_lrelu.shape[4]))
            h5_linear = tf_utils.linear(h4_flatten, 1, name='h3_linear')

            return tf.nn.sigmoid(h5_linear), h5_linear

    def distilled_student(self, student, teacher, beta=0.):
        return self.beta * teacher + (1 - self.beta) * student

    def gradient_penalty(self, flow, pretrained_flow):
        alpha = tf.random_uniform(shape=[128, 128, 128, 1, 3], minval=0., maxval=1.)
        distilled_flow = self.distilled_student(flow, pretrained_flow)
        differences = distilled_flow - pretrained_flow
        interpolates = pretrained_flow + (alpha * differences)
        interpolates = tf.transpose(interpolates, perm=[3, 0, 1, 2, 4])
        gradients = tf.gradients(self.basicDiscriminator(interpolates, is_reuse=True),
                                [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2, 3]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)

        return gradient_penalty

    @staticmethod
    def get_model_no_params(model):
        no_params = 0
        for variable in model.trainable_variables:
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            no_params += variable_parameters

        return no_params


    def build(self, img1, img2, seg1, seg2, pt1, pt2, pretrained_flow, gamma):
        '''
        :param img1: fixed image
        :param img2:
        :param seg1:
        :param seg2:
        :param pt1:
        :param pt2:
        :return:
        '''
        stem_results = []
        flows = []

        init_stem_params, one_cascade_params, n_cas = 0, 0, len(self.stems) - 1
        if self.dataset.find('liver') != -1 and self.b_name == 'LDR':
            stem_result = None
            for idx, (stem, params) in enumerate(self.stems):
                if stem_result != None:
                    flows.append(stem_result['agg_flow'])
                if self.warp_gradient:
                    if len(stem_results) > 0:
                        stem_result = stem(img1, stem_results[-1]['warped'])
                    else:
                        stem_result = stem(img1, img2)
                    if idx == 0:
                        init_stem_params = self.get_model_no_params(stem)
                    else:
                        one_cascade_params = self.get_model_no_params(stem)
                else:
                    stem_result = stem(img1, tf.stop_gradient(
                        stem_results[-1]['warped']))

                '''
                if len(stem_results) == 1 and 'W' in stem_results[-1]:
                    I = tf.constant([1, 0, 0, 0, 1, 0, 0, 0, 1],
                                    tf.float32, [1, 3, 3])
                    stem_result['agg_flow'] = tf.einsum(
                        'bij,bxyzj->bxyzi', stem_results[-1]['W'] + I, 
                        stem_result['flow']) + stem_results[-1]['flow']
                '''
                if len(stem_results) > 0:
                    stem_result['agg_flow'] = self.reconstruction(
                        [stem_results[-1]['agg_flow'], stem_result['flow']]) \
                                + stem_result['flow']
                else:
                    stem_result['agg_flow'] = stem_result['flow']
                stem_result['warped'] = self.reconstruction(
                    [img2, stem_result['agg_flow']])
                stem_results.append(stem_result)
        else:
            stem_result = self.stems[0][0](img1, img2)
            init_stem_params = self.get_model_no_params(self.stems[0][0])
            stem_result['warped'] = self.reconstruction(
                [img2, stem_result['flow']])
            stem_result['agg_flow'] = stem_result['flow']
            stem_results.append(stem_result)

            # Cascade Deformation
            for stem, params in self.stems[1:]:
                flows.append(stem_result['agg_flow'])
                if self.warp_gradient:
                    stem_result = stem(img1, stem_results[-1]['warped'])
                    one_cascade_params = self.get_model_no_params(stem)
                else:
                    stem_result = stem(img1, tf.stop_gradient(
                        stem_results[-1]['warped']))

                if len(stem_results) == 1 and 'W' in stem_results[-1]:
                    I = tf.constant([1, 0, 0, 0, 1, 0, 0, 0, 1],
                                    tf.float32, [1, 3, 3])
                    stem_result['agg_flow'] = tf.einsum(
                        'bij,bxyzj->bxyzi', stem_results[-1]['W'] + I,
                        stem_result['flow']) + stem_results[-1]['flow']
                else:
                    stem_result['agg_flow'] = self.reconstruction(
                        [stem_results[-1]['agg_flow'], stem_result['flow']]) \
                                              + stem_result['flow']
                stem_result['warped'] = self.reconstruction(
                    [img2, stem_result['agg_flow']])
                stem_results.append(stem_result)

        # unsupervised learning with simlarity loss and regularization loss
        for stem_result, (stem, params) in zip(stem_results, self.stems):
            if 'W' in stem_result:
                stem_result['loss'] = stem_result['det_loss'] * \
                    self.det_factor + \
                    stem_result['ortho_loss'] * self.ortho_factor
                if params['raw_weight'] > 0:
                    stem_result['raw_loss'] = self.similarity_loss(
                        img1, stem_result['warped'])
                    stem_result['loss'] = stem_result['loss'] + \
                        stem_result['raw_loss'] * params['raw_weight']
            else:
                if params['raw_weight'] > 0:
                    stem_result['raw_loss'] = self.similarity_loss(
                        img1, stem_result['warped'])
                if params['reg_weight'] > 0:
                    stem_result['reg_loss'] = self.regularize_loss(
                        stem_result['flow']) * self.reg_factor
                stem_result['loss'] = sum(
                    [stem_result[k] * params[k.replace('loss', 'weight')] \
                            for k in stem_result if k.endswith('loss')])

        ret = {}

        flow = stem_results[-1]['agg_flow']
        warped = stem_results[-1]['warped']
        d_loss = None

        # calculate discriminator 
        # losspy_func(func=check_discriminator_loss,
        #         inp=[pretrained_flow, d_loss], Tout=tf.float32)
        if self.aldk:
            _, flow_logits = self.basicDiscriminator(flow)
            _, pretrained_flow_logits = self.basicDiscriminator(pretrained_flow,
                    is_reuse=True)
            d_loss = tf.reduce_mean(flow_logits) - tf.reduce_mean(pretrained_flow_logits)
            gp_loss = self.gradient_penalty(flow_logits, pretrained_flow_logits)
            d_loss = d_loss + self.lamb * gp_loss
            d_loss = tf.py_func(func=check_discriminator_loss,
                    inp=[pretrained_flow, d_loss], Tout=tf.float32)

        jacobian_det = self.jacobian_det(flow)
        loss = sum([r['loss'] * params['weight']
                    for r, (stem, params) in zip(stem_results, self.stems)])
        if self.aldk:
            loss = sum([gamma[0] * loss, (1 - gamma[0]) * d_loss])

        pt_mask1 = tf.reduce_any(tf.reduce_any(pt1 >= 0, -1), -1)
        pt_mask2 = tf.reduce_any(tf.reduce_any(pt2 >= 0, -1), -1)
        pt1 = tf.maximum(pt1, 0.0)

        moving_pt1 = pt1 + self.trilinear_sampler([flow, pt1])

        pt_mask = tf.cast(pt_mask1, tf.float32) * tf.cast(pt_mask2, tf.float32)
        landmark_dists = tf.sqrt(tf.reduce_sum(
            (moving_pt1 - pt2) ** 2, axis=-1)) * tf.expand_dims(pt_mask, axis=-1)
        landmark_dist = tf.reduce_mean(landmark_dists, axis=-1)

        if self.framework.segmentation_class_value is None:
            seg_fixed = seg1
            warped_seg_moving = self.reconstruction([seg2, flow])
            dice_score, jacc_score = mask_metrics(seg_fixed, warped_seg_moving)
            jaccs = [jacc_score]
            dices = [dice_score]
        else:
            def mask_class(seg, value):
                return tf.cast(tf.abs(seg - value) < 0.5, tf.float32) * 255
            jaccs = []
            dices = []
            fixed_segs = []
            warped_segs = []
            for k, v in self.framework.segmentation_class_value.items():
                #print('Segmentation {}, {}'.format(k, v))
                fixed_seg_class = mask_class(seg1, v)
                warped_seg_class = self.reconstruction(
                    [mask_class(seg2, v), flow])
                class_dice, class_jacc = mask_metrics(
                    fixed_seg_class, warped_seg_class)
                ret['jacc_{}'.format(k)] = class_jacc
                jaccs.append(class_jacc)
                dices.append(class_dice)
                fixed_segs.append(fixed_seg_class)
                warped_segs.append(warped_seg_class)
            seg_fixed = tf.stack(fixed_segs, axis=-1)
            warped_seg_moving = tf.stack(warped_segs, axis=-1)
            dice_score, jacc_score = tf.add_n(
                dices) / len(dices), tf.add_n(jaccs) / len(jaccs)

        ret.update({'loss': tf.reshape(loss, (1, )),
                    'dice_score': dice_score,
                    'jacc_score': jacc_score,
                    'dices': tf.stack(dices, axis=-1),
                    'jaccs': tf.stack(jaccs, axis=-1),
                    'landmark_dist': landmark_dist,
                    'landmark_dists': landmark_dists,
                    'real_flow': flow,
                    'pt_mask': pt_mask,
                    'reconstruction': warped * 255.0,
                    'image_reconstruct': warped,
                    'warped_moving': warped * 255.0,
                    'seg_fixed': seg_fixed,
                    'warped_seg_moving': warped_seg_moving,
                    'image_fixed': img1,
                    'moving_pt': moving_pt1,
                    #'flows': tf.stack(flows, 1),
                    'jacobian_det': jacobian_det})

        for i, r in enumerate(stem_results):
            for k in r:
                if k.endswith('loss'):
                    ret['{}_{}'.format(i, k)] = r[k]
            ret['warped_seg_moving_%d' %
                i] = self.reconstruction([seg2, r['agg_flow']])
            ret['warped_moving_%d' % i] = r['warped']
            ret['flow_%d' % i] = r['flow']
            ret['real_flow_%d' % i] = r['agg_flow']

        total_parameters = 0
        for stem, param in self.stems:
            total_parameters += self.get_model_no_params(stem)

        print('number of Cascade: ', n_cas)
        print('Total params: ', total_parameters)
        print('Init Stem Params: ', init_stem_params)
        print('One cascade part params: ', one_cascade_params)
        assert total_parameters == n_cas*one_cascade_params + init_stem_params

        tf_total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            tf_total_parameters += variable_parameters
        # print('Discriminator params: ', tf_total_parameters - total_parameters)
        return ret

    def similarity_loss(self, img1, warped_img2):
        sizes = np.prod(img1.shape.as_list()[1:])
        flatten1 = tf.reshape(img1, [-1, sizes])
        flatten2 = tf.reshape(warped_img2, [-1, sizes])

        if self.fast_reconstruction:
            _, pearson_r, _ = tf.user_ops.linear_similarity(flatten1, flatten2)
        else:
            mean1 = tf.reshape(tf.reduce_mean(flatten1, axis=-1), [-1, 1])
            mean2 = tf.reshape(tf.reduce_mean(flatten2, axis=-1), [-1, 1])
            var1 = tf.reduce_mean(tf.square(flatten1 - mean1), axis=-1)
            var2 = tf.reduce_mean(tf.square(flatten2 - mean2), axis=-1)
            cov12 = tf.reduce_mean(
                (flatten1 - mean1) * (flatten2 - mean2), axis=-1)
            pearson_r = cov12 / tf.sqrt((var1 + 1e-6) * (var2 + 1e-6))

        raw_loss = 1 - pearson_r
        raw_loss = tf.reduce_sum(raw_loss)
        return raw_loss

    def regularize_loss(self, flow):
        ret = ((tf.nn.l2_loss(flow[:, 1:, :, :] - flow[:, :-1, :, :]) +
                tf.nn.l2_loss(flow[:, :, 1:, :] - flow[:, :, :-1, :]) +
                tf.nn.l2_loss(flow[:, :, :, 1:] - flow[:, :, :, :-1])) \
                        / np.prod(flow.shape.as_list()[1:5]))
        return ret

    def jacobian_det(self, flow):
        _, var = tf.nn.moments(tf.linalg.det(tf.stack([
            flow[:, 1:, :-1, :-1] - flow[:, :-1, :-1, :-1] +
            tf.constant([1, 0, 0], dtype=tf.float32),
            flow[:, :-1, 1:, :-1] - flow[:, :-1, :-1, :-1] +
            tf.constant([0, 1, 0], dtype=tf.float32),
            flow[:, :-1, :-1, 1:] - flow[:, :-1, :-1, :-1] +
            tf.constant([0, 0, 1], dtype=tf.float32)
        ], axis=-1)), axes=[1, 2, 3])
        return tf.sqrt(var)
