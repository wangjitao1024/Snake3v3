from abc import abstractmethod, ABC
from typing import Any

import numpy as np
import tensorflow as tf

import models.utils as utils
from models import model_registry
from models.distributions import CategoricalPd
from models.tf_v1_model import TFV1Model
from models.utils import conv, fc, conv_to_fc

__all__ = ['ACModel', 'ACMLPModel', 'ACCNNModel']

# from https://stackoverflow.com/questions/39088489/tensorflow-periodic-padding
# original name: periodic_padding_flexible
def circular_pad(tensor, axis,padding=1):# 实现循环padding
    """
        add periodic padding to a tensor for specified axis
        tensor: input tensor
        axis: on or multiple axis to pad along, int or tuple
        padding: number of cells to pad, int or tuple

        return: padded tensor
    """
    if isinstance(axis,int):
        axis = (axis,)
    if isinstance(padding,int):
        padding = (padding,)

    ndim = len(tensor.shape)
    for ax,p in zip(axis,padding):
        # create a slice object that selects everything from all axes,
        # except only 0:p for the specified for right, and -p: for left

        ind_right = [slice(-p,None) if i == ax else slice(None) for i in range(ndim)]
        ind_left = [slice(0, p) if i == ax else slice(None) for i in range(ndim)]
        right = tensor[ind_right]
        left = tensor[ind_left]
        middle = tensor
        tensor = tf.concat([right,middle,left], axis=ax)

    return tensor

def cirbasicblock(input,title,filter_num,firstpad,stride=1):
    activ = tf.nn.relu
    convout = activ(conv(circular_pad(input,(1,2),firstpad), '{}c1'.format(title), nf=filter_num, rf=3, stride=stride, init_scale=np.sqrt(2)))
    convout = conv(circular_pad(convout,(1,2),(1,1)), '{}c2'.format(title), nf=filter_num, rf=3, stride=1, init_scale=np.sqrt(2))
    if stride != 1:
        resout = conv(input, '{}c_res'.format(title), nf=filter_num, rf=1, stride=stride, init_scale=np.sqrt(2))
    else:
        resout = input
    #print("{}convout_shape:{}".format(title,convout.shape))
    #print("{}resout_shape:{}".format(title,resout.shape))
    output = activ(convout + resout)
    return output        

class ACModel(TFV1Model, ABC):
    def __init__(self, observation_space, action_space, config=None, model_id='0', *args, **kwargs):
        with tf.variable_scope(model_id):
            self.x_ph = utils.placeholder(shape=observation_space, dtype='uint8')
            self.encoded_x_ph = tf.to_float(self.x_ph)
            self.a_ph = utils.placeholder(dtype=tf.int32)
            self.legal_action = tf.placeholder(dtype= tf.int32, shape=(None, 4)) 
        self.logits = None
        self.vf = None

        super(ACModel, self).__init__(observation_space, action_space, config, model_id, scope=model_id,
                                      *args, **kwargs)

        pd = CategoricalPd(self.logits)
        self.action = pd.sample()
        self.neglogp = pd.neglogp(self.action)
        self.neglogp_a = pd.neglogp(self.a_ph)
        self.entropy = pd.entropy()

    def forward(self, states: Any, *args, **kwargs) -> Any:
        return self.sess.run([self.action, self.vf, self.neglogp], feed_dict={self.x_ph: states})

    @abstractmethod
    def build(self, *args, **kwargs) -> None:
        pass


@model_registry.register('acmlp')
class ACMLPModel(ACModel):

    def build(self) -> None:
        with tf.variable_scope(self.scope):
            with tf.variable_scope('pi'):
                self.logits = utils.mlp(self.encoded_x_ph, [64, 64, self.action_space], tf.tanh)

            with tf.variable_scope('v'):
                self.vf = tf.squeeze(utils.mlp(self.encoded_x_ph, [64, 64, 1], tf.tanh), axis=1)


@model_registry.register('accnn')
class ACCNNModel(ACModel, ABC):

    def build(self, *args, **kwargs) -> None:
        with tf.variable_scope(self.scope):
            with tf.variable_scope('cnn_base'):
                # scaled_images = tf.cast(self.encoded_x_ph, tf.float32) / 255.
                input_images = tf.cast(self.encoded_x_ph, tf.float32)
                activ = tf.nn.relu

                outstem = activ(conv(circular_pad(input_images,(1,2),(1,1)), 'c_stem', nf=16, rf=3, stride=1, init_scale=np.sqrt(2)))
                #print(outstem.shape)
                outstem = tf.nn.max_pool(circular_pad(outstem,(1,2),(1,1)),[1,2,2,1],[1,1,1,1],padding='VALID')
                #print(outstem.shape)

                outresnet = cirbasicblock(outstem,"rb1_1",16,(1,1),1)
                outresnet = cirbasicblock(outresnet,"rb1_2",16,(1,1),1)
                outresnet = cirbasicblock(outresnet,"rb2_1",16,(1,1),2)
                outresnet = cirbasicblock(outresnet,"rb2_2",16,(1,1),1)
                outresnet = cirbasicblock(outresnet,"rb3_1",32,(1,1),2)
                outresnet = cirbasicblock(outresnet,"rb3_2",32,(1,1),1)
                outresnet = cirbasicblock(outresnet,"rb4_1",32,(0,1),2)
                outresnet = cirbasicblock(outresnet,"rb4_2",32,(1,1),1)
                outresnet = conv_to_fc(outresnet)

                latent = activ(fc(outresnet, 'fc1', nh=64, init_scale=np.sqrt(2)))

            with tf.variable_scope('pi'):
                pih1 = activ(fc(latent, 'pi_fc1', 64, init_scale=1))
                pih2 = activ(fc(pih1, 'pi_fc2', 64, init_scale=1))
                logits_without_mask = fc(pih2, 'pi_fc3', self.action_space, init_scale=1)
                self.logits = logits_without_mask + 1000. *tf.to_float(self.legal_action-1)

            with tf.variable_scope('v'):
                vh1 = activ(fc(latent, 'v_fc1', 64, init_scale=1))
                vh2 = activ(fc(vh1, 'v_fc2', 64, init_scale=1))
                self.vf = tf.squeeze(fc(vh2, 'v_fc3', 1, init_scale=1), axis=1)

