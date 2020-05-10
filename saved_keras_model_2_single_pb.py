#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020-05-08 20:00
# @Author  : YouSheng
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.framework.graph_util import convert_variables_to_constants
import tensorflow as tf
import os
import keras.backend as K
import shutil
from keras.preprocessing import image
import numpy as np
from keras.applications.resnet50 import preprocess_input, ResNet50
from absl import logging


def save_with_single_pb(model, output_dir, use_saved_model=True):
    """
    convert Keras model to the single .pb file in output_dir
    :param model: Keras model
    :param output_dir: the output dir to save .pb file
    :param use_saved_model: use the tf.saved_model API to save the graph_def, otherwise just write into file
    :return:
    """
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    sess = K.get_session()

    if LOG:
        logging.info("input tensor name is: {}".format(model.get_input_at(0).name))  # input_1:0
        logging.info(
            'output tensor name is: {}'.format(model.get_output_at(0).name))  # global_average_pooling2d_1/Mean:0
        logging.info('original graph.node num is {}'.format(len(sess.graph_def.node)))

    output_node_name = model.get_output_at(0).op.name  # 'global_average_pooling2d_1/Mean'
    constant_graph_def = convert_variables_to_constants(sess, sess.graph_def, [output_node_name])

    if use_saved_model:
        x_name = model.get_input_at(0).name  # input_1:0
        y_name = model.get_output_at(0).name  # global_average_pooling2d_1/Mean:0
        write_saved_model(output_dir, constant_graph_def, x_name, {'output': y_name})
    else:
        # write out constant graph directly
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        with tf.gfile.GFile(os.path.join(output_dir, 'save_model_wo_signature.pb'), 'wb') as f:
            f.write(constant_graph_def.SerializeToString())


def write_saved_model(output_dir, constant_graph_def, x_name, outputs_map):
    """
    write constant graph_def into output_dir as a pb file using tf.saved_model API
    将静态的graph_def使用 tf.saved_model API 写入目标文件夹中
    :param output_dir:
    :param constant_graph_def: graph_def returned by graph_util.convert_variables_to_constants()
    :param x_name: the name of input tensor
    :param outputs_map: <the signature name, name of the output tensor> dict
    :return:
    """
    with tf.Graph().as_default() as graph:
        with tf.Session(graph=graph) as sess:
            # must have name='', otherwise imported nodes will be auto added prefix 'import/'
            tf.import_graph_def(constant_graph_def, name='')

            builder = tf.saved_model.builder.SavedModelBuilder(output_dir)
            if LOG:
                logging.info('new constant graph.node num is {}'.format(len(sess.graph_def.node)))
            if CHECK_VALUE:
                print('-------------New graph-------------\n')
                preds = sess.run(sess.graph.get_tensor_by_name(outputs_map['output']),
                                 feed_dict={sess.graph.get_tensor_by_name(x_name): x_input})
                logging.info('after constant predict output: {}'.format(np.array(preds).squeeze()[:10]))

            tensor_info_inputs = {
                'inputs': tf.saved_model.utils.build_tensor_info(sess.graph.get_tensor_by_name(x_name)),
                # 'learning_phase': tf.saved_model.utils.build_tensor_info(K.learning_phase())
            }
            tensor_info_outputs = {}
            for k, v in outputs_map.items():
                tensor_info_outputs[k] = tf.saved_model.utils.build_tensor_info(sess.graph.get_tensor_by_name(v))

            signature_def = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs=tensor_info_inputs,
                    outputs=tensor_info_outputs,
                    method_name=signature_constants.PREDICT_METHOD_NAME))  # "tensorflow/serving/predict"

            builder.add_meta_graph_and_variables(
                sess, [tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                    signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:  # "serving_default"
                        signature_def,
                },
            )
            builder.save(as_text=False)


def save(use_saved_model=True):
    model_ = ResNet50(include_top=False, pooling='avg', weights='imagenet')
    logging.info('K.learning_phase() = {}'.format(K.learning_phase()))

    if CHECK_VALUE:
        results = model_.predict(x_input)
        logging.info('keras.model.predict: {}'.format(results.squeeze()[:10]))
    save_with_single_pb(model=model_, output_dir=OUT_PUTDIR, use_saved_model=use_saved_model)


def load_saved_model(saved_model_dir):
    """
        读取保存好的tf-serving .pb模型，参数存成了constant依附于graph中
        :param saved_model_dir:
        :return:
        """
    print("\nStart to load tensorflow saved model...")
    sess = tf.Session()

    print('\tReading params from {}'.format(saved_model_dir))

    signature_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    input_key = 'inputs'
    output_key = 'output'
    meta_graph_def = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], saved_model_dir)
    signature_map = meta_graph_def.signature_def

    x_tensor_info = signature_map[signature_key].inputs[input_key]
    y_tensor_info = signature_map[signature_key].outputs[output_key]
    print("\tinput_tensor_name: ", x_tensor_info.name)
    print("\toutput_tensor_name: ", y_tensor_info.name)

    images = tf.saved_model.utils.get_tensor_from_tensor_info(x_tensor_info, sess.graph)
    outputs = tf.saved_model.utils.get_tensor_from_tensor_info(y_tensor_info, sess.graph)

    print("\tFinish reconstruct tensorflow model.\tThere are {} nodes in current graph.".format(len(sess.graph_def.node)))

    return sess, images, outputs

def test_constant_model(input_name='input_1:0', output_name='global_average_pooling2d_1/Mean:0'):
    from tensorflow.python.platform import gfile
    with tf.Session() as sess:
        pb_dir = os.path.join(OUT_PUTDIR, 'save_model_wo_signature.pb')
        with gfile.FastGFile(pb_dir, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        result = tf.import_graph_def(graph_def, return_elements=[output_name], name='')
        print('\ttest_constant_model output tensor', result)
        preds = sess.run(result, feed_dict={sess.graph.get_tensor_by_name(input_name): x_input})
        print('\ttest_constant_model.preds=', np.array(preds).squeeze()[:10])


def test(use_saved_model=True):
    """
    :param use_saved_model: load and test the pb model which is saved with tf.saved_model API or just with write file
    """
    if use_saved_model:
        sess, images, scores = load_saved_model(OUT_PUTDIR)
        pred_attributes = sess.run([scores],
                                   feed_dict={images: x_input})
        logging.info("predict value: {}".format(np.array(pred_attributes).squeeze()[:10]))
        sess.close()
    else:
        # if just write constant graph_def into binary pb file, it must feed tensor name when loading.
        test_constant_model(input_name='input_1:0', output_name='global_average_pooling2d_1/Mean:0')


def prepare_test_img_input(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x_input = preprocess_input(x)
    return x_input

if __name__ == '__main__':
    LOG = True
    logging.set_verbosity(logging.INFO)
    CHECK_VALUE = True  # to check the predict values are the same or not, between origin Keras model and output pb model
    OUT_PUTDIR = 'output_single_pb'
    
    x_input = prepare_test_img_input(img_path='images/34rews.jpg')

    # There is a bug if call K.set_learning_phase(0)
    # when using keras.layer.BN and convert_variables_to_constants together.
    # Refer to :https://github.com/amir-abdi/keras_to_tensorflow/issues/109
    # K.set_learning_phase(0)

    save(use_saved_model=True)
    # test(use_saved_model=True)
