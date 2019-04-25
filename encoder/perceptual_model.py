import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
import keras.backend as K


def load_images(images_list, img_size):
    loaded_images = list()
    for img_path in images_list:
        img = image.load_img(img_path, target_size=(img_size, img_size))
        img = np.expand_dims(img, 0)
        loaded_images.append(img)
    loaded_images = np.vstack(loaded_images)
    preprocessed_images = preprocess_input(loaded_images)
    return preprocessed_images


class PerceptualModel:
    def __init__(self, img_size, layer=9, batch_size=1, sess=None):
        self.sess = tf.get_default_session() if sess is None else sess
        K.set_session(self.sess)
        self.img_size = img_size
        self.layer = layer
        self.batch_size = batch_size

        self.perceptual_model = None
        self.ref_img_features = None
        self.features_weight = None
        self.loss = None

        self.features_weight_placeholder = None
        self.img_features_placeholder = None

        self.features_weight_op = None
        self.img_features_op = None

    def build_perceptual_model(self, generated_image_tensor):
        vgg16 = VGG16(include_top=False, input_shape=(self.img_size, self.img_size, 3))
        self.perceptual_model = Model(vgg16.input, vgg16.layers[self.layer].output)
        generated_image = preprocess_input(tf.image.resize_images(generated_image_tensor,
                                                                  (self.img_size, self.img_size), method=1))
        generated_img_features = self.perceptual_model(generated_image)

        # want to test out some different initializers than zeros ,, 
        # based on poking around it seems like xavier/glorot uniform might be good with vgg16

        # self.ref_img_features = tf.get_variable('ref_img_features', shape=generated_img_features.shape,
        #                                         dtype='float32', initializer=tf.initializers.zeros())
        # self.features_weight = tf.get_variable('features_weight', shape=generated_img_features.shape,
        #                                        dtype='float32', initializer=tf.initializers.zeros())

        self.ref_img_features = tf.get_variable('ref_img_features', shape=generated_img_features.shape,
                                                dtype='float32', initializer=tf.initializers.glorot_normal())
        self.features_weight = tf.get_variable('features_weight', shape=generated_img_features.shape,
                                               dtype='float32', initializer=tf.initializers.glorot_normal())                                       
        self.sess.run([self.features_weight.initializer, self.features_weight.initializer])

        self.loss = tf.losses.mean_squared_error(self.features_weight * self.ref_img_features,
                                                 self.features_weight * generated_img_features) / 82890.0
        self.features_weight_placeholder = tf.placeholder(self.features_weight.dtype, shape=self.features_weight.get_shape())
        self.img_features_placeholder = tf.placeholder(self.ref_img_features.dtype, shape=self.ref_img_features.get_shape())

        self.features_weight_op = self.features_weight.assign(self.features_weight_placeholder)
        self.img_features_op = self.ref_img_features.assign(self.img_features_placeholder)

    def set_reference_images(self, images_list):
        assert(len(images_list) != 0 and len(images_list) <= self.batch_size)
        loaded_image = load_images(images_list, self.img_size)
        image_features = self.perceptual_model.predict_on_batch(loaded_image)

        # in case if number of images less than actual batch size
        # can be optimized further
        weight_mask = np.ones(self.features_weight.shape)
        if len(images_list) != self.batch_size:
            features_space = list(self.features_weight.shape[1:])
            existing_features_shape = [len(images_list)] + features_space
            empty_features_shape = [self.batch_size - len(images_list)] + features_space

            existing_examples = np.ones(shape=existing_features_shape)
            empty_examples = np.zeros(shape=empty_features_shape)
            weight_mask = np.vstack([existing_examples, empty_examples])

            image_features = np.vstack([image_features, np.zeros(empty_features_shape)])

        self.sess.run(self.features_weight_op, {self.features_weight_placeholder: weight_mask})
        self.sess.run(self.img_features_op, {self.img_features_placeholder: image_features})

    def optimize(self, vars_to_optimize, iterations=500, learning_rate=1., weight_decay=0.3):

        vars_to_optimize = vars_to_optimize if isinstance(vars_to_optimize, list) else [vars_to_optimize]
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

        # test out ADAM with no LR sched/decay
        # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        

        # or use wd version of Adam
        # weight_decay = 0.3
        optimizer = tf.contrib.opt.AdamWOptimizer(weight_decay=weight_decay,learning_rate=learning_rate)

        min_op = optimizer.minimize(self.loss, var_list=[vars_to_optimize])

        self.sess.run(tf.variables_initializer(optimizer.variables()))

        # have to re-init vars here since Adam or else problems
        # self.sess.run([tf.global_variables_initializer(),self.features_weight.initializer, self.features_weight.initializer])

        for _ in range(iterations):
            _, loss = self.sess.run([min_op, self.loss])
            yield loss

