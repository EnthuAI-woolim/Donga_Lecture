import tensorflow as tf
import numpy as np

class ImageClassifier_MLP:
    def __init__(self, img_shape_x, img_shape_y, num_labels):
        self.img_shape_x = img_shape_x
        self.img_shape_y = img_shape_y
        self.num_labels = num_labels
        self.classifier = None

    def fit(self, train_imgs, train_labels, num_epochs):
        self.classifier.fit(train_imgs, train_labels, epochs=num_epochs)

    def predict(self, test_imgs):
        predictions = self.classifier.predict(test_imgs)
        return predictions

    def build_MLP_model(self):
        input_layer = tf.keras.layers.Input(shape=[self.img_shape_x, self.img_shape_y, ])
        flatten_layer = tf.keras.layers.Flatten()(input_layer)

        ac_func_relu = tf.keras.activations.relu
        hidden_layer_1 = tf.keras.layers.Dense(units=128, activation=ac_func_relu)(flatten_layer)
        hidden_layer_2 = tf.keras.layers.Dense(units=128, activation=ac_func_relu)(hidden_layer_1)

        ac_func_softmax = tf.keras.activations.softmax
        output_layer = tf.keras.layers.Dense(units=self.num_labels, activation=ac_func_softmax)(hidden_layer_2)
        classifier_model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

        opt_alg = tf.keras.optimizers.Adam(learning_rate=0.001)
        loss_cross_e = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        classifier_model.compile(optimizer=opt_alg, loss=loss_cross_e, metrics=['accuracy'])

        self.classifier = classifier_model

    # class ImageClassifier_MLP 의 static method
    @staticmethod
    def to_onehotvec_label(index_labels, dim):
        num_labels = len(index_labels)
        onehotvec_labels = np.zeros((num_labels, dim))
        for i, idx in enumerate(index_labels):
            onehotvec_labels[i][idx] = 1.0
        onehotvec_labels_tf = tf.convert_to_tensor(onehotvec_labels)
        return onehotvec_labels_tf

