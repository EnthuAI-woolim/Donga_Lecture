import tensorflow as tf

class MLP:
    # "hidden_layer_conf" is the array indicates the number of layers (num_of_elements)
    # and the number of elements in each layer.
    def __init__(self, hidden_layer_conf, num_output_nodes):
        self.hidden_layer_conf = hidden_layer_conf
        self.num_output_nodes = num_output_nodes
        self.logic_op_model = None

    ### A member function of Class MLP_image
    def build_model(self):
        input_layer = tf.keras.Input(shape=[2, ]) # 벡터
        hidden_layers = input_layer

        ## xor_classifier.py에서의 코드에 의하면 self.hidden_layer_conf의 값은 “[4]” -> hidden layer 1층이고, node 수는 4개로 설정되어 있음
        if self.hidden_layer_conf is not None:
            # hidden_layer_conf는 리스트 형태로 각 인덱스의 값은 hidden layer에서 한 layer에서의 node 수가 저장됨
            for num_hidden_nodes in self.hidden_layer_conf:
                hidden_layers = tf.keras.layers.Dense(units=num_hidden_nodes,
                                                      activation=tf.keras.activations.sigmoid,
                                                      use_bias=True)(hidden_layers) # (hidden_layers) : tensorflow에서 각 layer를 이어주는 api

        ## xor_classifier.py에서의 코드에 의하면 self.num_output_nodes 의 값은 1
        output = tf.keras.layers.Dense(units=self.num_output_nodes,
                                       activation=tf.keras.activations.sigmoid,
                                       use_bias=True)(hidden_layers)

        # Model() 클래스 : 맵핑을 해서, 모델을 등록시킴
        self.logic_op_model = tf.keras.Model(inputs=input_layer, outputs=output)

        sgd = tf.keras.optimizers.SGD(learning_rate=0.1)
        self.logic_op_model.compile(optimizer=sgd, loss="mse")

    # fit() : 학습
    def fit(self, x, y, batch_size, epochs):
        self.logic_op_model.fit(x=x, y=y, batch_size=batch_size, epochs=epochs)

    # predict() : 예측
    def predict(self, x, batch_size):
        prediction = self.logic_op_model.predict(x=x, batch_size=batch_size)
        return prediction