import tensorflow as tf
from MLP import MLP

def xor_classifier_example ():
    input_data = tf.constant([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    input_data = tf.cast(input_data, tf.float32)

    xor_labels = tf.constant([0.0, 1.0, 1.0, 0.0])
    xor_labels = tf.cast(xor_labels, tf.float32)

    batch_size = 1
    epochs = 1500

    mlp_classifier = MLP(hidden_layer_conf=[4], num_output_nodes=1)
    mlp_classifier.build_model()
    mlp_classifier.fit(x=input_data, y=xor_labels, batch_size=batch_size, epochs=epochs)

    ######## MLP XOR prediciton
    prediction = mlp_classifier.predict(x=input_data, batch_size=batch_size)
    input_and_result = zip(input_data, prediction)
    print("====== MLP XOR classifier result =====")
    for x, y in input_and_result:
        if y > 0.5:
            print("%d XOR %d => %.2f => 1" % (x[0], x[1], y))
        else:
            print("%d XOR %d => %.2f => 0" % (x[0], x[1], y))


# Entry point
if __name__ == '__main__':
    xor_classifier_example()