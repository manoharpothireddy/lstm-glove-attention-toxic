import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="toxicity")
class AttentionPooling(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.W = None
        self.b = None
        self.u = None

    def build(self, input_shape):
        features = int(input_shape[-1])
        self.W = self.add_weight(
            shape=(features, features),
            initializer="glorot_uniform",
            trainable=True,
            name="W",
        )
        self.b = self.add_weight(
            shape=(features,),
            initializer="zeros",
            trainable=True,
            name="b",
        )
        self.u = self.add_weight(
            shape=(features,),
            initializer="glorot_uniform",
            trainable=True,
            name="u",
        )
        super().build(input_shape)

    def call(self, x, mask=None):
        uit = tf.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        ait = tf.tensordot(uit, self.u, axes=1)

        if mask is not None:
            mask = tf.cast(mask, tf.float32)
            ait = ait + (1.0 - mask) * (-1e9)

        a = tf.nn.softmax(ait, axis=1)
        a = tf.expand_dims(a, axis=-1)
        return tf.reduce_sum(x * a, axis=1)

    def compute_mask(self, inputs, mask=None):
        return None
