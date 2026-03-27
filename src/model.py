import tensorflow as tf

from .attention import AttentionPooling


def build_model(
    *,
    vocab_size: int,
    embedding_dim: int,
    max_len: int,
    embedding_matrix,
    num_labels: int,
    lstm_units: int,
    dense_units: int,
    dropout: float,
    spatial_dropout: float,
    learning_rate: float,
):
    inputs = tf.keras.Input(shape=(max_len,), dtype=tf.int32)

    x = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[embedding_matrix],
        trainable=False,
        mask_zero=True,
        name="glove_embedding",
    )(inputs)

    x = tf.keras.layers.SpatialDropout1D(spatial_dropout)(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(lstm_units, return_sequences=True),
        name="bilstm",
    )(x)

    att = AttentionPooling(name="attention_pool")(x)
    mx = tf.keras.layers.GlobalMaxPooling1D(name="max_pool")(x)
    av = tf.keras.layers.GlobalAveragePooling1D(name="avg_pool")(x)

    fused = tf.keras.layers.Concatenate(name="fusion")([att, mx, av])
    fused = tf.keras.layers.Dense(dense_units, activation="relu")(fused)
    fused = tf.keras.layers.Dropout(dropout)(fused)

    outputs = tf.keras.layers.Dense(num_labels, activation="sigmoid", name="labels")(fused)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.AUC(multi_label=True, num_labels=num_labels, name="auc")],
    )
    return model
