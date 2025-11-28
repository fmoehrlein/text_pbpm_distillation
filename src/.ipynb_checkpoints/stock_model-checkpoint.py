import tensorflow as tf
from sklearn.metrics import r2_score
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import LSTM, BatchNormalization, Dense, Dropout, Concatenate

class StockLSTMModel:
    def __init__(self, input_dim, window_size, output_dim=1, text_embedding_dim=None, text_fusion=False):
        self.window_size = window_size
        self.input_dim = input_dim
        self.text_embedding_dim = text_embedding_dim
        self.output_dim = output_dim
        self.model = self._build_model(text_embedding_dim=text_embedding_dim, text_fusion=text_fusion)
    
    def _build_model(self, text_fusion=False, text_embedding_dim=None):

        # Stock price input and LSTM branch
        stock_input = Input(shape=(self.window_size, self.input_dim), name="stock_input")
        x = LSTM(500, return_sequences=True)(stock_input)
        x = LSTM(450, dropout=0.1)(x)
        x = BatchNormalization()(x)

        if text_fusion:
            if text_embedding_dim is None:
                raise ValueError("text_embedding_dim must be specified if text_fusion=True")
            
            # Text embedding input and branch
            text_input = Input(shape=(text_embedding_dim,), name="text_input")
            t = Dense(128, activation="relu")(text_input)
            t = Dropout(0.1)(t)
            t = Dense(64, activation="relu")(t)

            # Fusion
            combined = Concatenate()([x, t])
            combined = Dense(64, activation="relu")(combined)
            combined = Dense(32, activation="relu")(combined)
            output = Dense(self.output_dim)(combined)

            model = Model(inputs=[stock_input, text_input], outputs=output)
        else:
            # No text fusion: just continue the stock branch
            x = Dense(64, activation="relu")(x)
            x = Dense(32, activation="relu")(x)
            output = Dense(self.output_dim)(x)

            model = Model(inputs=stock_input, outputs=output)

        model.compile(
            optimizer="adam",
            loss="mae",
            metrics=[
                tf.keras.metrics.MeanAbsoluteError(name="mae"),
                tf.keras.metrics.MeanAbsolutePercentageError(name="mape"),
                tf.keras.metrics.MeanSquaredError(name="mse"),
            ],
        )

        return model

    def fit(self, train_ds, val_ds, epochs=100, patience=10):
        # Adapt normalization layer to input data (optional, if used in model)
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )

        history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=[early_stop]
        )
        return history


    def evaluate(self, test_ds):
        preds = []
        targets = []

        for x_batch, y_batch in test_ds:
            y_pred = self.model.predict(x_batch, verbose=0)
            preds.extend(y_pred.flatten())
            targets.extend(y_batch.numpy().flatten())

        preds = tf.convert_to_tensor(preds)
        targets = tf.convert_to_tensor(targets)

        mae = tf.keras.metrics.mean_absolute_error(targets, preds).numpy()
        mape = tf.keras.metrics.mean_absolute_percentage_error(targets, preds).numpy()
        mse = tf.keras.metrics.mean_squared_error(targets, preds).numpy()
        r2 = r2_score(targets.numpy(), preds.numpy())

        return {
            "MAE": mae,
            "MAPE": mape,
            "MSE": mse,
            "R2": r2
        }


class StockLSTMModel_old:
    def __init__(self, input_dim, window_size, output_dim=1, text_embedding_dim=None, text_fusion=False):
        self.window_size = window_size
        self.input_dim = input_dim
        self.text_embedding_dim = text_embedding_dim
        self.output_dim = output_dim
        self.model = self._build_model(text_embedding_dim=text_embedding_dim, text_fusion=text_fusion)
    
    def _build_model(self, text_fusion=False, text_embedding_dim=None):

        # Stock price input and LSTM branch
        stock_input = Input(shape=(self.window_size, self.input_dim), name="stock_input")
        x = LSTM(500, return_sequences=True)(stock_input)
        x = LSTM(450, dropout=0.1)(x)
        x = BatchNormalization()(x)

        if text_fusion:
            if text_embedding_dim is None:
                raise ValueError("text_embedding_dim must be specified if text_fusion=True")
            
            # Text embedding input and branch
            text_input = Input(shape=(text_embedding_dim,), name="text_input")
            t = Dense(128, activation="relu")(text_input)
            t = Dropout(0.1)(t)
            t = Dense(64, activation="relu")(t)

            # Fusion
            combined = Concatenate()([x, t])
            combined = Dense(64, activation="relu")(combined)
            combined = Dense(32, activation="relu")(combined)
            output = Dense(self.output_dim)(combined)

            model = Model(inputs=[stock_input, text_input], outputs=output)
        else:
            # No text fusion: just continue the stock branch
            x = Dense(64, activation="relu")(x)
            x = Dense(32, activation="relu")(x)
            output = Dense(self.output_dim)(x)

            model = Model(inputs=stock_input, outputs=output)

        model.compile(
            optimizer="adam",
            loss="mae",
            metrics=[
                tf.keras.metrics.MeanAbsoluteError(name="mae"),
                tf.keras.metrics.MeanAbsolutePercentageError(name="mape"),
                tf.keras.metrics.MeanSquaredError(name="mse"),
            ],
        )

        return model

    def fit(self, train_ds, val_ds, epochs=100, patience=10):
        # Adapt normalization layer to input data (optional, if used in model)
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )

        history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=[early_stop]
        )
        return history


    def evaluate(self, test_ds):
        preds = []
        targets = []

        for x_batch, y_batch in test_ds:
            y_pred = self.model.predict(x_batch, verbose=0)
            preds.extend(y_pred.flatten())
            targets.extend(y_batch.numpy().flatten())

        preds = tf.convert_to_tensor(preds)
        targets = tf.convert_to_tensor(targets)

        mae = tf.keras.metrics.mean_absolute_error(targets, preds).numpy()
        mape = tf.keras.metrics.mean_absolute_percentage_error(targets, preds).numpy()
        mse = tf.keras.metrics.mean_squared_error(targets, preds).numpy()
        r2 = r2_score(targets.numpy(), preds.numpy())

        return {
            "MAE": mae,
            "MAPE": mape,
            "MSE": mse,
            "R2": r2
        }

