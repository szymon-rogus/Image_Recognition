from tensorflow.keras import layers, models, optimizers


class CfModel:
    def __init__(self):
        self.model = models.Sequential()

    def create_architecture(self):
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(None, None, 3), padding='same'))
        self.model.add(layers.experimental.preprocessing.Rescaling(scale=1. / 255, offset=0.0))
        self.create_cnn_block(20, 0.1)
        self.create_cnn_block(40, 0.2)
        self.create_cnn_block(80, 0.3)
        self.create_cnn_gap_block(160, 0.4)
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(10, activation='softmax'))

    def create_cnn_block(self, filters, dropout_rate):
        self.model.add(layers.Conv2D(filters, (3, 3), activation='relu', padding='same'))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Conv2D(filters, (3, 3), activation='relu', padding='same'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Dropout(dropout_rate))

    def create_cnn_gap_block(self, filters, dropout_rate):
        self.model.add(layers.Conv2D(filters, (3, 3), activation='relu', padding='same'))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Conv2D(filters, (3, 3), activation='relu', padding='same'))
        self.model.add(layers.GlobalAveragePooling2D())
        self.model.add(layers.Dropout(dropout_rate))

    def get_summary(self):
        return self.model.summary()

    def save_model(self, filename):
        self.model.save(filename)

    def load_model(self, filename):
        self.model = models.load_model(filename)

    def compile(self):
        opt = optimizers.SGD(learning_rate=0.001, momentum=0.9)
        self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    def fit(self, x_train, y_train_vector, x_test, y_test_vector):
        return self.model.fit(x_train, y_train_vector, epochs=10, batch_size=64, validation_data=(x_test, y_test_vector))

    def predict(self, dataset):
        return self.model.predict(dataset)
