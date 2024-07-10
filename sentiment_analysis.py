import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout

#Load IMDB dataset
imdb = tf.keras.datasets.imdb
(x_train, y_train), (x_test, y_test) = imdb.load_data()

#Generators to convert data into a tensorflow Dataset object
def train_generator():
    for x, y in zip(x_train, y_train):
        yield x, y

def test_generator():
    for x, y in zip(x_test, y_test):
        yield x, y

#Construct dataset from generators, batch and pad data, Shuffle training set
train_ds = tf.data.Dataset.from_generator(
    train_generator, 
    output_signature=(
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )
).shuffle(25000).padded_batch(16, padded_shapes=([None], []))

test_ds = tf.data.Dataset.from_generator(
    test_generator, 
    output_signature=(
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )
).padded_batch(16, padded_shapes=([None], []))

#Get vocab size and set embedding dimensions
vocab_size = len(imdb.get_word_index())+1
embedding_dim = 16

class SentimentAnalysis(Model):
    #Instantiate network layers
    def __init__(self):
        super().__init__()
        self.embed = Embedding(vocab_size, embedding_dim, mask_zero=True)
        self.lstm1 = LSTM(embedding_dim, return_sequences=True, kernel_regularizer= tf.keras.regularizers.l2(0.01))
        self.drop1 = Dropout(0.2)
        self.lstm2 = LSTM(embedding_dim)
        self.drop2 = Dropout(0.2)
        self.d1 = Dense(8, activation="relu")
        self.d2 = Dense(1, activation="sigmoid")

    #Feed input into the network
    def call(self, x):
        x = self.embed(x)
        x = self.lstm1(x)
        x = self.drop1(x)
        x = self.lstm2(x)
        x = self.drop2(x)
        x = self.d1(x)
        return self.d2(x)
    
#Create model
model = SentimentAnalysis()

#Declare loss and optimizer
loss_obj = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam()

#Declare metrics
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')

#Processes one batch of training data: updates weights according to gradient
def train_step(texts, labels):
    with tf.GradientTape() as tape:
        predictions = model(texts, training=True)
        loss = loss_obj(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

#Processes one batch of testing data: measures accuracy and loss
def test_step(texts, labels):
    predictions = model(texts, training=False)
    loss = loss_obj(labels, predictions)

    test_loss(loss)
    test_accuracy(labels, predictions)

EPOCHS = 5

#For every epoch
for epoch in range(EPOCHS):
    #Reset metrics
    train_loss.reset_state()
    train_accuracy.reset_state()
    test_loss.reset_state()
    test_accuracy.reset_state()

    #Train model on each batch
    for texts, labels in train_ds:
        train_step(texts, labels)
    
    #Test model on each batch
    for texts, labels in test_ds:
        test_step(texts, labels)

    #Print relevant info
    print(
        f'Epoch {epoch + 1}, '
        f'Loss: {train_loss.result():0.2f}, ' #Training loss
        f'Accuracy: {train_accuracy.result() * 100:0.2f}, ' #Training accuracy
        f'Test Loss: {test_loss.result():0.2f}, ' #Testing loss
        f'Test Accuracy: {test_accuracy.result() * 100:0.2f}' #Testing accuracy
    )