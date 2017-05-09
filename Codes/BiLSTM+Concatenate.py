import numpy as np
import pandas as pd
import numpy as np
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import Bidirectional
from keras.layers import Dense, Input, LSTM, Embedding
from keras.models import Model
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint

print("Loading Data...")
with open('dictionary_116661_300.pickle', 'rb') as f:
    dictionary = pickle.load(f)
with open('word_embedding_116661_300.pickle', 'rb') as f:
    word_embed = pickle.load(f)
with open('sent1_index.pickle', 'rb') as f:
    sent1_index = pickle.load(f)
with open('sent2_index.pickle', 'rb') as f:
    sent2_index = pickle.load(f)
with open('sent1_index_test.pickle', 'rb') as f:
    sent1_index_test = pickle.load(f)
with open('sent2_index_test.pickle', 'rb') as f:
    sent2_index_test = pickle.load(f)

train = pd.read_csv('train.csv')
labels = np.array(train['is_duplicate'].tolist())

print(len(sent1_index))
print(len(sent2_index))
print(len(sent1_index_test))
print(len(sent2_index_test))

print("Setting Parameters...")
MAX_SEQUENCE_LENGTH = 30
MAX_NB_WORDS = word_embed.shape[0]
EMBEDDING_DIM = word_embed.shape[1]

re_weight = True

if re_weight:
    class_weight = {0: 1.309028344, 1: 0.472001959}
else:
    class_weight = None

print("Preparing Train Data...")
data_1 = pad_sequences(sent1_index, maxlen=MAX_SEQUENCE_LENGTH)
data_2 = pad_sequences(sent2_index, maxlen=MAX_SEQUENCE_LENGTH)

print("Preparing Test Data...")
test_data_1 = pad_sequences(sent1_index_test, maxlen=MAX_SEQUENCE_LENGTH)
test_data_2 = pad_sequences(sent2_index_test, maxlen=MAX_SEQUENCE_LENGTH)

VALIDATION_SPLIT=0.66
print("Separating Train and Validation Data...")
np.random.seed(368)
perm = np.random.permutation(len(sent1_index))
idx_train = perm[:int(len(sent1_index)*(1-VALIDATION_SPLIT))]
idx_val = perm[int(len(sent1_index)*(1-VALIDATION_SPLIT)):]

data_1_train = np.vstack((data_1[idx_train], data_2[idx_train]))
data_2_train = np.vstack((data_2[idx_train], data_1[idx_train]))
labels_train = np.concatenate((labels[idx_train], labels[idx_train]))

data_1_val = np.vstack((data_1[idx_val], data_2[idx_val]))
data_2_val = np.vstack((data_2[idx_val], data_1[idx_val]))
labels_val = np.concatenate((labels[idx_val], labels[idx_val]))

weight_val = np.ones(len(labels_val))
if re_weight:
    weight_val *= 0.472001959
    weight_val[labels_val==0] = 1.309028344

num_lstm = np.random.randint(100, 175)
rate_drop_lstm = 0.15 + np.random.rand() * 0.25
#rate_drop_dense = 0.15 + np.random.rand() * 0.25
STAMP = 'lstm_%d_%.2f'%(num_lstm, rate_drop_lstm)
print(STAMP)
embedding_layer = Embedding(MAX_NB_WORDS,
                            EMBEDDING_DIM,
                            weights=[word_embed], mask_zero=True,
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)
Bilstm_layer = Bidirectional(LSTM(num_lstm ,dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm))
#lstm_layer = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)

sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences_1 = embedding_layer(sequence_1_input)
x1 = Bilstm_layer(embedded_sequences_1)
#x1 = lstm_layer(embedded_sequences_1)
sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences_2 = embedding_layer(sequence_2_input)
y1 = Bilstm_layer(embedded_sequences_2)
#y1 = lstm_layer(embedded_sequences_2)

merged = concatenate([x1, y1])
merged = BatchNormalization()(merged)
merged = Dense(200, activation='relu')(merged)
merged = BatchNormalization()(merged)
merged = Dense(200, activation='relu')(merged)
merged = BatchNormalization()(merged)
merged = Dense(200, activation='relu')(merged)
merged = BatchNormalization()(merged)
merged = Dense(200, activation='relu')(merged)
merged = BatchNormalization()(merged)
preds = Dense(1, activation='sigmoid')(merged)

print("Training the Model...")
model = Model(inputs=[sequence_1_input, sequence_2_input], \
        outputs=preds)
model.compile(loss='binary_crossentropy',
        optimizer='nadam',
        metrics=['acc'])

print(STAMP)

early_stopping =EarlyStopping(monitor='val_loss', patience=3)
bst_model_path = STAMP + '.h5'
model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

hist = model.fit([data_1_train, data_2_train], labels_train, \
        validation_data=([data_1_val, data_2_val], labels_val, weight_val), \
        epochs=200, batch_size=1024, shuffle=True, \
        class_weight=class_weight, callbacks=[early_stopping, model_checkpoint])

model.load_weights(bst_model_path)
bst_val_score = min(hist.history['val_loss'])
print(bst_val_score)

preds = model.predict([test_data_1, test_data_2], batch_size=2048, verbose=1)
preds += model.predict([test_data_2, test_data_1], batch_size=2048, verbose=1)
preds /= 2

train_preds = model.predict([data_1, data_2], batch_size=2048, verbose=1)
train_preds += model.predict([data_2, data_1], batch_size=2048, verbose=1)
train_preds /= 2

train_name =  'train_pred_' + STAMP + '.csv'
sub = pd.DataFrame()
sub['id'] = train['id']
sub['is_duplicate_1'] = train_preds
sub.to_csv(train_name, index=False)
print(train_name)
test_name = 'test_' + STAMP + '.csv'
sub = pd.DataFrame()
test = pd.read_csv('test.csv')
sub['test_id'] = test['test_id']
del train, test
sub['is_duplicate'] = preds
sub.to_csv(test_name, index=False)
print(test_name)












