from keras import backend as K
import keras
import numpy as np
from keras.backend.tensorflow_backend import set_session
from keras.layers import Input
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Dropout, Activation, SpatialDropout1D
from keras.layers import Input, Bidirectional, RNN, Concatenate,  Flatten
from keras.layers.pooling import GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers.recurrent import LSTM, GRU
from keras.optimizers import Adam, RMSprop, SGD
from keras.models import Model
from keras.layers import GlobalMaxPooling1D, MaxPooling1D, Concatenate, Permute, Reshape, merge, BatchNormalization
from capsule import *


def attention_3d_block(inputs):
    # https://github.com/keras-team/keras/issues/1472
    # https://github.com/philipperemy/keras-attention-mechanism
    TIME_STEPS = 20
    INPUT_DIM = 2
    # if True, the attention vector is shared across the input_dimensions where the attention is applied.
    SINGLE_ATTENTION_VECTOR = False
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    # print(a.shape, input_dim, TIME_STEPS)
    # a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(TIME_STEPS, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    # a_probs = Permute((2, 1), name='attention_vec')(a)
    a_probs = Permute((2, 1))(a)
    # print(inputs.shape, a_probs.shape)
    # output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    output_attention_mul = Concatenate(axis=1)([inputs, a_probs])
    return output_attention_mul

    """
    # https://stackoverflow.com/questions/42918446/how-to-add-an-attention-mechanism-in-keras
    activations = LSTM(units, return_sequences=True)(embedded)

    # compute importance for each step
    attention = Dense(1, activation='tanh')(activations)
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(units)(attention)
    attention = Permute([2, 1])(attention)

    sent_representation = merge([activations, attention], mode='mul')
    """
def model_attention_applied_after_lstm():
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    lstm_units = 32
    lstm_out = LSTM(lstm_units, return_sequences=True)(inputs)
    
    attention_mul = attention_3d_block(lstm_out)
    attention_mul = Flatten()(attention_mul)
    output = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(input=[inputs], output=output)
    return model

def f1_score(y_true, y_pred):
    """
    https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras
    """
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def get_bi_gru_model():
    # 3个串联  0.10
    drop = 0.60# 0.55
    # dropout_p = 0.5
    learning_rate = 0.001  # 0.0001
    gru_units= 128  # 100
    maxlen = 100
    inputs = Input(shape=(maxlen,))
    embed_layer = Embedding(len(word2index) + 1, EMBEDDING_DIM, weights=[embedding_matrix], 
                            input_length=maxlen, trainable=True)(inputs)
    embed_layer = SpatialDropout1D(drop)(embed_layer)
    """x = LSTM(output_dim=100,activation='relu',inner_activation='relu', return_sequences=True)(x)"""
    x1 = Bidirectional(GRU(gru_units, activation='relu', dropout=dropout_p, recurrent_dropout=dropout_p, 
                          return_sequences=True))(embed_layer)
    x1 = attention_3d_block(x1)
    x2 = Bidirectional(GRU(gru_units, activation='relu', dropout=dropout_p, recurrent_dropout=dropout_p, 
                      return_sequences=True))(embed_layer) 
    
    x2 = attention_3d_block(x2)
    x3 = Concatenate(axis=1)([x1, x2])
    avg_pool = GlobalAveragePooling1D()(x3)
    max_pool = GlobalMaxPooling1D()(x3)
    # print(avg_pool.shape, max_pool.shape)
    x5 = Concatenate(axis=1)([avg_pool, max_pool])
    
    fc = Dense(300)(x5)
    bn = BatchNormalization()(fc)
    bn = Activation('relu')(bn)
    bn_dropout = Dropout(drop)(bn)
    # bn_dropout = Flatten()(bn_dropout)
    outputs = Dense(30, activation='sigmoid')(bn_dropout)
    # print(outputs.shape)
    model = Model(inputs=inputs, outputs=outputs)

    adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    sgd = SGD(lr=learning_rate, momentum=0.0, decay=0.0, nesterov=False)
    rmsprop = RMSprop(lr=learning_rate, rho=0.9, epsilon=None, decay=0.0)
    nadam = keras.optimizers.Nadam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=[f1_score])
    return model



def get_text_cnn_model():
    # 加残差或者是attention
    # 3个串联  0.10
    drop = 0.60# 0.55
    # dropout_p = 0.5
    learning_rate = 0.005  # 0.0001
    Dim_capsule = 30 # 128
    gru_units= Dim_capsule>>1
    inputs = Input(shape=(maxlen,))
    embed_layer = Embedding(len(word2index) + 1, EMBEDDING_DIM, weights=[embedding_matrix], 
                            input_length=maxlen, trainable=False)(inputs)
    embed_layer = SpatialDropout1D(drop)(embed_layer)
     
 
    capsule1 = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings, # kernel_size=(3, 1),
                      share_weights=True)(embed_layer)
    bn1 = BatchNormalization()(capsule1)
                                    

    x1 = Bidirectional(GRU(gru_units, activation='relu', dropout=dropout_p, recurrent_dropout=dropout_p, 
                          return_sequences=True))(embed_layer)
    print(bn1.shape, x1.shape)                                      
    concat = Concatenate(axis=1)([bn1, x1])
    bn = Flatten()(concat)

    fc = Dense(300)(bn)
    bn = BatchNormalization()(fc)
    bn = Activation('relu')(bn)
    bn_dropout = Dropout(drop)(bn)
    outputs = Dense(30, activation="sigmoid")(bn_dropout)
    model = Model(inputs=inputs, outputs=outputs)

    # model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
    # validation_data=(p_X_test, p_y_test))
    # score, acc = model.evaluate(p_X_test, p_y_test, batch_size=batch_size)
    
    
    
  
def get_bi_lstm_model():
    # 3个串联  0.10
    drop = 0.60# 0.55

    gru_units= 128  # 100
    maxlen = 100
    inputs = Input(shape=(maxlen,))
    embed_layer = Embedding(len(word2index) + 1, EMBEDDING_DIM, weights=[embedding_matrix], 
                            input_length=maxlen, trainable=True)(inputs)
    embed_layer = SpatialDropout1D(drop)(embed_layer)
    x1 = Bidirectional(LSTM(gru_units, activation='relu', dropout=drop, recurrent_dropout=drop, 
                            return_sequences=True))(embed_layer)
    x1 = attention_3d_block(x1)
    x2 = Bidirectional(LSTM(gru_units, activation='relu', dropout=drop, recurrent_dropout=drop, 
                            return_sequences=True))(embed_layer)
    x2 = attention_3d_block(x2)
    x3 = Concatenate(axis=1)([x1, x2])
    avg_pool = GlobalAveragePooling1D()(x3)
    max_pool = GlobalMaxPooling1D()(x3)
    # print(avg_pool.shape, max_pool.shape)
    x5 = Concatenate(axis=1)([avg_pool, max_pool])
    print(x5.shape)
    fc = Dense(300)(x5)
    bn = BatchNormalization()(fc)
    bn = Activation('relu')(bn)
    bn_dropout = Dropout(drop)(bn)
    # bn_dropout = Flatten()(bn_dropout)
    outputs = Dense(30, activation='sigmoid')(bn_dropout)
    # print(outputs.shape)
    model = Model(inputs=inputs, outputs=outputs)

    learning_rate = 0.01  # 0.0001
    adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
    sgd = SGD(lr=learning_rate, momentum=0.0, decay=0.0, nesterov=False)
    rmsprop = RMSprop(lr=learning_rate, rho=0.9, epsilon=None, decay=0.0)
    nadam = keras.optimizers.Nadam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=[f1_score])
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1_score])
    # model.fit(X_train, y_train, batch_size=16, nb_epoch=1, validation_split=0.1, shuffle=True)
              # validation_data=(p_X_test, p_y_test))
    # score, acc = model.evaluate(p_X_test, p_y_test, batch_size=batch_size)
    return model



def get_capsule_model():
    maxlen = 100
    learning_rate = 0.00001
    input1 = Input(shape=(maxlen,))
    embed_layer = Embedding(len(word2index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=maxlen,
                            trainable=True)(input1)
    embed_layer = SpatialDropout1D(rate_drop_dense)(embed_layer)

    x = Bidirectional(
        GRU(gru_len, activation='relu', dropout=dropout_p, recurrent_dropout=dropout_p, return_sequences=True))(
        embed_layer)
    # capsule是一个卷积网络
    capsule = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings,
                      share_weights=True)(x)
    # output_capsule = Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)))(capsule)
    avg_pool = GlobalAveragePooling1D()(capsule)
    max_pool = GlobalMaxPooling1D()(capsule)
    capsule_concat  = Concatenate()([avg_pool, max_pool])
    
    capsule = Flatten()(capsule)
    caspule = BatchNormalization()(capsule_concat)
    capsule = Activation('relu')(capsule)
    
    capsule = Dropout(dropout_p)(capsule)
    output = Dense(30, activation='sigmoid')(capsule)
    model = Model(inputs=input1, outputs=output)
    # print(capsule.shape, output.shape)

    adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
    sgd = SGD(lr=learning_rate, momentum=0.0, decay=0.0, nesterov=False)
    rmsprop = RMSprop(lr=learning_rate, rho=0.9, epsilon=None, decay=0.0)
    nadam = keras.optimizers.Nadam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=[f1_score])
    return model


def get_hi_bi_gru_model():
    # 3个串联  0.10
    drop = 0.7# 0.55
    # dropout_p = 0.5
    learning_rate = 0.0001  # 0.0001
    gru_units= 128 # 100
    maxlen = 100
    inputs = Input(shape=(maxlen,))
    embed_layer = Embedding(len(word2index) + 1, EMBEDDING_DIM, weights=[embedding_matrix], 
                            input_length=maxlen, trainable=True)(inputs)
    embed_layer = SpatialDropout1D(drop)(embed_layer)
    """x = LSTM(output_dim=100,activation='relu',inner_activation='relu', return_sequences=True)(x)"""
    x1 = Bidirectional(GRU(gru_units, activation='relu', dropout=dropout_p, recurrent_dropout=dropout_p, 
                          return_sequences=True))(embed_layer)
    x1 = attention_3d_block(x1)
    x2 = Bidirectional(GRU(gru_units, activation='relu', dropout=dropout_p, recurrent_dropout=dropout_p, 
                      return_sequences=True))(embed_layer)
    x2 = attention_3d_block(x2)
    x3= Bidirectional(GRU(gru_units, activation='relu', dropout=dropout_p, recurrent_dropout=dropout_p, 
                      return_sequences=True))(x2)
    x3 = attention_3d_block(x3)
    """
    x3 = Bidirectional(GRU(gru_units, activation='relu', dropout=dropout_p, recurrent_dropout=dropout_p, 
                       return_sequences=True))(x)
    x4 = Concatenate(axis=1)([x1, x2, x3])
    # x = Dense(200, activation='relu')(x)
    x4 = Dropout(drop)(x4)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    x5 = Concatenate(axis=1)([avg_pool, max_pool])
    """
    x4 = Concatenate(axis=1)([x1, x3])
    avg_pool = GlobalAveragePooling1D()(x4)
    max_pool = GlobalMaxPooling1D()(x4)
    x5 = Concatenate(axis=1)([avg_pool, max_pool])
    fc = Dense(300)(x5)
    bn = BatchNormalization()(fc)
    bn = Activation('relu')(bn)
    bn_dropout = Dropout(drop)(bn)
    # bn_dropout = Flatten()(bn_dropout)
    outputs = Dense(30, activation='sigmoid')(bn_dropout)
    print(outputs.shape)
    model = Model(inputs=inputs, outputs=outputs)

    adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
    sgd = SGD(lr=learning_rate, momentum=0.0, decay=0.0, nesterov=False)
    rmsprop = RMSprop(lr=learning_rate, rho=0.9, epsilon=None, decay=0.0)

    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=[f1_score])
    # model.fit(X_train, y_train, batch_size=16, nb_epoch=1, validation_split=0.1, shuffle=True)
              # validation_data=(p_X_test, p_y_test))
    # score, acc = model.evaluate(p_X_test, p_y_test, batch_size=batch_size)
    #  model.load_weights(fname, by_name=True)
    return model


def get_hi_text_cnn_model():
    drop = 0.60 
    # dropout_p = 0.5
    learning_rate = 0.005  # 0.0001
    inputs = Input(shape=(maxlen,))
    embed_layer = Embedding(len(word2index) + 1, EMBEDDING_DIM, weights=[embedding_matrix], 
                            input_length=maxlen, trainable=False)(inputs)
    embed_layer = SpatialDropout1D(drop)(embed_layer)
     
    # 第一条支路：
    capsule1 = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings, # kernel_size=(3, 1),
                      share_weights=True)(embed_layer)
    bn1 = BatchNormalization()(capsule1)
                                          
    # 第二条支路：   
    capsule2 = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings, # kernel_size=(3, 1),
                      share_weights=True)(embed_layer)
    bn2 = BatchNormalization()(capsule2)
           
    capsule3 = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings, # kernel_size=(3, 1),
                      share_weights=True)(bn2)  
    bn3 = BatchNormalization()(capsule3)
   
    # concat, fc+bn+relu
    bn = Concatenate(axis=1)([bn1, bn3])
    bn = Flatten()(bn)

    fc = Dense(300)(bn)
    bn = BatchNormalization()(fc)
    bn = Activation('relu')(bn)
    bn_dropout = Dropout(drop)(bn)
    outputs = Dense(30, activation="sigmoid")(bn_dropout)
    print(outputs.shape)
    model = Model(inputs=inputs, outputs=outputs)

    # model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
    # validation_data=(p_X_test, p_y_test))
    # score, acc = model.evaluate(p_X_test, p_y_test, batch_size=batch_size)
    
    
    
def fast_text_model():
    # 3个串联  0.10
    drop = 0.7# 0.55
    # dropout_p = 0.5
    learning_rate = 0.001  # 0.0001
    gru_units= 128  # 100
    maxlen = 100
    inputs = Input(shape=(maxlen,))
    embed_layer = Embedding(len(word2index) + 1, EMBEDDING_DIM, weights=[embedding_matrix], 
                            input_length=maxlen, trainable=True)(inputs)
    embed_layer = SpatialDropout1D(drop)(embed_layer)
    """x = LSTM(output_dim=100,activation='relu',inner_activation='relu', return_sequences=True)(x)"""
    pool = GlobalAveragePooling1D()(embed_layer)
    print(pool.shape)
    fc = Dense(400, activation='relu')(pool)
    
    bn = BatchNormalization()(fc)
    bn_relu = Activation('relu')(bn)
    bn_dropout = Dropout(drop)(bn_relu)
    
    outputs = Dense(30, activation='sigmoid')(bn_dropout)
    # print(outputs.shape)
    model = Model(inputs=inputs, outputs=outputs)

    adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6)
    sgd = SGD(lr=learning_rate, momentum=0.0, decay=0.0, nesterov=False)
    rmsprop = RMSprop(lr=learning_rate, rho=0.9, epsilon=None, decay=0.0)
    nadam = Nadam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1_score])
    return model

    
    
    
def RCNN_model():
    drop = 0.5
    learning_rate = 0.005  # 0.0001
    maxlen = 100
    num_filter = 128
    gru_units = 150
    inputs = Input(shape=(maxlen,))
    embed_layer = Embedding(len(word2index) + 1, EMBEDDING_DIM, weights=[embedding_matrix], 
                            input_length=maxlen, trainable=True)(inputs)
    embed_layer = SpatialDropout1D(drop)(embed_layer)
    
    # 1
    bi_lstm_1 = Bidirectional(GRU(gru_units, activation='relu', dropout=dropout_p, recurrent_dropout=dropout_p, 
                          return_sequences=True))(embed_layer)
    # bi_lstm_1 = attention_3d_block(bi_lstm_1)
    bi_lstm_2 = Bidirectional(GRU(gru_units, activation='relu', dropout=dropout_p, recurrent_dropout=dropout_p, 
                          return_sequences=True))(bi_lstm_1)
    # bi_lstm_2 = attention_3d_block(bi_lstm_2)
    # 2
    concat = Concatenate(axis=1)([bi_lstm_2, embed_layer])
    # pool = Lambda(lambda x: tf.reshape(tf.nn.top_k(tf.transpose(x,[0,2,1]),k=2)[0],shape=[-1,6]))(concat)
    # pool = KMaxPooling(k=concat.shape[2])(concat)
    # pool = kmaxpooling.call(concat)
    pool = MaxPooling1D(pool_size=2, strides=2, padding='valid')(concat)

    # print(concat.shape, pool.shape)  # (?, ?, 300) (?, ?, 300) 128
    # pool = Reshape((-1, 300))(pool)
    print(concat.shape, pool.shape)
    conv = Conv1D(filters=num_filter, kernel_size=2, strides=1, padding='same', activation='relu')(pool)
    # conv_flatten = Flatten()(pool)
    
    avg_pool = GlobalAveragePooling1D()(conv)
    max_pool = GlobalMaxPooling1D()(conv)
    conv_concat  = Concatenate()([avg_pool, max_pool])
    
    conv_bn = BatchNormalization()(conv_concat)
    conv_relu = Activation('relu')(conv_bn)
    conv_drop = Dropout(drop)(conv_relu)
    outputs = Dense(30, activation="sigmoid")(conv_drop)
    
    # print(inputs.shape, outputs.shape)
    model = Model(inputs=inputs, outputs=outputs)

    # model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1_score])
    return model



def get_text_inception():
    maxlen = 100

    num_filter = 64
    drop = 0.6
    inputs = Input(shape=(maxlen,))
    embed_layer = Embedding(len(word2index)+1, EMBEDDING_DIM, weights=[embedding_matrix], 
                                input_length=maxlen, trainable=True)(inputs)
    embed_layer = SpatialDropout1D(drop)(embed_layer)
    # 1
    conv1 = Conv1D(filters=num_filter, kernel_size=1, strides=1, padding='same', activation=None)(embed_layer)
    conv1 = Dropout(drop)(conv1)
    # 2
    conv1_1 = Conv1D(filters=num_filter, kernel_size=1, strides=1, padding='same', activation=None)(embed_layer)
    conv1_bn = BatchNormalization()(conv1_1)
    conv1_relu = Activation('relu')(conv1_bn)
    conv1_relu = Dropout(drop)(conv1_relu)
    conv1_3 = Conv1D(filters=num_filter, kernel_size=3, strides=1, padding='same', activation=None)(conv1_relu)
    # conv_3 = Dropout(drop)(conv1_3)
    # 3
    conv3_1 = Conv1D(filters=num_filter, kernel_size=3, strides=1, padding='same', activation=None)(embed_layer)
    conv3_bn = BatchNormalization()(conv3_1)
    conv3_relu = Activation('relu')(conv3_bn)
    conv3_relu = Dropout(drop)(conv3_relu)
    conv3_5 = Conv1D(filters=num_filter, kernel_size=5, strides=1, padding='same', activation=None)(conv3_relu)
    # conv3_5 = Dropout(drop)(conv3_5)
    # 4
    conv3 = Conv1D(filters=num_filter, kernel_size=3, strides=1, padding='same', activation=None)(embed_layer)
    # conv3 = Dropout(drop)(conv3)
    # concat
    conv_concat = Concatenate(axis=1)([conv1, conv1_3, conv3_5, conv3])
    
    avg_pool = GlobalAveragePooling1D()(conv_concat)
    max_pool = GlobalMaxPooling1D()(conv_concat)
    conv_concat  = Concatenate()([avg_pool, max_pool])
    
    # conv_flatten = Flatten()(conv_concat)
    conv_drop = Dropout(drop)(conv_concat)
    conv_bn = BatchNormalization()(conv_drop)
    conv_relu = Activation('relu')(conv_bn)
    conv_dropout = Dropout(drop)(conv_relu)
    outputs = Dense(30, activation='sigmoid')(conv_dropout)
    model = Model(inputs=inputs, outputs=outputs)
    
    learning_rate = 0.0001
    adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
    sgd = SGD(lr=learning_rate, momentum=0.0, decay=0.0, nesterov=False)
    rmsprop = RMSprop(lr=learning_rate, rho=0.9, epsilon=None, decay=0.0)
    
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1_score])
    return model