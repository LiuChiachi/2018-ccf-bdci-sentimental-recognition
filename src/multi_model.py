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

def multi_model():  # conv 中的acativation???
    maxlen = 100
    num_filter = 128
    drop = 0.55
    inputs = Input(shape=(maxlen,))
    
    embed_layer = Embedding(len(word2index) + 1, EMBEDDING_DIM, weights=[embedding_matrix], 
                            input_length=maxlen, trainable=True)(inputs)
    embed_layer = SpatialDropout1D(drop)(embed_layer)
    """**************************************text_cnn*********************************************************"""
    conv1 = Conv1D(filters=num_filter, kernel_size=1, strides=1, padding='same')(embed_layer)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv1D(filters=num_filter, kernel_size=1, strides=1, padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1_pool = MaxPooling1D(pool_size=2, strides=None, padding='same')(conv1)
   
    conv2 = Conv1D(filters=num_filter, kernel_size=2, strides=1, padding='same')(embed_layer)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv1D(filters=num_filter, kernel_size=2, strides=1, padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2_pool = MaxPooling1D(pool_size=2, strides=None, padding='same')(conv2)
    
    conv3 = Conv1D(filters=num_filter, kernel_size=3, strides=1, padding='same')(embed_layer)
    conv3 = BatchNormalization()(conv3)
    conv1 = Activation('relu')(conv3)
    conv3 = Conv1D(filters=num_filter, kernel_size=3, strides=1, padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    conv3_pool = MaxPooling1D(pool_size=2, strides=None, padding='valid')(conv3)
    
    conv4 = Conv1D(filters=num_filter, kernel_size=4, strides=1, padding='same')(embed_layer)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv1D(filters=num_filter, kernel_size=4, strides=1, padding='same')(conv4)
    conv4 = BatchNormalization()(conv1)
    conv4 = Activation('relu')(conv4)
    conv4_pool = MaxPooling1D(pool_size=2, strides=None, padding='same')(conv4)
    
    conv5 = Conv1D(filters=num_filter, kernel_size=5, strides=1, padding='same')(embed_layer)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv1D(filters=num_filter, kernel_size=5, strides=1, padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    conv5_pool =MaxPooling1D(pool_size=2, strides=None, padding='valid')(conv5)
    
    conv_concat = Concatenate(axis=1)([conv1_pool, conv2_pool, conv3_pool, conv4_pool, conv5_pool])
    # conv_concat = Flatten()(conv_concat)
    conv_concat = Dropout(drop)(conv_concat)
    # conv_concat = Activation('linear')(conv_concat)
    conv_concat = Dense(300)(conv_concat)
    conv_bn = BatchNormalization()(conv_concat)
    conv_relu = Activation('relu')(conv_bn)
    conv_dropout = Dropout(drop)(conv_relu)
    outputs1 = Dense(30, activation='sigmoid', name='text_cnn')(conv_dropout)
 

    """***************************************bi_gru********************************************************"""
    gru_units= 64 # 100
 
 
    """x = LSTM(output_dim=100,activation='relu',inner_activation='relu', return_sequences=True)(x)"""
    x1 = Bidirectional(GRU(gru_units, activation='relu', dropout=dropout_p, recurrent_dropout=dropout_p, 
                          return_sequences=True))(embed_layer)
  
    x2 = Bidirectional(GRU(gru_units, activation='relu', dropout=dropout_p, recurrent_dropout=dropout_p, 
                      return_sequences=True))(embed_layer) 
    x3= Bidirectional(GRU(gru_units, activation='relu', dropout=dropout_p, recurrent_dropout=dropout_p, 
                      return_sequences=True))(x2)
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
    outputs2 = Dense(30, activation='sigmoid', name='bi_gru')(bn_dropout)
    
    """******************************************hi_bi_gru*****************************************************"""

    drop = 0.55# 0.55
    gru_units= 64 # 100
    maxlen = 100 
    """x = LSTM(output_dim=100,activation='relu',inner_activation='relu', return_sequences=True)(x)"""
    x1 = Bidirectional(GRU(gru_units, activation='relu', dropout=dropout_p, recurrent_dropout=dropout_p, 
                          return_sequences=True))(embed_layer)
  
    x2 = Bidirectional(GRU(gru_units, activation='relu', dropout=dropout_p, recurrent_dropout=dropout_p, 
                      return_sequences=True))(embed_layer) 
    x3= Bidirectional(GRU(gru_units, activation='relu', dropout=dropout_p, recurrent_dropout=dropout_p, 
                      return_sequences=True))(x2)
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
    outputs3 = Dense(30, activation='sigmoid', name='hi_bi_gru')(bn_dropout)

    """******************************************capsule*****************************************************"""
 

    x = Bidirectional(
        GRU(gru_len, activation='relu', dropout=dropout_p, recurrent_dropout=dropout_p, return_sequences=True))(
        embed_layer)
    # capsule是一个卷积网络
    capsule = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings,
                      share_weights=True)(x)
    # output_capsule = Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)))(capsule)
    capsule = Flatten()(capsule)
    capsule = Dropout(dropout_p)(capsule)
    outputs4 = Dense(30, activation='sigmoid', name='capsule')(capsule)
    
    outputs = Add()([outputs1, outputs2, outputs3, outputs4])
    model = Model(inputs=inputs, outputs=outputs)
    learning_rate = 0.001  # 0.0001

    adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-5)
    sgd = SGD(lr=learning_rate, momentum=0.0, decay=0.0, nesterov=False)
    rmsprop = RMSprop(lr=learning_rate, rho=0.9, epsilon=None, decay=0.0)

    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=[f1_score])
    return model


def multi_model_independent_embedding():  # conv 中的acativation???
    maxlen = 100
    num_filter = 128
    drop = 0.55
    inputs = Input(shape=(maxlen,))

    """**************************************text_cnn*********************************************************"""
    
    embed_layer1 = Embedding(len(word2index) + 1, EMBEDDING_DIM, weights=[embedding_matrix], 
                            input_length=maxlen, trainable=True)(inputs)

    embed_layer1= SpatialDropout1D(drop)(embed_layer1)

    conv1 = Conv1D(filters=num_filter, kernel_size=1, strides=1, padding='same',activation=None)(embed_layer1)
    # conv1 = Conv2D(filters=num_filter, kernel_size=(1,EMBEDDING_DIM), strides=(1, 1), padding='valid', activation='relu')(embed_layer)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv1D(filters=num_filter, kernel_size=1, strides=1, padding='same', activation=None)(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1_pool = MaxPooling1D(pool_size=2, strides=None, padding='same')(conv1)
    

    conv2 = Conv1D(filters=num_filter, kernel_size=2, strides=1, padding='same', activation=None)(embed_layer1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv1D(filters=num_filter, kernel_size=2, strides=1, padding='same', activation=None)(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2_pool = MaxPooling1D(pool_size=2, strides=None, padding='same')(conv2)
    
    conv3 = Conv1D(filters=num_filter, kernel_size=3, strides=1, padding='same', activation=None)(embed_layer1)
    conv3 = BatchNormalization()(conv3)
    conv1 = Activation('relu')(conv3)
    conv3 = Conv1D(filters=num_filter, kernel_size=3, strides=1, padding='same', activation=None)(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    conv3_pool = MaxPooling1D(pool_size=2, strides=None, padding='same')(conv3)
    
    conv4 = Conv1D(filters=num_filter, kernel_size=4, strides=1, padding='same', activation=None)(embed_layer1)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv1D(filters=num_filter, kernel_size=4, strides=1, padding='same', activation=None)(conv4)
    conv4 = BatchNormalization()(conv1)
    conv4 = Activation('relu')(conv4)
    conv4_pool = MaxPooling1D(pool_size=2, strides=None, padding='same')(conv4)
    
    conv5 = Conv1D(filters=num_filter, kernel_size=5, strides=1, padding='same', activation=None)(embed_layer1)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv1D(filters=num_filter, kernel_size=5, strides=1, padding='same', activation=None)(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    conv5_pool = MaxPooling1D(pool_size=2, strides=None, padding='same')(conv5)
    
    conv_concat = Concatenate(axis=1)([conv1_pool, conv2_pool, conv3_pool, conv4_pool, conv5_pool])
    
    
    avg_pool = GlobalAveragePooling1D()(conv_concat)
    max_pool = GlobalMaxPooling1D()(conv_concat)
    conv_concat = Concatenate()([avg_pool, max_pool])
    # conv_concat = Flatten()(conv_concat)
    conv_concat = Dropout(drop)(conv_concat)
    conv_concat = Dense(300)(conv_concat)
    conv_bn = BatchNormalization()(conv_concat)
    conv_relu = Activation('relu')(conv_bn)
    # conv_dropout = Dropout(drop)(conv_relu)
    outputs1 = Dense(30, activation='sigmoid', name='textCNN')(conv_relu)
 

    """***************************************bi_gru********************************************************"""
    gru_units= 64 # 100
 
    embed_layer2 = Embedding(len(word2index) + 1, EMBEDDING_DIM, weights=[embedding_matrix], 
                            input_length=maxlen, trainable=True)(inputs)
    """x = LSTM(output_dim=100,activation='relu',inner_activation='relu', return_sequences=True)(x)"""
    x1 = Bidirectional(GRU(gru_units, activation='relu', dropout=dropout_p, recurrent_dropout=dropout_p, 
                          return_sequences=True))(embed_layer2)
    x1 = attention_3d_block(x1)
    x2 = Bidirectional(GRU(gru_units, activation='relu', dropout=dropout_p, recurrent_dropout=dropout_p, 
                      return_sequences=True))(embed_layer2) 
    
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
    outputs2 = Dense(30, activation='sigmoid', name='bi_gru')(bn_dropout)
    
    """******************************************hi_bi_gru*****************************************************"""
    drop = 0.55
    embed_layer3= Embedding(len(word2index) + 1, EMBEDDING_DIM, weights=[embedding_matrix], 
                            input_length=maxlen, trainable=True)(inputs)
    x1 = Bidirectional(GRU(gru_units, activation='relu', dropout=dropout_p, recurrent_dropout=dropout_p, 
                          return_sequences=True))(embed_layer3)
    x1 = attention_3d_block(x1)
    x2 = Bidirectional(GRU(gru_units, activation='relu', dropout=dropout_p, recurrent_dropout=dropout_p, 
                      return_sequences=True))(embed_layer3)
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
    outputs3 = Dense(30, activation='sigmoid', name='hi_bi_gru')(bn_dropout)

    """******************************************capsule*****************************************************"""
    embed_layer4= Embedding(len(word2index) + 1, EMBEDDING_DIM, weights=[embedding_matrix], 
                            input_length=maxlen, trainable=True)(inputs)
    x = Bidirectional(GRU(gru_len, activation='relu', dropout=dropout_p, recurrent_dropout=dropout_p,
                         return_sequences=True))(embed_layer4)
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
    outputs4 = Dense(30, activation='sigmoid', name='capsule')(capsule)
    # now we get outputs1, outputs2, outputs3, outputs4
    # outputs = Add()([outputs1, outputs2, outputs3, outputs4])
    output_concat = Concatenate(axis=1)([outputs1, outputs2, outputs3, outputs4])
    outputs = Dropout(drop)(output_concat)
    outputs = Dense(30, activation='sigmoid')(outputs)

    model = Model(inputs=inputs, outputs=outputs)
    
    learning_rate = 0.00001
    adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
    sgd = SGD(lr=learning_rate, momentum=0.0, decay=0.0, nesterov=False)
    rmsprop = RMSprop(lr=learning_rate, rho=0.9, epsilon=None, decay=0.0)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=[f1_score])  # 先'adam'
    
    return model
