from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import pandas as pd
import jieba
import os
from keras.preprocessing import sequence
import pickle as pk
from keras import backend as K

"""
找到embedding词典
分词、移除停用词
查找字典
padding，拼接
"""

EMBEDDING_DIM = 300
MATRIX_DIR = '../data/output/matrixes'
train_file = '../data/input/train_2.csv'
test_file = '../data/input/test_public_2.csv'

import  os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def load_stop_words():
    # load stopwords
    with open('../conf/hlp_stop_words.txt', encoding='utf-8') as f:
        stop_words = set([l.strip() for l in f])

    return stop_words


def load_embeddings_index():
    # load Glove Vectors
    embeddings_index = {}

    embfile = '../conf/sgns.baidubaike.bigram-char'
    with open(embfile, encoding='utf-8') as f:
        for i, line in enumerate(f):
            values = line.split()
            words = values[:-EMBEDDING_DIM]
            word = ''.join(words)
            try:
                coefs = np.asarray(values[-EMBEDDING_DIM:], dtype='float32')
                embeddings_index[word] = coefs
            except:
                pass
    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index


def raw_file_2_matrix(train_file, test_file, stop_words, test_size=0):
    train_df = pd.read_csv(train_file, encoding='utf-8')
    train_df = train_df.sample(frac=1)
    test_df = pd.read_csv(test_file, encoding='utf-8')
    train_df['label'] = train_df['subject'].str.cat(train_df['sentiment_value'].astype(str))
    remove_stop_words = True
    # 移除停用词
    if remove_stop_words:
        train_df['content'] = train_df.content.map(
            lambda x: ''.join([e for e in x.strip().split() if e not in stop_words]))
        test_df['content'] = test_df.content.map(
            lambda x: ''.join([e for e in x.strip().split() if e not in stop_words]))
    else:
        train_df['content'] = train_df.content.map(lambda x: ''.join(x.strip().split()))
        test_df['content'] = test_df.content.map(lambda x: ''.join(x.strip().split()))

    # 建立 content: label的字典
    train_dict = {}
    train_id_label_dict = {}
    for ind, row in train_df.iterrows():
        content, label, content_id = row['content'], row['label'], row['content_id']
        if train_dict.get(content) is None:
            train_dict[content] = set([label])
        else:
            train_dict[content].add(label)
        train_id_label_dict[content_id] = label


    conts = []  # content list
    labels = []  # label list
    for k, v in train_dict.items():
        conts.append(k)
        labels.append(v)

    from sklearn.model_selection import train_test_split
    train_conts, valid_conts, train_labels, valid_labels = train_test_split(conts, labels, test_size=test_size)
    
    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(train_labels) # 将the ground truth变成one hot 编码
    y_valid= mlb.transform(valid_labels)
    
    pk.dump(mlb, open('multi_label_encoder.sav','wb'))
    # mlb = pk.load(open('multi_label_encoder.sav','rb'))

    train_content_list = [jieba.lcut(str(c)) for c in train_conts] # 为每个内容分词，得到一个分词后的list

    valid_content_list = [jieba.lcut(str(c)) for c in valid_conts] # 为每个内容分词，得到一个分词后的list

    test_content_list = [jieba.lcut(c) for c in test_df.content.astype(str).values]# 对test测试集分词，然后得到分词后list

    word_set = set([word for row in list(train_content_list) + list(valid_content_list) + list(test_content_list) for word in row]) # 得到训练集+测试集的词汇表
    print(len(word_set))
    word2index = {w: i + 1 for i, w in enumerate(word_set)} # 为词汇表得到词汇索引

    train_seqs = [[word2index[w] for w in l] for l in train_content_list]  # 得到每句话，每句话是由每个词对应的index组成的
    valid_seqs = [[word2index[w] for w in l] for l in valid_content_list]  # 得到每句话，每句话是由每个词对应的index组成的
    seqs_dev = [[word2index[w] for w in l] for l in test_content_list]  # 得到测试集上每个词的index
    return train_seqs, valid_seqs, seqs_dev,word2index, y_train, y_valid, train_id_label_dict, valid_labels



def raw_file2matrix(train_file, test_file, stop_words):
    train_df = pd.read_csv(train_file, encoding='utf-8')
    train_df = train_df.sample(frac=1)
    test_df = pd.read_csv(test_file, encoding='utf-8')
    train_df['label'] = train_df['subject'].str.cat(train_df['sentiment_value'].astype(str))
    remove_stop_words = True
    # 移除停用词
    if remove_stop_words:
        train_df['content'] = train_df.content.map(
            lambda x: ''.join([e for e in x.strip().split() if e not in stop_words]))
        test_df['content'] = test_df.content.map(
            lambda x: ''.join([e for e in x.strip().split() if e not in stop_words]))
    else:
        train_df['content'] = train_df.content.map(lambda x: ''.join(x.strip().split()))
        test_df['content'] = test_df.content.map(lambda x: ''.join(x.strip().split()))

    
    train_dict = {}# 建立 content: label的字典
    train_id_label_dict = {} # 建立 content id : label的字典
    for ind, row in train_df.iterrows():
        content, label, content_id = row['content'], row['label'], row['content_id']
        if train_dict.get(content) is None:
            train_dict[content] = set([label])
        else:
            train_dict[content].add(label)
        train_id_label_dict[content_id] = label
        
        
    conts = []  # content list
    labels = []  # label list
    for k, v in train_dict.items():
        conts.append(k)
        labels.append(v)

    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(labels) # 将the ground truth变成one hot 编码

    pk.dump(mlb, open('../src/multi_label_encoder.sav','wb'))
    # mlb = pk.load(open('multi_label_encoder.sav','rb'))

    content_list = [jieba.lcut(str(c)) for c in conts] # 为每个内容分词，得到一个分词后的list

    test_content_list = [jieba.lcut(c) for c in test_df.content.astype(str).values]# 对test测试集分词，然后得到分词后list

    word_set = set([word for row in list(content_list) + list(test_content_list) for word in row]) # 得到训练集+测试集的词汇表
    print(len(word_set))
    word2index = {w: i + 1 for i, w in enumerate(word_set)} # 为词汇表得到词汇索引

    seqs = [[word2index[w] for w in l] for l in content_list]  # 得到每句话，每句话是由每个词对应的index组成的
    seqs_dev = [[word2index[w] for w in l] for l in test_content_list]  # 得到测试集上每个词的index
    return seqs, seqs_dev, word2index, y_train


def raw_file2matrix_char(train_file, test_file, stop_words):
    train_df = pd.read_csv(train_file, encoding='utf-8')
    train_df = train_df.sample(frac=1)
    test_df = pd.read_csv(test_file, encoding='utf-8')
    train_df['label'] = train_df['subject'].str.cat(train_df['sentiment_value'].astype(str))
    remove_stop_words = True
    # 移除停用词
    if remove_stop_words:
        train_df['content'] = train_df.content.map(
            lambda x: ''.join([e for e in x.strip().split() if e not in stop_words]))
        test_df['content'] = test_df.content.map(
            lambda x: ''.join([e for e in x.strip().split() if e not in stop_words]))
    else:
        train_df['content'] = train_df.content.map(lambda x: ''.join(x.strip().split()))
        test_df['content'] = test_df.content.map(lambda x: ''.join(x.strip().split()))

    
    train_dict = {}# 建立 content: label的字典
    train_id_label_dict = {} # 建立 content id : label的字典
    for ind, row in train_df.iterrows():
        content, label, content_id = row['content'], row['label'], row['content_id']
        if train_dict.get(content) is None:
            train_dict[content] = set([label])
        else:
            train_dict[content].add(label)
        train_id_label_dict[content_id] = label
        
        
    conts = []  # content list
    labels = []  # label list
    for k, v in train_dict.items():
        conts.append(k)
        labels.append(v)

    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(labels) # 将the ground truth变成one hot 编码

    pk.dump(mlb, open('../src/multi_label_encoder.sav','wb'))
    # mlb = pk.load(open('multi_label_encoder.sav','rb'))

    content_list = [c for c in conts] # 为每个内容分词，得到一个分词后的list

    test_content_list = [c for c in test_df.content.astype(str).values]# 对test测试集分词，然后得到分词后list

    word_set = set([word for row in list(content_list) + list(test_content_list) for word in row]) # 得到训练集+测试集的词汇表
    print(len(word_set))
    word2index = {w: i + 1 for i, w in enumerate(word_set)} # 为词汇表得到词汇索引

    seqs = [[word2index[w] for w in l] for l in content_list]  # 得到每句话，每句话是由每个词对应的index组成的
    seqs_dev = [[word2index[w] for w in l] for l in test_content_list]  # 得到测试集上每个词的index
    return seqs, seqs_dev, word2index, y_train




def get_embedding_matrix(word2index, embeddings_index):  # 把词表变成embedding matrix，便于日后输入embedding层
    # 之前保存了内容的是embedding_index
    embedding_matrix = np.zeros((len(word2index) + 1, EMBEDDING_DIM))
    for word, i in word2index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix



def get_padding_data(seqs_train, seqs_valid, seqs_dev, maxlen=100):
    x_train = sequence.pad_sequences(seqs_train, maxlen=maxlen)
    x_valid = sequence.pad_sequences(seqs_valid, maxlen=maxlen)
    x_dev = sequence.pad_sequences(seqs_dev, maxlen=maxlen)
    return x_train, x_valid, x_dev



def get_padding_data_1(seqs, seqs_dev, maxlen=100):
    x_train = sequence.pad_sequences(seqs, maxlen=maxlen)
    x_dev = sequence.pad_sequences(seqs_dev, maxlen=maxlen)
    return x_train, x_dev



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



def pred2res(pred4):
    tmp = [[i for i in row] for row in pred4] # 就是pred4
    pro = tmp
    for i, v in enumerate(tmp):  # 带着索引的循环。对于每句话来说的30个概率
        if max(v) <= 0.5: # 不存在互斥一说
            max_val = max(v)
            pro[i] = [1 if j == max_val else 0 for j in v]
        else: 
            pro[i] = [int(round(j)) for j in v] # 四舍五入->概率变成了0,1 

    for i in range(len(tmp)):#对每个人
        for j in range(10): # 每个主题（内含三个情感）
            count = pro[i][3*j]+pro[i][j*3+1]+pro[i][j*3+2]
            if count > 1:
                max_sum = np.argmax(tmp[i][j*3:j*3+3], 0)
                pro[i][j*3:j*3+3] = [0]*3
                pro[i][j*3+max_sum] = 1


    # 将0-1矩阵转成语义表示的label
    mlb = pk.load(open('../src/multi_label_encoder.sav', 'rb'))
    pro = np.asanyarray(pro)  # 把内部任何的都变成了array

    res = mlb.inverse_transform(pro)  # 变成语义的特征了
    content_ids = []
    subjs = []
    sent_vals = []
    valid_id_content_dict = {}
    test_df = pd.read_csv(test_file, encoding='utf-8')
    for c, r in zip(test_df.content_id, res):  # 组队循环。接下来的部分是拆分label（与之前的连接是逆运算）
        for t in r:
            if '-' in t:
                sent_val = -1
                subj = t[:-2]
            else:
                sent_val = int(t[-1])  # 0或者1
                subj = t[:-1]
            content_ids .append(c)  # cids是content_id的列表。如果有多对主题-情感。那么id也会多次重复
            subjs.append(subj)
            sent_vals.append(sent_val)

    res_df = pd.DataFrame({'content_id': content_ids , 'subject': subjs, 'sentiment_value': sent_vals,
                           'sentiment_word': ['' for i in range(len(content_ids))]})

    columns = ['content_id', 'subject', 'sentiment_value', 'sentiment_word']
    res_df = res_df.reindex(columns=columns)
    return res, res_df

def get_f1_score(valid_labels, res):
    tp = 0
    fp = 0
    fn = 0
    for i in range(len(res)):
        for su_se_pair in res[i]:
            if su_se_pair in valid_labels[i]:
                tp += 1
            else:
                fp += 1
        for su_se_pair in valid_labels[i]:
            if not(su_se_pair in res[i]):
                fn += 1
                
    p = tp/(tp+fp+0.0)
    r = tp/(tp+fn+0.0)
    f1 = 2*p*r/(p+r)
    print("f1 score: ", f1)
    return f1



def generate_shuffle_array(X_train, y_train, num=4):
    X_trains = [X_train, X_train, X_train, X_train]
    y_trains = [y_train, y_train, y_train, y_train]
    for i in range(len(X_trains)):
        if i== 0:
            continue
        for sentence in X_trains[i]:
            np.random.shuffle(sentence)
    return np.vstack(X_trains), np.vstack(y_trains)
    X_trains, y_trains = generate_shuffle_array(X_train)
    
    

def drop_array(X_train, num=20):
    maxlen = X_train.shape[1]

    for i in range(X_train.shape[0]):
        if np.random.rand() > 0.4: # 有一半的可能不删除什么
            for j in range(num):
                index = int(np.random.rand()*maxlen)
                # print(index)
                X_train[i][index] = 0
                    
                    

if __name__ == '__main__':
    VALID = False
    if VALID:
        test_size = 0.333
    else:
        test_size = 0
        
    stop_words = load_stop_words()
    # 将raw file转成用word_index表示的句子们,y_train已经是one_hot_encoder了
    
    seqs, seqs_dev, word2index, y_train = raw_file2matrix(train_file, test_file, stop_words)
    embeddings_index = load_embeddings_index()
    embedding_matrix = get_embedding_matrix(word2index, embeddings_index)  # word-index-embedding是它们之间的链接关系
   
    # X_train, X_valid, X_test = get_padding_data(seqs_train, seqs_valid, seqs_dev)  # seqs needs to be a list of a list.把列表变成矩阵，列数是embedding_dim
    X_train, X_dev = get_padding_data_1(seqs, seqs_dev)
    
    if not os.path.exists(MATRIX_DIR):
        os.mkdir(MATRIX_DIR)
    np.savetxt(os.path.join(MATRIX_DIR, 'X_train_2'), X_train)
    np.savetxt(os.path.join(MATRIX_DIR,'X_test_2'), X_dev)
    np.savetxt(os.path.join(MATRIX_DIR, 'y_train_2'), y_train)
    