# Imports
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from nltk.corpus import reuters
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import pandas as pd
import nltk
import time
import sys
import torch
from sklearn.metrics import f1_score
from nltk import word_tokenize
from nltk.corpus import stopwords
import string, re
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, balanced_accuracy_score 
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,roc_curve)
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
from flair.embeddings import FlairEmbeddings,ELMoEmbeddings, WordEmbeddings
from flair.embeddings import DocumentPoolEmbeddings, FastTextEmbeddings ,Sentence, XLNetEmbeddings
from sentence_transformers import SentenceTransformer
from sklearn.neural_network import MLPClassifier




def clean_text(text: str, rm_numbers=True, rm_punct=True, rm_stop_words=True, rm_short_words=True):
    """ Function to perform common NLP pre-processing tasks. """

    # make lowercase
    text = text.lower()
    # remove punctuation
    if rm_punct:
        text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    # remove numbers
    if rm_numbers:
        text = re.sub(r'\d+', '', text)
    # remove whitespaces
    text = text.strip()
    # remove stopwords
    if rm_stop_words:
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text)
        text_list = [w for w in word_tokens if not w in stop_words]
        text = ' '.join(text_list)
    # remove short words
    if rm_short_words:
        text_list = [w for w in text.split() if len(w) >= 3]
        text = ' '.join(text_list)
    return text

# Train data  
def reuters_train_dataset(train=True, test=False, clean_txt=True):
    doc_ids = reuters.fileids()
    splits = [split_set for (requested, split_set) in [(train, 'train'), (test, 'test')] if requested]
    print('Loading Train data ... ')
    for split_set in splits:
        split_set_doc_ids = list(filter(lambda doc: doc.startswith(split_set), doc_ids))
        train_data = []
        for id in split_set_doc_ids:
            if clean_txt:
                text = clean_text(reuters.raw(id))
            else:
                text = ' '.join(word_tokenize(reuters.raw(id)))
            labels = reuters.categories(id)
            train_data.append({
              'text': text,
              'label': labels[0],
          })
    return train_data
# Test data          
def reuters_test_dataset(train=False, test=True, clean_txt=True):
    doc_ids = reuters.fileids()
    splits = [split_set for (requested, split_set) in [(train, 'train'), (test, 'test')] if requested]
    print('Loading Test data ... ')
    for split_set in splits:
        split_set_doc_ids = list(filter(lambda doc: doc.startswith(split_set), doc_ids))
        test_data = []
        for id in split_set_doc_ids:
            if clean_txt:
                text = clean_text(reuters.raw(id))
            else:
                text = ' '.join(word_tokenize(reuters.raw(id)))
            labels = reuters.categories(id)
          
            test_data.append({
              'text': text,
              'label': labels[0],
          })
    return test_data


# Various Embeddings - Bert, Universal Sentence Encoder, Infersent, GloVe, Fasttext, Ensemble Embeddings(Flair + GloVe)

# Embedding using Universal Sentence Encoder
def embed_module(module):
    with tf.Graph().as_default():
        sentences = tf.placeholder(tf.string)
        embed = hub.Module(module)
        embeddings = embed(sentences)
        session = tf.train.MonitoredSession()
    return lambda x: session.run(embeddings, {sentences: x})

def use_embeddings():
    train_data_list = []
    test_data_list = []
    module_url = "/opt/notebooks/embedding_model/"
    # Import the Universal Sentence Encoder's TF Hub module
    embed = embed_module(module_url)
    train_data_list = embed(final_train['text'].tolist())
    test_data_list = embed(final_test['text'].tolist())
    val_data_list = embed(final_val['text'].tolist())
    return train_data_list, test_data_list, val_data_list

# Bert
def bert_embeddings():
    train_data_list = []
    test_data_list = []
    val_data_list = []
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    train_data_list = model.encode(final_train['text'].tolist())
    test_data_list = model.encode(final_test['text'].tolist())
    val_data_list = model.encode(final_val['text'].tolist())
    return train_data_list, test_data_list, val_data_list

# Flair - GloVe - XLNet - FastText
def other_embeddings(embd):
    sess = tf.InteractiveSession()
    train_data_list = []
    test_data_list = []
    val_data_list = []
    if embd == 'glove':
        print('Starting Glove Embedding...')
        glove_embedding = WordEmbeddings('glove')
        document_embeddings = DocumentPoolEmbeddings(embeddings=[glove_embedding])
    elif embd == 'xlnet':
        print('Starting XLNet Embedding...')
        xlnet_embedding = XLNetEmbeddings('xlnet-large-cased')
        document_embeddings = DocumentPoolEmbeddings(embeddings=[xlnet_embedding])
    elif embd == 'fasttext':
        print('Starting Fasttext Embedding...')
        fasttext_embedding = WordEmbeddings('en')
        document_embeddings = DocumentPoolEmbeddings(embeddings=[fasttext_embedding])
    elif embd == 'elmo':
        print('Starting ELMo Embedding...')
        elmo_embedding = ELMoEmbeddings()
        document_embeddings = DocumentPoolEmbeddings(embeddings=[elmo_embedding])
    else:
        # init Flair embeddings
        flair_forward_embedding = FlairEmbeddings('multi-forward')
        flair_backward_embedding = FlairEmbeddings('multi-backward')
        glove_embedding = WordEmbeddings('glove')
        # now create the DocumentPoolEmbeddings object that combines all embeddings
        document_embeddings = DocumentPoolEmbeddings(embeddings=[glove_embedding, flair_forward_embedding, flair_backward_embedding])
    print('Train embedding Started...')
    for text in final_train['text'].tolist():
        text = Sentence(text)
        document_embeddings.embed(text)
        emb = text.get_embedding().detach().numpy()
        emb = tf.constant(emb).eval()
        train_data_list.append(emb)
    print('Embedded Train data!!')
    print('Test embedding Started...')
    for text in final_test['text'].tolist():
        text = Sentence(text)
        document_embeddings.embed(text)
        emb = text.get_embedding().detach().numpy()
        emb = tf.constant(emb).eval()
        test_data_list.append(emb)
    print('Embedded Test data!!')
    for text in final_val['text'].tolist():
        text = Sentence(text)
        document_embeddings.embed(text)
        emb = text.get_embedding().detach().numpy()
        emb = tf.constant(emb).eval()
        val_data_list.append(emb)
    print('Embedded Test data!!')
    return train_data_list, test_data_list, val_data_list


def infersent_embeddings():
    train_data_list = []
    test_data_list = []
    sys.path.append('/opt/notebooks/OCSVM_ISF_LOF_USE_Baselines/InferSent-master')
    # Load model
    from models import InferSent
    model_version = 1
    MODEL_PATH = "/opt/notebooks/OCSVM_ISF_LOF_USE_Baselines/InferSent-master/encoder/infersent%s.pkl" % model_version
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}
    model = InferSent(params_model)
    model.load_state_dict(torch.load(MODEL_PATH))
    # Keep it on CPU or put it on GPU
    use_cuda = False
    model = model.cuda() if use_cuda else model
    # If infersent1 -> use GloVe embeddings. If infersent2 -> use InferSent embeddings.
    W2V_PATH = '/opt/notebooks/OCSVM_ISF_LOF_USE_Baselines/InferSent-master/glove.840B.300d-003.txt' if model_version == 1 else '/opt/notebooks/OCSVM_ISF_LOF_USE_Baselines/InferSent-master/fastText/crawl-300d-2M.vec'
    model.set_w2v_path(W2V_PATH)
    # Load embeddings of K most frequent words
    model.build_vocab_k_words(K=100000)
    train_data_list = model.encode(final_train['text'].tolist(), bsize=128, tokenize=False, verbose=True)
    print('nb sentences encoded : {0}'.format(len(train_data_list)))
    test_data_list = model.encode(final_test['text'].tolist(), bsize=128, tokenize=False, verbose=True)
    print('nb sentences encoded : {0}'.format(len(test_data_list)))
    return train_data_list, test_data_list


# OCSVM model
def oneclass_svm(dataset, kernel, nu):
    svm = OneClassSVM(kernel=kernel, nu=nu).fit(dataset)
    return svm
  
# Isolation Forest
def isolationForest(dataset, rng):
    isolationforest = IsolationForest(behaviour='new', max_samples=100, random_state=rng, contamination='auto').fit(dataset)
    return isolationforest

def local_outlier_factory(dataset, neighbours):
    lof = LocalOutlierFactor(n_neighbors=neighbours, contamination=0.1,novelty=True).fit(dataset)
    return lof

def pca(train, test, val):
    #print('Principal Component Analysis ...')
    ss = StandardScaler()
    ss.fit(train)
    train = ss.transform(train)
    test = ss.transform(test)
    val = ss.transform(val)
    pca = PCA()
    pca = pca.fit(train)
    #print('Explained variance percentage = %0.2f' % sum(pca.explained_variance_ratio_))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    train = pca.transform(train)
    test = pca.transform(test)
    val = pca.transform(val)
    return train, test, val
def results(model, train_true_labels, train_predicted_labels, test_true_labels, test_predicted_labels):
    print('Model: ', model)
    #train_f1 = f1_score(train_true_labels, train_predicted_labels, average='macro')  
    #print('Train F1 Score: ', train_f1)
    test_f1 = f1_score(test_true_labels, test_predicted_labels, average='macro')  
    print('Test F1 Score: ', test_f1)
    #print("Train accuracy:", accuracy_score(train_true_labels, train_predicted_labels))
    accuracy = accuracy_score(test_true_labels, test_predicted_labels)
    print("Test accuracy:", accuracy )
    #results_global_list.append(test_f1)
    #results_global_list.append(accuracy)
    results = confusion_matrix(test_true_labels, test_predicted_labels, [1,-1]) 
    #print('Confusion Matrix :')
    #print(results) 
    #print('Report : ')
    LABELS = ["known", "unknown"]
    plt.figure(figsize=(6, 6))
    sns.heatmap(results, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
    plt.title("Confusion matrix - Reuters - " + str(model))
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.show()
    return test_f1, accuracy
    #print(classification_report(test_true_labels, test_predicted_labels))


def data_for_sota():
    # Numerical Labeling - 1 for normal class; -1 for anomaly class                    
    train_data_df['new_label'] = train_data_df['label']
    train_data_df.loc[train_data_df['label'] == TRAIN_CLASS, 'new_label'] = 1
    train_data_df.loc[train_data_df['label'] != TRAIN_CLASS, 'new_label'] = -1
    val_data_df['new_label'] = val_data_df['label']
    val_data_df.loc[val_data_df['label'] == TRAIN_CLASS, 'new_label'] = 1
    val_data_df.loc[val_data_df['label'] != TRAIN_CLASS, 'new_label'] = -1
    test_data_df['new_label'] = test_data_df['label']
    test_data_df.loc[test_data_df['label'] == TRAIN_CLASS, 'new_label'] = 1
    test_data_df.loc[test_data_df['label'] != TRAIN_CLASS, 'new_label'] = -1
    # Preparing train data with only normal class
    final_train = train_data_df.loc[train_data_df['new_label'] == 1]
    print(final_train['new_label'].value_counts())
    #val data
    final_val = val_data_df
    print(final_val['new_label'].value_counts())
    # Test data with both normal and other classes
    final_test = test_data_df
    print(final_test['new_label'].value_counts())
    return final_train, final_test, final_val


def data_for_autoencoder():
    # Converting the train and test data into dataframe
    train_data_df['new_label'] = train_data_df['label']
    train_data_df.loc[train_data_df['label'] == TRAIN_CLASS, 'new_label'] = 0
    train_data_df.loc[train_data_df['label'] != TRAIN_CLASS, 'new_label'] = 1
    val_data_df['new_label'] = val_data_df['label']
    val_data_df.loc[val_data_df['label'] == TRAIN_CLASS, 'new_label'] = 0
    val_data_df.loc[val_data_df['label'] != TRAIN_CLASS, 'new_label'] = 1
    test_data_df['new_label'] = test_data_df['label']
    test_data_df.loc[test_data_df['label'] == TRAIN_CLASS, 'new_label'] = 0
    test_data_df.loc[test_data_df['label'] != TRAIN_CLASS, 'new_label'] = 1
    # Preparing train data with only normal class
    final_train = train_data_df.loc[train_data_df['new_label'] == 0]
    print(final_train['new_label'].value_counts())
    final_val = val_data_df
    print(final_val['new_label'].value_counts())
    # Test data with both normal and other classes
    final_test = test_data_df
    print(final_test['new_label'].value_counts())
    return final_train, final_test, final_val

def mostFrequent(arr): 
    n = len(arr)
    # Insert all elements in Hash. 
    Hash = dict() 
    for i in range(n): 
        if arr[i] in Hash.keys(): 
            Hash[arr[i]] += 1
        else: 
            Hash[arr[i]] = 1
  
    # find the max frequency 
    max_count = 0
    res = -1
    for i in Hash:  
        if (max_count < Hash[i]):  
            res = i 
            max_count = Hash[i] 
          
    return res  

def ensemble_results(model, test_true_labels, test_predicted_labels):
    print('Model: ', model)
    test_f1 = f1_score(test_true_labels, test_predicted_labels, average='macro')  
    print('Test F1 Score: ', test_f1)
    accuracy = accuracy_score(test_true_labels, test_predicted_labels)
    print("Test accuracy:", accuracy)
    results_global_list.append(round(test_f1,2))
    results_global_list.append(round(accuracy*100,1))
    #results = confusion_matrix(test_true_labels, test_predicted_labels, [1,-1]) 
    #print('Confusion Matrix :')
    #print(results) 
    #print('Report : ')
    #LABELS = ["known", "unknown"]
    '''plt.figure(figsize=(6, 6))
    sns.heatmap(results, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
    plt.title("Confusion matrix - Reuters - " + str(model))
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.show()
    #print(classification_report(test_true_labels, test_predicted_labels))
    '''

    
def autoencoder_model(X_Test, Y_Test):
    input_dim = X_train.shape[1]
    print('Input dimension: ',input_dim)
    encoding_dim = 14
    nb_epoch = 100
    batch_size = 32

    input_layer = Input(shape=(input_dim, ))
    encoder = Dense(encoding_dim, activation="tanh", 
                    activity_regularizer=regularizers.l1(10e-5))(input_layer)
    encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)
    decoder = Dense(int(encoding_dim / 2), activation='tanh')(encoder)
    decoder = Dense(input_dim, activation='relu')(decoder)
    autoencoder = Model(inputs=input_layer, outputs=decoder)


    autoencoder.compile(optimizer='adam', 
                        loss='mean_squared_error', 
                        metrics=['accuracy'])

    checkpointer = ModelCheckpoint(filepath="model_news20.h5",
                                   verbose=0,
                                   save_best_only=True)
    tensorboard = TensorBoard(log_dir='./logs_reuters',
                              histogram_freq=0,
                              write_graph=True,
                              write_images=True)

    history = autoencoder.fit(X_train, X_train,
                        epochs=nb_epoch,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_data=(X_Test, X_Test),
                        verbose=1,
                        callbacks=[checkpointer, tensorboard]).history
    
    autoencoder = load_model('model_news20.h5')
    
    predictions = autoencoder.predict(X_Test)

    mse = np.mean(np.power(X_Test - predictions, 2), axis=1)
    error_df = pd.DataFrame({'reconstruction_error': mse,
                            'true_class': Y_Test})

    #error_df.describe()
    auto_score_pred = []
    for scor in predictions.tolist():
        auto_score_pred.append(np.mean(scor))
    normalized_auto = [((x-min(auto_score_pred))/(max(auto_score_pred)-min(auto_score_pred))) for x in auto_score_pred]
    print(len(normalized_auto))
    return normalized_auto

    
if __name__ == "__main__":

    directory = "/opt/notebooks/datasets/reuters"
    nltk.download('stopwords')
    nltk.download('punkt')
    if directory not in nltk.data.path:
        nltk.data.path.append(directory)
    
    print('Loading data...')
    train_data = reuters_train_dataset()
    test_data = reuters_test_dataset()
    
    # Converting the train and test data into dataframe
    train_data_df = pd.DataFrame(train_data[:4000], columns = ['text' , 'label'])
    val_data_df = pd.DataFrame(train_data[4000:], columns = ['text' , 'label'])
    test_data_df = pd.DataFrame(test_data, columns = ['text' , 'label'])
    
    
    group_list = ['earn', 'acq', 'crude', 'interest', 'trade', 'ship', 'money-fx']
    train_data_df = train_data_df[train_data_df['label'].isin(group_list)]
    val_data_df = val_data_df[val_data_df['label'].isin(group_list)]
    test_data_df = test_data_df[test_data_df['label'].isin(group_list)]
    
    
    embedding_list = ['fasttext']
    for embedding in embedding_list:
        print('Running for Embeddings: ', embedding)
        f1_scores = []
        for group in group_list:
            if group == 'trade':
                start = time.time()
                print('Running for News group: ', group)
                TRAIN_CLASS = group
                results_global_list = []

                #Load data
                final_train, final_test, final_val = data_for_sota()
                print('Embeddings Started: ')
                emb_start = time.time()
                # Specify the embeddings 'glove', 'xlnet', 'fasttext', 'elmo', 'ensemble'
                if embedding == 'use':
                    train_data_list_global, test_data_list_global, val_data_list_global = use_embeddings()
                if embedding == 'bert':    
                    train_data_list_global, test_data_list_global, val_data_list_global = bert_embeddings()
                if embedding == 'infersent':
                    train_data_list_global, test_data_list_global = infersent_embeddings()
                if embedding == 'elmo':
                    train_data_list_global, test_data_list_global = elmo_embeddings()
                if embedding == 'xlnet':
                    train_data_list_global, test_data_list_global = other_embeddings('xlnet')
                if embedding == 'fasttext':
                    train_data_list_global, test_data_list_global, val_data_list_global = other_embeddings('fasttext')
                print('Embedding time: ', time.time() - emb_start)

                train_data_list = train_data_list_global
                test_data_list = test_data_list_global
                val_data_list = val_data_list_global
                #PCA
                train_data_list, test_data_list, val_data_list = pca(train_data_list, test_data_list, val_data_list)
                #---------------------------------- Model 1: OCSVM-----------------------------------
                print('----------------- OCSVM Model -----------------')
                svm_model = oneclass_svm(train_data_list, 'rbf', 0.1)
                y_pred_train = svm_model.predict(train_data_list)
                y_pred_test = svm_model.predict(test_data_list)
                y_test_scores = svm_model.score_samples(test_data_list)
                y_val_scores = svm_model.score_samples(val_data_list)
                f1, acc = results('one Class SVM', final_train['new_label'].tolist(), y_pred_train, final_test['new_label'].tolist(), y_pred_test)
                results_global_list.append(round(f1,2))
                results_global_list.append(round(acc*100,1))

                test_score_list = y_test_scores.tolist()
                val_score_list = y_val_scores.tolist()
                normalized_test_svm = [((x-min(test_score_list))/(max(test_score_list)-min(test_score_list))) for x in test_score_list]
                normalized_val_svm = [((x-min(val_score_list))/(max(val_score_list)-min(val_score_list))) for x in val_score_list]
                print(len(normalized_test_svm))
                print(len(normalized_val_svm))

                #---------------------------------- Model 2: Isolation Forest-----------------------------------
                print('----------------- isolation forest -----------------')
                svm_model = oneclass_svm(train_data_list, 'rbf', 0.1)
                rand_state = np.random.RandomState(42)
                iso_forest = isolationForest(train_data_list, rand_state)
                y_pred_iso_train = iso_forest.predict(train_data_list)
                y_pred_iso_test = iso_forest.predict(test_data_list)
                y_test_scores_isf = iso_forest.score_samples(test_data_list)
                y_val_scores_isf = iso_forest.score_samples(val_data_list)
                f1, acc = results('Isolation Forest', final_train['new_label'].tolist(), y_pred_iso_train, final_test['new_label'].tolist(), y_pred_iso_test)
                results_global_list.append(round(f1,2))
                results_global_list.append(round(acc*100,1))

                test_score_list = y_test_scores_isf.tolist()
                val_score_list_isf = y_val_scores_isf.tolist()
                normalized_test_isf = [((x-min(test_score_list))/(max(test_score_list)-min(test_score_list))) for x in test_score_list]
                normalized_val_isf = [((x-min(val_score_list_isf))/(max(val_score_list_isf)-min(val_score_list_isf))) for x in val_score_list_isf]
                print(len(normalized_test_isf))
                print(len(normalized_val_isf))

                '''#----------------------------------Model 3: Local Outlier Factory-----------------------------------
                print('----------------- local outlier factory -----------------')
                svm_model = oneclass_svm(train_data_list, 'rbf', 0.1)
                lof = local_outlier_factory(train_data_list, 250)
                y_pred_lof_train = lof.predict(train_data_list)
                y_pred_lof_test = lof.predict(test_data_list)
                f1, acc = results('Local Outlier Factory', final_train['new_label'].tolist(), y_pred_lof_train, final_test['new_label'].tolist(), y_pred_lof_test)
                results_global_list.append(round(f1,2))
                results_global_list.append(round(acc*100,1))
                '''

                #----------------------------- Model 4: Data processing for AutoEncoder-------------------------------
                # Converting the train and test data into dataframe
                # Converting the train and test data into dataframe
                train_data_df = pd.DataFrame(train_data[:4000], columns = ['text' , 'label'])
                val_data_df = pd.DataFrame(train_data[4000:], columns = ['text' , 'label'])
                test_data_df = pd.DataFrame(test_data, columns = ['text' , 'label'])


                group_list = ['earn', 'acq', 'crude', 'interest', 'trade', 'ship', 'money-fx']
                train_data_df = train_data_df[train_data_df['label'].isin(group_list)]
                val_data_df = val_data_df[val_data_df['label'].isin(group_list)]
                test_data_df = test_data_df[test_data_df['label'].isin(group_list)]


                final_train, final_test, final_val = data_for_autoencoder()

                X_train = train_data_list_global
                y_train = final_train['new_label'].tolist()
                X_test = test_data_list_global
                y_test = final_test['new_label'].tolist()
                X_val = val_data_list_global
                y_val = final_val['new_label'].tolist()   

                X_train = np.asarray(X_train)
                X_test = np.asarray(X_test)
                X_val = np.asarray(X_val)




                # ******************************** Auto Encoder Model Bulding ***************************************
                print('----------------- Auto Encoder -----------------')
                auto_val_prediction_scores = autoencoder_model(X_val, y_val)
                concat_array_val = np.array((normalized_val_svm, normalized_val_isf, auto_val_prediction_scores), dtype=float)
                ensemble_array_val = np.append(val_data_list_global, concat_array_val.T, axis=1)
                mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
                mlp.fit(ensemble_array_val, y_val)


                auto_test_prediction_scores = autoencoder_model(X_test, y_test)
                concat_array_test = np.array((normalized_test_svm, normalized_test_isf, auto_test_prediction_scores), dtype=float)
                ensemble_array_test = np.append(test_data_list_global, concat_array_test.T, axis=1)
                ensemble_predictions = mlp.predict(ensemble_array_test)
                text = 'F1 Score for'+ str(group)+ ': '
                print(text , f1_score(y_test,ensemble_predictions, average='macro'))
                f1_scores.append(f1_score(y_test,ensemble_predictions, average='macro'))


                '''
                #Load the model for prediction
                autoencoder = load_model('model_news20.h5')
                predictions = autoencoder.predict(X_test)

                mse = np.mean(np.power(X_test - predictions, 2), axis=1)
                error_df = pd.DataFrame({'reconstruction_error': mse,
                                        'true_class': y_test})

                #error_df.describe()


                fpr, tpr, thresholds = roc_curve(error_df.true_class, error_df.reconstruction_error)
                roc_auc = auc(fpr, tpr)

                precision, recall, th = precision_recall_curve(error_df.true_class, error_df.reconstruction_error)

                accuracies = []
                for threshold in th:
                    y_pred = [1 if e > threshold else 0 for e in error_df.reconstruction_error.values]
                    accuracies.append(balanced_accuracy_score(error_df.true_class, y_pred))
                print("Maximum Test accuracy:", max(accuracies))
                #print(accuracies.index(max(accuracies)))
                print('Threshold for maximal accuracy: ', th[accuracies.index(max(accuracies))])
                threshold = th[accuracies.index(max(accuracies))]

                groups = error_df.groupby('true_class')

                LABELS = ["known", "unknown"]
                y_pred_autoencoder = [1 if e > threshold else 0 for e in error_df.reconstruction_error.values]
                conf_matrix = confusion_matrix(error_df.true_class, y_pred_autoencoder,[0,1])
                test_f1 = f1_score(error_df.true_class, y_pred_autoencoder, average='macro')  
                print('Test F1 Score: ', test_f1)
                accuracy = balanced_accuracy_score(error_df.true_class, y_pred_autoencoder)
                print("Test accuracy:", accuracy)
                results_global_list.append(round(test_f1,2))
                results_global_list.append(round(accuracy*100,1))

                # Base individual models end. Ensemble approaches begin from here

                for i in range(len(y_pred_autoencoder)):
                    if y_pred_autoencoder[i] == 0:
                        y_pred_autoencoder[i] = 1
                    else:
                        y_pred_autoencoder[i] = -1
                #print(y_pred_autoencoder)
                final_train, final_test = data_for_sota()
                print('----------------- Ensemble model -----------------')
                y_pred_test_ensemble = []
                for i in range(len(final_test['new_label'].tolist())):
                    arr = [ y_pred_test[i], y_pred_iso_test[i], y_pred_autoencoder[i]]
                    y_pred_test_ensemble.append(mostFrequent(arr))

                ensemble_results('Ensemble', final_test['new_label'].tolist(), y_pred_test_ensemble)

                print('f1scores and accuracies: ',results_global_list)

                '''
        filename = '/opt/notebooks/OCSVM_ISF_LOF_USE_Baselines/results_output_new/exp_ensemble_reuters.txt'
        with open(filename, 'w') as f:
            for item in f1_scores:
                f.write("%s\t" % str(item))
        print("total time taken this loop: ", time.time() - start)