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
#import tf_sentencepiece
import sys, json, os
import torch, csv
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

def quora_dataset():
    input_file = '/opt/notebooks/datasets/questions.csv'
    train_data = []
    test_data = []
    train = {}
    test = {}
    with open(input_file, encoding="utf8") as file:
        csv_data = csv.reader(file)
        next(csv_data, None)
        for elem1 in csv_data:
            train_text = elem1[3]
            train['text'] = clean_text(train_text)
            train['label'] = 'duplicate'
            train_data.append(train)
            train = {}
            test_text = elem1[4]
            test['text'] = clean_text(test_text)
            if int(elem1[5]) == 1:
                test['label'] = 'duplicate'
            else:
                test['label'] = 'novel'
            test_data.append(test)
            test = {}
        return train_data, test_data 


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
    embed = embed_module("https://tfhub.dev/google/universal-sentence-encoder/1")
    train_data_list = embed(final_train['text'].tolist())
    test_data_list = embed(final_test['text'].tolist())
    return train_data_list, test_data_list
                                             
# Bert
def bert_embeddings():
    train_data_list = []
    test_data_list = []
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    train_data_list = model.encode(final_train['text'].tolist())
    test_data_list = model.encode(final_test['text'].tolist())
    return train_data_list, test_data_list

# Flair - GloVe - XLNet - FastText
def other_embeddings(embd):
    sess = tf.InteractiveSession()
    train_data_list = []
    test_data_list = []
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
    return train_data_list, test_data_list

def elmo_vectors(x):
    elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
    embeddings = elmo(x, signature="default", as_dict=True)["elmo"]
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        # return average of ELMo features
        return sess.run(tf.reduce_mean(embeddings,1))
        
def elmo_embeddings():
    train_data_list = []
    test_data_list = []
    
    elmo_train = [elmo_vectors(final_train['text'].tolist())]
    elmo_test = [elmo_vectors(final_test['text'].tolist())]
    for i in range(len(final_train['text'].tolist())):
        train_data_list.append(elmo_train[0][i])
    for i in range(len(final_test['text'].tolist())):
        test_data_list.append(elmo_test[0][i])
    return train_data_list, test_data_list

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

def pca(train, test):
    #print('Principal Component Analysis ...')
    ss = StandardScaler()
    ss.fit(train)
    train = ss.transform(train)
    test = ss.transform(test)
    pca = PCA()
    pca = pca.fit(train)
    #print('Explained variance percentage = %0.2f' % sum(pca.explained_variance_ratio_))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    train = pca.transform(train)
    test = pca.transform(test)
    return train, test

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
    test_data_df['new_label'] = test_data_df['label']
    test_data_df.loc[test_data_df['label'] == TRAIN_CLASS, 'new_label'] = 1
    test_data_df.loc[test_data_df['label'] != TRAIN_CLASS, 'new_label'] = -1
    # Preparing train data with only normal class
    final_train = train_data_df.loc[train_data_df['new_label'] == 1]
    #print(final_train['new_label'].value_counts())
    # Test data with both normal and other classes
    final_test = test_data_df
    #print(final_test['new_label'].value_counts())
    return final_train, final_test


def data_for_autoencoder():
    # Converting the train and test data into dataframe
    train_data_df['new_label'] = train_data_df['label']
    train_data_df.loc[train_data_df['label'] == TRAIN_CLASS, 'new_label'] = 0
    train_data_df.loc[train_data_df['label'] != TRAIN_CLASS, 'new_label'] = 1
    test_data_df['new_label'] = test_data_df['label']
    test_data_df.loc[test_data_df['label'] == TRAIN_CLASS, 'new_label'] = 0
    test_data_df.loc[test_data_df['label'] != TRAIN_CLASS, 'new_label'] = 1
    # Preparing train data with only normal class
    final_train = train_data_df.loc[train_data_df['new_label'] == 0]
    #print(final_train['new_label'].value_counts())
    # Test data with both normal and other classes
    final_test = test_data_df
    #print(final_test['new_label'].value_counts())
    return final_train, final_test

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

if __name__ == "__main__":

    #directory = "/opt/notebooks/datasets/reuters"
    nltk.download('stopwords')
    nltk.download('punkt')
    #if directory not in nltk.data.path:
    #    nltk.data.path.append(directory)
    
    print('Loading data...')
    # Calling the quora dataset functions
    train_data, test_data = quora_dataset()
    
    
    # Converting the train and test data into dataframe
    train_data_df = pd.DataFrame(train_data, columns = ['text' , 'label'])
    test_data_df = pd.DataFrame(test_data, columns = ['text' , 'label'])
    
    #train_data_df = train_data_df.loc[:80000]
    #test_data_df = test_data_df.loc[:80000]
    train_data_df = train_data_df.drop(list(train_data_df[train_data_df['text'] == ''].index))
    test_data_df = test_data_df.drop(list(test_data_df[test_data_df['text'] == ''].index))
    group_list = ['duplicate']
    
    embedding_list = ['use']
    for embedding in embedding_list:
        print('Running for Embeddings: ', embedding)
        for group in group_list:
            start = time.time()
            print('Running for News group: ', group)
            TRAIN_CLASS = group
            results_global_list = []
            
            #Load data
            final_train, final_test = data_for_sota()
            print('Embeddings Started: ')
            emb_start = time.time()
            # Specify the embeddings 'glove', 'xlnet', 'fasttext', 'elmo', 'ensemble'
            if embedding == 'use':
                train_data_list_global, test_data_list_global = use_embeddings()
            if embedding == 'bert':    
                train_data_list_global, test_data_list_global = bert_embeddings()
            if embedding == 'infersent':
                train_data_list_global, test_data_list_global = infersent_embeddings()
            if embedding == 'elmo':
                train_data_list_global, test_data_list_global = elmo_embeddings()
            if embedding == 'glove':
                train_data_list_global, test_data_list_global = other_embeddings('glove')
            if embedding == 'fasttext':
                train_data_list_global, test_data_list_global = other_embeddings('fasttext')
            print('Embedding time: ', time.time() - emb_start)
            
            train_data_list = train_data_list_global
            test_data_list = test_data_list_global
            #PCA
            train_data_list, test_data_list = pca(train_data_list, test_data_list)
            #---------------------------------- Model 1: OCSVM-----------------------------------
            print('----------------- OCSVM Model -----------------')
            svm_model = oneclass_svm(train_data_list, 'rbf', 0.1)
            y_pred_train = svm_model.predict(train_data_list)
            y_pred_test = svm_model.predict(test_data_list)
            f1, acc = results('one Class SVM', final_train['new_label'].tolist(), y_pred_train, final_test['new_label'].tolist(), y_pred_test)
            results_global_list.append(round(f1,2))
            results_global_list.append(round(acc*100,1))
            
            #---------------------------------- Model 2: Isolation Forest-----------------------------------
            print('----------------- isolation forest -----------------')
            svm_model = oneclass_svm(train_data_list, 'rbf', 0.1)
            rand_state = np.random.RandomState(42)
            iso_forest = isolationForest(train_data_list, rand_state)
            y_pred_iso_train = iso_forest.predict(train_data_list)
            y_pred_iso_test = iso_forest.predict(test_data_list)
            f1, acc = results('Isolation Forest', final_train['new_label'].tolist(), y_pred_iso_train, final_test['new_label'].tolist(), y_pred_iso_test)
            results_global_list.append(round(f1,2))
            results_global_list.append(round(acc*100,1))
            
            #----------------------------------Model 3: Local Outlier Factory-----------------------------------
            '''print('----------------- local outlier factory -----------------')
            svm_model = oneclass_svm(train_data_list, 'rbf', 0.1)
            lof = local_outlier_factory(train_data_list, 250)
            y_pred_lof_train = lof.predict(train_data_list)
            y_pred_lof_test = lof.predict(test_data_list)
            f1, acc = results('Local Outlier Factory', final_train['new_label'].tolist(), y_pred_lof_train, final_test['new_label'].tolist(), y_pred_lof_test)
            results_global_list.append(round(f1,2))
            results_global_list.append(round(acc*100,1))'''
            
            
            #----------------------------- Model 4: Data processing for AutoEncoder-------------------------------
            # Converting the train and test data into dataframe

            train_data_df = pd.DataFrame(train_data, columns = ['text' , 'label'])
            test_data_df = pd.DataFrame(test_data, columns = ['text' , 'label'])
            
            #train_data_df = train_data_df.loc[:80000]
            #test_data_df = test_data_df.loc[:80000]
            train_data_df = train_data_df.drop(list(train_data_df[train_data_df['text'] == ''].index))
            test_data_df = test_data_df.drop(list(test_data_df[test_data_df['text'] == ''].index))
            final_train, final_test = data_for_autoencoder()
        
            X_train = train_data_list_global
            y_train = final_train['new_label'].tolist()
            X_test = test_data_list_global
            y_test = final_test['new_label'].tolist()    
            
            X_train = np.asarray(X_train)
            X_test = np.asarray(X_test)
            
            # ******************************** Auto Encoder Model Bulding ***************************************
            print('----------------- Auto Encoder -----------------')
            input_dim = X_train.shape[1]
            encoding_dim = 14
            nb_epoch = 100
            batch_size = 2
            
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
            
            checkpointer = ModelCheckpoint(filepath="model_quora.h5",
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
                                validation_data=(X_test, X_test),
                                verbose=1,
                                callbacks=[checkpointer, tensorboard]).history
            '''
            plt.plot(history['loss'])
            plt.plot(history['val_loss'])
            plt.title('Model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper right')
            '''
            #Load the model for prediction
            autoencoder = load_model('model_quora.h5')
            predictions = autoencoder.predict(X_test)
            
            mse = np.mean(np.power(X_test - predictions, 2), axis=1)
            error_df = pd.DataFrame({'reconstruction_error': mse,
                                    'true_class': y_test})
            
            #error_df.describe()
            
            
            fpr, tpr, thresholds = roc_curve(error_df.true_class, error_df.reconstruction_error)
            roc_auc = auc(fpr, tpr)
            
            '''plt.title('Receiver Operating Characteristic')
            plt.plot(fpr, tpr, label='AUC = %0.4f'% roc_auc)
            plt.legend(loc='lower right')
            plt.plot([0,1],[0,1],'r--')
            plt.xlim([-0.001, 1])
            plt.ylim([0, 1.001])
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.show()
            '''
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
            '''fig, ax = plt.subplots()
            for name, group in groups:
                ax.plot(group.index, group.reconstruction_error, marker='o', ms=3.5, linestyle='',
                        label= "known" if name == 1 else "unknown")
            ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
            ax.legend()
            plt.title("Reconstruction error for different classes")
            plt.ylabel("Reconstruction error")
            plt.xlabel("Data point index")
            plt.show()'''
            
            LABELS = ["known", "unknown"]
            y_pred_autoencoder = [1 if e > threshold else 0 for e in error_df.reconstruction_error.values]
            conf_matrix = confusion_matrix(error_df.true_class, y_pred_autoencoder,[0,1])
            test_f1 = f1_score(error_df.true_class, y_pred_autoencoder, average='macro')  
            print('Test F1 Score: ', test_f1)
            accuracy = balanced_accuracy_score(error_df.true_class, y_pred_autoencoder)
            print("Test accuracy:", accuracy)
            results_global_list.append(round(test_f1,2))
            results_global_list.append(round(accuracy*100,1))
            '''plt.figure(figsize=(6,6))
            sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
            plt.title("Confusion matrix - Reuters - Autoencoder")
            plt.ylabel('True class')
            plt.xlabel('Predicted class')
            plt.show()
            print(classification_report(error_df.true_class, y_pred_autoencoder))'''
            
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
            filename = '/opt/notebooks/OCSVM_ISF_LOF_USE_Baselines/results_output/quora/'+embedding+'_'+group+'.txt'
            with open(filename, 'w') as f:
                for item in results_global_list:
                    f.write("%s\n" % str(item))
            print("total time taken this loop: ", time.time() - start)