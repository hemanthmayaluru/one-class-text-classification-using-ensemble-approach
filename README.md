# One-Class-Text-Classification-using-Ensemble-Models
One class text classification using an ensemble of models. 
This work is done as part of my Master Thesis. 

Traditional classification algorithms work in a closed-world scenario where the training data contains all existing classes. In contrast, open set classifiers can handle new input that does not belong to any of the classes seen during training. Open set classification has been studied intensively in the computer vision domain, primarily in handwriting recognition, face recognition, object classification and computer forensics. Here we are interested in open set classification in natural language processing in one class document classification. We propose a new system based on autoencoder for one class classification of documents leveraging the full text. Extending further, we propose a novel ensemble based classifier model, a combination of several basic classifiers, to detect if an incoming document belongs to the class known from training or an unknown class. We compare and evaluate our methods on existing one class classification datasets for NLP - 20 Newsgroups, reuters and webkb. We also extract and use a new full-text dataset from arxiv.org. Our methods significantly outperforms the current state-of-the-art approaches for one class document classification.


The methods include:

Autoencoder <br /> 
One Class SVM <br /> 
Isolation Forest <br /> 
Ensemble using Majority Voting <br /> 
Ensemble using Neural Network <br /> 


One can download the entire repo and run the .py files. Alternately one can run jupyter notebooks as well.

Before running the files, download the embeddings from the following URLs:

Universal Sentence Encoder: https://tfhub.dev/google/universal-sentence-encoder/1

Infersent: https://github.com/facebookresearch/InferSent

# Hardware Requirements

CPUs: 36 <br />  
RAM: 270GB <br />
OS: Debian GNU/Linux 9 <br />

# Software Requirements
Requirements include the following packages:

python 3.6 <br />
jupyter 1.0.0 <br />
scikit-learn 0.21.3 <br />
nltk 3.4 <br />
numpy 1.17.4 <br />
pandas 0.24.2 <br />
tensorflow 1.13.1 <br />
pytorch 1.3.0 <br />
keras 2.3.1 <br />
flair 0.4.4 <br />
sentence_transformers 0.2.3 <br />
pdfminer.six 20181108 <br />
beautifulsoup4 4.7.1


