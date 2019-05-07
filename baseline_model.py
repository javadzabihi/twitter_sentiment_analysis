
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the clean dataset
twitter_dataset = pd.read_csv('clean_tweeter_data.csv',index_col=0)

X = twitter_dataset.text
y = twitter_dataset.target
# Train, Validation, and Test set : 98%, 2%, 2%
X_train, X, y_train, y = train_test_split(X, y, test_size = 0.02, random_state = 101)
X_val, X_test, y_val, y_test = train_test_split(X, y, test_size = 0.5, random_state = 101)


def accuracy_summary(Pipeline, X_train, y_train, X_test, y_test):

    sentiment_fit = Pipeline.fit(X_train, y_train)
    y_pred = sentiment_fit.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print ("accuracy score: {0:.2f}%".format(accuracy*100))
    return accuracy

def train_test_and_evaluate(Pipeline, X_train, y_train, X_test, y_test):

    sentiment_fit = Pipeline.fit(X_train, y_train)
    y_pred = sentiment_fit.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_mat = np.array(confusion_matrix(y_test, y_pred, labels=[0,1]))
    confusion = pd.DataFrame(conf_mat, index=['negative', 'positive'],
                             columns=['predicted_negative','predicted_positive'])
    print ("accuracy score: {0:.2f}%".format(accuracy*100))
    print ("-"*80)
    print ("Confusion Matrix\n")
    print (confusion)
    print ("-"*80)
    print ("Classification Report\n")
    print (classification_report(y_test, y_pred, target_names=['negative','positive']))

def nfeature_accuracy_checker(vectorizer, n_features,classifier, stop_words=None,ngram_range=(1, 1)):

    result = []
    for n in n_features:
        vectorizer.set_params(stop_words=stop_words, max_features=n, ngram_range=ngram_range)
        checker_pipeline = Pipeline([('vectorizer', vectorizer), ('classifier', classifier)])
        print("Validation result for {} features".format(n))
        nfeature_accuracy = accuracy_summary(checker_pipeline, X_train, y_train, X_val, y_val)
        result.append((n, nfeature_accuracy))
    return result


# Feature Extraction
count_vec = CountVectorizer()
classifier = LogisticRegression()
n_features = np.arange(10000,100001,10000)

term_freq_df = pd.read_csv('term_freq_df.csv',index_col=0)

my_stop_words = frozenset(list(term_freq_df.sort_values(by='total', ascending=False).iloc[:10].index))

# Unigram Without Stop Words
feature_result_wosw = nfeature_accuracy_checker(count_vec, n_features,classifier,stop_words='english')
# Unigram With Stop Words
feature_result_ug = nfeature_accuracy_checker(count_vec, n_features,classifier)
# Unigram Without Custom Stop Words
feature_result_wocsw = nfeature_accuracy_checker(count_vec, n_features,classifier,stop_words=my_stop_words)
# Accuracy Plot
plt.figure(figsize=(8, 4))
sns.set(font_scale=1)
sns.set_style('whitegrid')
with_stop_words = pd.DataFrame(feature_result_ug,columns=['nfeatures','validation_accuracy'])
without_custom_stop_words = pd.DataFrame(feature_result_wocsw,columns=['nfeatures','validation_accuracy'])
without_stop_words = pd.DataFrame(feature_result_wosw,columns=['nfeatures','validation_accuracy'])
plt.plot(with_stop_words.nfeatures, with_stop_words.validation_accuracy, label='with stop words')
plt.plot(without_custom_stop_words.nfeatures, without_custom_stop_words.validation_accuracy,label='without custom stop words')
plt.plot(without_stop_words.nfeatures, without_stop_words.validation_accuracy,label='without stop words')
#plt.title("Without stop words VS With stop words (Unigram): Accuracy")
plt.xlabel("Number of features")
plt.ylabel("Validation set accuracy")
plt.legend(loc=0)
plt.tight_layout()
plt.savefig('unigram_countvectorizer.png')
plt.show()

# Bigram With Sop Words
feature_result_bg = nfeature_accuracy_checker(count_vec, n_features,classifier,ngram_range=(1, 2))
# Trigram With Stop Words
feature_result_tg = nfeature_accuracy_checker(count_vec, n_features,classifier,ngram_range=(1, 3))
# Accuracy Plot
plt.figure(figsize=(8, 4))
sns.set(font_scale=1)
sns.set_style('whitegrid')
nfeatures_plot_tg = pd.DataFrame(feature_result_tg,columns=['nfeatures','validation_accuracy'])
nfeatures_plot_bg = pd.DataFrame(feature_result_bg,columns=['nfeatures','validation_accuracy'])
nfeatures_plot_ug = pd.DataFrame(feature_result_ug,columns=['nfeatures','validation_accuracy'])
plt.plot(nfeatures_plot_tg.nfeatures, nfeatures_plot_tg.validation_accuracy,label='trigram')
plt.plot(nfeatures_plot_bg.nfeatures, nfeatures_plot_bg.validation_accuracy,label='bigram')
plt.plot(nfeatures_plot_ug.nfeatures, nfeatures_plot_ug.validation_accuracy, label='unigram')
#plt.title("N-gram(1~3) test result : Accuracy")
plt.xlabel("Number of features")
plt.ylabel("Validation set accuracy")
plt.legend(loc=0)
plt.tight_layout()
plt.savefig('uni_bi_tri_countvectorizer.png')
plt.show()


# Classification metrics
tg_count_vec = CountVectorizer(max_features=60000,ngram_range=(1, 3))
tg_pipeline = Pipeline([('vectorizer', tg_count_vec),('classifier', classifier)])
train_test_and_evaluate(tg_pipeline, X_train, y_train, X_val, y_val)

# TF-IDF
tf_idf = TfidfVectorizer()
feature_result_ugt = nfeature_accuracy_checker(tf_idf, n_features,classifier)
feature_result_bgt = nfeature_accuracy_checker(tf_idf, n_features, classifier, ngram_range=(1, 2))
feature_result_tgt = nfeature_accuracy_checker(tf_idf, n_features, classifier, ngram_range=(1, 3))
# Accuracy plot
plt.figure(figsize=(8, 4))
sns.set(font_scale=1)
sns.set_style('whitegrid')
nfeatures_plot_tgt = pd.DataFrame(feature_result_tgt,columns=['nfeatures','validation_accuracy'])
nfeatures_plot_bgt = pd.DataFrame(feature_result_bgt,columns=['nfeatures','validation_accuracy'])
nfeatures_plot_ugt = pd.DataFrame(feature_result_ugt,columns=['nfeatures','validation_accuracy'])
plt.plot(nfeatures_plot_tgt.nfeatures, nfeatures_plot_tgt.validation_accuracy,label='trigram tfidf vectorizer',color='royalblue')
plt.plot(nfeatures_plot_tg.nfeatures, nfeatures_plot_tg.validation_accuracy,label='trigram count vectorizer',linestyle=':', color='royalblue')
plt.plot(nfeatures_plot_bgt.nfeatures, nfeatures_plot_bgt.validation_accuracy,label='bigram tfidf vectorizer',color='orangered')
plt.plot(nfeatures_plot_bg.nfeatures, nfeatures_plot_bg.validation_accuracy,label='bigram count vectorizer',linestyle=':',color='orangered')
plt.plot(nfeatures_plot_ugt.nfeatures, nfeatures_plot_ugt.validation_accuracy, label='unigram tfidf vectorizer',color='gold')
plt.plot(nfeatures_plot_ug.nfeatures, nfeatures_plot_ug.validation_accuracy, label='unigram count vectorizer',linestyle=':',color='gold')
#plt.title("N-gram(1~3) test result : Accuracy")
plt.xlabel("Number of features")
plt.ylabel("Validation set accuracy")
plt.legend(loc=0)
plt.tight_layout()
plt.savefig('tfidf_countvectorizer.png')
plt.show()
# Classification metrics
tg_tfidf_vec = TfidfVectorizer(max_features=100000,ngram_range=(1, 3))
tg_pipeline = Pipeline([('vectorizer', tg_tfidf_vec),('classifier', classifier)])
train_test_and_evaluate(tg_pipeline, X_train, y_train, X_val, y_val)

