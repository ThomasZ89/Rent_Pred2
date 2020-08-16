import pandas as pd
from sklearn.metrics import plot_confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC, SVC, SVR
from sklearn import metrics
import pickle
import matplotlib.pyplot as plt


df = pd.read_csv(r"static/housing_text.csv")

regressor =0
train_new_model = 0
if train_new_model == 1:
    count_vect = CountVectorizer()
    X_counts = count_vect.fit_transform(df["text"])

    tfidf_transformer = TfidfTransformer()
    X_tfidf = tfidf_transformer.fit_transform(X_counts)
    if regressor == 1:
        Y = df["gesamtmiete"]
        X_train, X_test, y_train, y_test = train_test_split(X_counts, Y, test_size=0.3, random_state=0)
        regr = SVR(kernel='linear', C=100, gamma='auto')
        regr.fit(X_tfidf, Y)
    else:
        Y = df["rank"]

        clf_svc = SVC(kernel="linear", probability=True)
        clf_svc.fit(X_tfidf, Y)

        y_pred = clf_svc.predict(X_tfidf)

        filename_model = 'static/SVC_rent_classifier.pkl'
        SVC_model_pickle = open(filename_model, 'wb')
        pickle.dump(clf_svc, SVC_model_pickle)
        SVC_model_pickle.close()

        filename_count = 'static/count_transformer.pkl'
        count_pickle = open(filename_count, 'wb')
        pickle.dump(count_vect, count_pickle)
        count_pickle.close()

        filename_tfidf = 'static/tfidf_transformer.pkl'
        tfidf_pickle = open(filename_tfidf, 'wb')
        pickle.dump(tfidf_transformer, tfidf_pickle)
        tfidf_pickle.close()


model_pkl = open(r"static/SVC_rent_classifier.pkl", 'rb')
model = pickle.load(model_pkl)
model_pkl.close()

transformer_count_pkl = open(r"static/count_transformer.pkl", "rb")
count_transformer = pickle.load(transformer_count_pkl)
transformer_count_pkl.close()

transformer_tfidf_pkl = open(r"static/tfidf_transformer.pkl", "rb")
tfidf_transformer = pickle.load(transformer_tfidf_pkl)
transformer_tfidf_pkl.close()


def predict_prob(string):
    text_transf = tfidf_transformer.transform(count_transformer.transform([string]))
    proba = model.predict_proba(text_transf)[0]
    return proba[0], proba[1], proba[2], proba[3], proba[4]




