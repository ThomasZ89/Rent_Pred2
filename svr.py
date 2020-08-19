import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import  mean_absolute_error
import pickle
import re




train = 0
save_pred = 0


german_stop_words = ['aber', 'alle', 'allem', 'allen', 'aller', 'alles', 'als', 'also', 'am', 'an', 'ander', 'andere', 'anderem', 'anderen', 'anderer', 'anderes', 'anderm', 'andern', 'anderr', 'anders', 'auch', 'auf', 'aus', 'bei', 'bin', 'bis', 'bist', 'da', 'damit', 'dann', 'der', 'den', 'des', 'dem', 'die', 'das', 'dass', 'daß', 'derselbe', 'derselben', 'denselben', 'desselben', 'demselben', 'dieselbe', 'dieselben', 'dasselbe', 'dazu', 'dein', 'deine', 'deinem', 'deinen', 'deiner', 'deines', 'denn', 'derer', 'dessen', 'dich', 'dir', 'du', 'dies', 'diese', 'diesem', 'diesen', 'dieser', 'dieses', 'doch', 'dort', 'durch', 'ein', 'eine', 'einem', 'einen', 'einer', 'eines', 'einig', 'einige', 'einigem', 'einigen', 'einiger', 'einiges', 'einmal', 'er', 'ihn', 'ihm', 'es', 'etwas', 'euer', 'eure', 'eurem', 'euren', 'eurer', 'eures', 'für', 'gegen', 'gewesen', 'hab', 'habe', 'haben', 'hat', 'hatte', 'hatten', 'hier', 'hin', 'hinter', 'ich', 'mich', 'mir', 'ihr', 'ihre', 'ihrem', 'ihren', 'ihrer', 'ihres', 'euch', 'im', 'in', 'indem', 'ins', 'ist', 'jede', 'jedem', 'jeden', 'jeder', 'jedes', 'jene', 'jenem', 'jenen', 'jener', 'jenes', 'jetzt', 'kann', 'kein', 'keine', 'keinem', 'keinen', 'keiner', 'keines', 'können', 'könnte', 'machen', 'man', 'manche', 'manchem', 'manchen', 'mancher', 'manches', 'mein', 'meine', 'meinem', 'meinen', 'meiner', 'meines', 'mit', 'muss', 'musste', 'nach', 'nicht', 'nichts', 'noch', 'nun', 'nur', 'ob', 'oder', 'ohne', 'sehr', 'sein', 'seine', 'seinem', 'seinen', 'seiner', 'seines', 'selbst', 'sich', 'sie', 'ihnen', 'sind', 'so', 'solche', 'solchem', 'solchen', 'solcher', 'solches', 'soll', 'sollte', 'sondern', 'sonst', 'über', 'um', 'und', 'uns', 'unsere', 'unserem', 'unseren', 'unser', 'unseres', 'unter', 'viel', 'vom', 'von', 'vor', 'während', 'war', 'waren', 'warst', 'was', 'weg', 'weil', 'weiter', 'welche', 'welchem', 'welchen', 'welcher', 'welches', 'wenn', 'werde', 'werden', 'wie', 'wieder', 'will', 'wir', 'wird', 'wirst', 'wo', 'wollen', 'wollte', 'würde', 'würden', 'zu', 'zum', 'zur', 'zwar', 'zwischen']

def remove_umlauts(tempVar):
    """
    Replace umlauts for a given text
    :param word: text as string
    :return: manipulated text as str
    """
    tempVar = tempVar.replace('ä', 'ae')
    tempVar = tempVar.replace('ö', 'oe')
    tempVar = tempVar.replace('ü', 'ue')
    tempVar = tempVar.replace('ä', 'ae')
    tempVar = tempVar.replace('ö', 'oe')
    tempVar = tempVar.replace('ü', 'ue')
    tempVar = tempVar.replace('ß', 'ss')
    return tempVar

def remove_stopwords(document):
    """
    Removes all stopwords from document.
    :param document: A document containing words.
    :return: The document without the stopwords
    """
    stopwords = german_stop_words
    words = document.split(" ")
    words_without_stopwords = [x.lower() for x in words if x not in stopwords]
    document_no_stopwords = ' '.join([str(x) for x in words_without_stopwords])
    return document_no_stopwords

def svr_predict_rent(text):
    model_pkl = open(r"static/SVR_rent_classifier.pkl", 'rb')
    model = pickle.load(model_pkl)
    model_pkl.close()

    transformer_count_pkl = open(r"static/SVR_count_transformer.pkl", "rb")
    count_transformer = pickle.load(transformer_count_pkl)
    transformer_count_pkl.close()

    transformer_tfidf_pkl = open(r"static/SVR_tfidf_transformer.pkl", "rb")
    tfidf_transformer = pickle.load(transformer_tfidf_pkl)
    transformer_tfidf_pkl.close()

    text = remove_stopwords(text)
    text = remove_umlauts(text)
    text = re.sub('[^a-zA-Z0-9 \n]', '', text)
    text_transf = tfidf_transformer.transform(count_transformer.transform(pd.Series(text)))

    return round(model.predict(text_transf)[0],2)


if train ==1:
    df = pd.read_csv(r"C:\Users\Thomas.Zoellinger\Documents\Jupyter Notebooks\housing_text.csv")
    # Remove special characters
    df["text"] = df["text"].str.replace('[^\w\s]', '')
    # Remove Stopwords
    df["text"] =df["text"].apply(lambda x: ' '.join([word for word in x.split() if word not in (german_stop_words)]))
    # Remove umlauts
    df["text"] = df["text"].apply(remove_umlauts)
    # Remove Stopwords again (maybe new ones from lemmatiziation)
    # df["text"] =df["text"].apply(lambda x: ' '.join([word for word in x.split() if word not in (german_stop_words)]))


    count_vect = CountVectorizer()
    X_counts = count_vect.fit_transform(df["text"])
    tfidf_transformer = TfidfTransformer()
    X_tfidf = tfidf_transformer.fit_transform(X_counts)

    Y = df["gesamtmiete"]
    # X_train, X_test, y_train, y_test = train_test_split(X_tfidf, Y, test_size=0.25, random_state=0)
    regr = SVR(kernel='linear', C=100, gamma='auto')
    regr.fit(X_tfidf, Y)

    # print(mean_absolute_error(y_test, y_pred))

    filename_model = 'static/SVR_rent_classifier.pkl'
    SVC_model_pickle = open(filename_model, 'wb')
    pickle.dump(regr, SVC_model_pickle)
    SVC_model_pickle.close()

    filename_count = 'static/SVR_count_transformer.pkl'
    count_pickle = open(filename_count, 'wb')
    pickle.dump(count_vect, count_pickle)
    count_pickle.close()

    filename_tfidf = 'static/SVR_tfidf_transformer.pkl'
    tfidf_pickle = open(filename_tfidf, 'wb')
    pickle.dump(tfidf_transformer, tfidf_pickle)
    tfidf_pickle.close()


if save_pred == 1:
    df = pd.read_csv(r"static/housing_text.csv")
    # Remove special characters
    df["text"] = df["text"].str.replace('[^\w\s]', '')
    # Remove Stopwords
    df["text"] =df["text"].apply(lambda x: ' '.join([word for word in x.split() if word not in (german_stop_words)]))
    # Remove umlauts
    df["text"] = df["text"].apply(remove_umlauts)

    count_vect = CountVectorizer()
    X_counts = count_vect.fit_transform(df["text"])
    tfidf_transformer = TfidfTransformer()
    X_tfidf = tfidf_transformer.fit_transform(X_counts)

    Y = df["gesamtmiete"]
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, Y, test_size=0.25, random_state=0)
    regr = SVR(kernel='linear', C=100, gamma='auto')
    regr.fit(X_train, y_train)
    y_test = pd.DataFrame(y_test)
    y_test.reset_index(drop=True, inplace=True)
    y_test["pred"] = pd.Series(regr.predict(X_test))
    y_test["diff"] = ((y_test["pred"]-y_test["gesamtmiete"])**2)**0.5
    print(mean_absolute_error(y_test["pred"], y_test["gesamtmiete"]))
    y_test[["gesamtmiete", "pred"]].to_csv("static/scatter_svr.csv",  header=["gesamtmiete", "pred"])
    y_test["diff"].to_csv("static/svr_error.csv", header=["diff"])


#t = "ZimmerWHO WE ARE:We are a start-up committed to solely hosting ambitious successful young professionals aged 21-35. We believe that hosting like-minded go-getters is essential for a pleasant flat atmosphere. With your relocation to Frankfurt to one of our luxurious co-living spaces, you will get direct access to an elite community of carefully selected individuals. You will instantly be integrated in Frankfurt’s social circle of bright rainmakers. Renting a room with us can be compared to joining a selective membership club. Ultimately, our mission is to provide aspiring personalities like you, who are newly drawn to Frankfurt, with prompt communal cohesion and moreover with an extensive network of thriving achievers. Not just within your own co-living space, but also beyond your flat. On a monthly basis, we organize events, where tenants from different co-living spaces come together to enjoy an exciting evening and are moreover given a platform to network and bond. These events can be in form of fancy restaurant dinners, exclusive catered house-parties, long weekend lunches or even personalized pub crawls. Of course, with pleasure, we carry all costs of our highly valued get-togethers. Regarding living comfort, we have established a new level of luxury in the co-living space segment. All our flats are equipped with the finest furniture highlighted by kingsized box-spring beds, modern fully fitted kitchens as well as all necessary appliances and utensils one might need in an apartment. Every flat of ours has been individually furnished with an eye for warmth and detail. Furthermore, we only choose prime locations for our apartments. Our shared flats can primarily be found in Frankfurt's most fancy and central residential area: Westend. With us, the rent you pay is an all-inclusive rent. No headaches, no subsequent payments or hidden costs. Everything is included: electricity, very fast 300 Mbit/s internet, water, heating and washing machine costs. You also don’t have to worry about any administrative TO DOs like city registration for example. We w"
#print(svr_predict_rent(t))