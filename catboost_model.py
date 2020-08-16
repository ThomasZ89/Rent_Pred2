from catboost import CatBoostRegressor
from catboost import Pool, cv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np
import random
import pickle
from functions import features
random.seed(30)

train_features = features + ["gesamtmiete"]
filtered_df = pd.read_csv(r'static/filtered_df.csv')

train_model = 0
if train_model == 1:
    boost_df = filtered_df.dropna(subset=train_features)
    X = boost_df[features]
    Y = pd.DataFrame(boost_df["gesamtmiete"])

    X_train, X_validation, y_train, y_validation = train_test_split(X, Y, train_size=0.75, random_state=1234)
    categorical_features_indices = np.where(X.dtypes != np.float)[0]
    train_pool = Pool(X_train, y_train, cat_features=categorical_features_indices)
    validate_pool = Pool(X_validation, y_validation, cat_features=categorical_features_indices)
    all_pool = Pool(X, Y, cat_features=categorical_features_indices)

    model=CatBoostRegressor(iterations=4000, depth=12, loss_function='RMSE', early_stopping_rounds=30)
    model.fit(X_train, y_train, cat_features=categorical_features_indices, eval_set=(X_validation, y_validation),
              plot=False, silent=True)
    pickle.dump(model, open('static/model.pkl', 'wb'))

    feature_importances = model.get_feature_importance(train_pool)
    feature_names = X_train.columns

    feat_df = pd.DataFrame(list(zip(feature_names, feature_importances)), columns=['Feature', 'Importance'])

    labels1 = ['zimmergröße', 'PLZ', 'm2_pro_pers', 'buildings', 'straße', 'beruf_wg', 'stock', 'car', 'bus', 'wohnung',
               'parkett', 'spülmaschine', 'studenten_wg', 'personen', 'status', 'fire', 'badewanne', 'bed',
               'ablösevereinbarung', 'rauchen', 'fliesen', 'keller', 'balkon', 'gemischt_wg', 'pvc', 'dielen',
               'haustiere', 'laminat', 'dusche', 'kabel', 'teppich', 'aufzug', 'fußbodenheizung', 'satellit',
               'frauen_wg', 'fahrradkeller', 'zweck_wg', 'keine_zweck_wg', 'terrasse', 'waschmaschine',
               'gartenmitbenutzung', 'garten', 'dauer', 'azubi_wg']
    more_info = ['How big is the room', 'What is the ZIP code', 'square meter per person', 'What kind of house is it',
                 'Streetname where the house is located', 'flat-sharing community for business people',
                 'Which house floor', 'How are the parking possibilities', 'How long is it to the next bus station',
                 'What is the size of the whole flat', 'Does it have a parquet floor', 'Does it have a dishwasher',
                 'flat-sharing community for students', 'Number of inhabitants',
                 'Is the advertisement currently active', 'What kind of heating system does the flat have',
                 'Does it have a bathtub', 'Is it furnished?', 'How much is the replacement agreement',
                 'Is smoking allowed', 'Does it have tiles', 'Does it have a cellar', 'Does it have a balcony',
                 'Is the flat-sharing community mixed', 'Does it have a PVC floor', 'Does it have floorboards',
                 'Are pets allowed', 'Does it have a laminate floor', 'Does it have a shower',
                 'Does it have a cable connection', 'Does it have a carpet', 'Does it have an elevator',
                 'Does it have underfloor heating', 'Does it have satellite reception',
                 'flat-sharing community for women', 'Cellar for bikes',
                 'flat-sharing community without a lot of social interaction',
                 'flat-sharing community with a lot of social interaction', 'Does it have a Terrace',
                 'Does it have a washing machine', 'Garden sharing', 'Garden', 'Duration of the rent',
                 'flat-sharing community for apprentices']

    info_df = pd.DataFrame(list(zip(labels1, more_info)), columns=['Feature', 'Meaning'])
    feat_df = feat_df.merge(info_df, left_on='Feature', right_on='Feature')
    feat_df.to_csv("static/feat_df.csv")
else:
    model = pickle.load(open('static/model.pkl', 'rb'))


eval_model = 0
if eval_model == 1:
    filtered_df = pd.read_csv(r'static/filtered_df.csv')
    X = filtered_df[train_features]
    Y = pd.DataFrame(filtered_df["gesamtmiete"])
    X_train, X_validation, y_train, y_validation = train_test_split(X, Y, train_size=0.75, random_state=1234)
    y_validation["y_pred"] = model.predict(X_validation)
    y_validation["y_pred"] = round(y_validation["y_pred"], 1)
    #filtered_df["pred"] = model.predict(X)
    print(mean_absolute_error(y_validation["gesamtmiete"], y_validation["y_pred"]))
    print(model.get_best_score())
    y_validation[["gesamtmiete", "y_pred"]].to_csv("static/scatter.csv",  header=["gesamtmiete", "pred"])

