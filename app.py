import matplotlib
matplotlib.use('Agg')
from flask import Flask, render_template, request
from functions import link_to_pandas, catboost_predict, get_text, features, save_shap_plot
from SVM import predict_prob
import pandas as pd
import numpy as np
import pickle
from svr import svr_predict_rent

app = Flask(__name__)


#df_saved = pd.read_excel('static/filtered_df.xlsx')
df_saved = pd.read_csv('static/filtered_df.csv')
feat_df = pd.read_csv(r"static/feat_df.csv").sort_values(by=['Importance'], ascending=False).round(2)
labels1 = list(feat_df["Feature"])
values1 = list(feat_df["Importance"])
feat_legend = list(feat_df["Meaning"])

labels = ['0€-390€', '390€-450€', '450€-511€', '511€-600€', '600€-1120€']
max1 = max(values1) * 1.1
labels2 = ["gesamtmiete"] + labels1
labels3 = ["Real rent", "Rent prediction based on metadata"]

PLZ_coeff = pd.read_csv("static/PLZ_param.csv")
filtered_df = pd.read_csv("static/filtered_df.csv")

plz_list = list(filtered_df.PLZ.value_counts().index)
building_list = list(filtered_df.buildings.value_counts().index)
straße_list = list(filtered_df.straße.value_counts().index)
bed_list = list(filtered_df.bed.value_counts().index)
car_list = list(filtered_df.car.value_counts().index)
fire_list = list(filtered_df.fire.value_counts().index)
status_list = list(filtered_df.status.value_counts().index)
rauchen_list = list(filtered_df.rauchen.value_counts().index)
stock_list = list(filtered_df.stock.value_counts().index)
bool_list = [0, 1]

info_df = pd.DataFrame(list(zip(labels1, feat_legend)), columns=['Feature', 'Meaning']).sort_values(by=['Feature'])


@app.route('/')
def home():
    return render_template("index.html", len=len(filtered_df))


@app.route('/feature_overview')
def test():
    return render_template("feature_overview.html",  table1=[info_df.to_html(classes='male', index=False)])


@app.route('/Feature_Exploration')
def test1():
    return render_template("Feature_Exploration.html")


@app.route('/map')
def map():
    return render_template('map.html')


@app.route('/coefficient_overview')
def index():
    return render_template("coefficient_overview.html", table1=[info_df.to_html(classes='male', index=False)],
                           labels1=labels1, values1=values1, max1=max1, legend=feat_legend
                           )


@app.route('/predict', methods=['POST', "GET"])
def predict():
    return render_template('predict.html')


@app.route('/predict2', methods=['POST', "GET"])
def predict2():
    link = request.form["predict"]
    df = link_to_pandas(link, df_saved=df_saved)
    X = df[features]
    model = pickle.load(open('static/model.pkl', 'rb'))
    save_shap_plot(model, X, "shap_values")
    prediction = round(np.asscalar(catboost_predict(dataframe=df)), 2)
    pred_values = [float(df.gesamtmiete.iloc[0]), float(prediction)]
    text = get_text(link)
    rank_pred = predict_prob(text)
    bar_values = [round(num, 2) for num in list(rank_pred)]
    max_value = max(bar_values) * 1.1
    max2 = max(pred_values) * 1.1
    plz_var = df["PLZ"].values[0]
    zimmer_var = df["zimmergröße"].values[0]
    building_var = df["buildings"].values[0]
    person_var = df["personen"].values[0]
    straße_var = df["straße"].values[0]
    bed_var = df["bed"].values[0]
    car_var = df["car"].values[0]
    fire_var = df["fire"].values[0]
    status_var = df["status"].values[0]
    rauchen_var = df["rauchen"].values[0]
    wohnung_var = df["wohnung"].values[0]
    bus_var = df["bus"].values[0]
    abloese_var = df["ablösevereinbarung"].values[0]
    dauer_var = df["dauer"].values[0]
    stock_var = df["stock"].values[0]
    beruf_wg = df["beruf_wg"].values[0]
    studenten_wg = df["studenten_wg"].values[0]
    spuelmaschine = df["spülmaschine"].values[0]
    parkett = df["parkett"].values[0]
    keller = df["keller"].values[0]
    balkon = df["balkon"].values[0]
    aufzug = df["aufzug"].values[0]
    fliesen = df["fliesen"].values[0]
    fussbodenheizung = df["fußbodenheizung"].values[0]
    gemischt_wg = df["gemischt_wg"].values[0]
    waschmaschine = df["waschmaschine"].values[0]
    teppich = df["teppich"].values[0]
    kabel = df["kabel"].values[0]
    dusche = df["dusche"].values[0]
    pvc = df["pvc"].values[0]
    badewanne = df["badewanne"].values[0]
    zweck_wg = df["zweck_wg"].values[0]
    laminat = df["laminat"].values[0]
    garten = df["garten"].values[0]
    azubi_wg = df["azubi_wg"].values[0]
    satellit = df["satellit"].values[0]
    terrasse = df["terrasse"].values[0]
    dielen = df["dielen"].values[0]
    frauen_wg = df["frauen_wg"].values[0]
    keine_zweck_wg = df["keine_zweck_wg"].values[0]
    fahrradkeller = df["fahrradkeller"].values[0]
    gartenmitbenutzung = df["gartenmitbenutzung"].values[0]
    haustiere = df["haustiere"].values[0]
    X.to_csv("static/scrape_df.csv", index=False)
    return render_template('predict2.html', titles=link, title1="test",
                           max=max_value, labels=labels, values=bar_values, pred_values=pred_values,
                           labels3=labels3, max2=max2, tables=[df[labels2].to_html(classes='male', index=False)],
                           plz_list=plz_list, plz_var=plz_var, zimmer_var=zimmer_var,
                           building_list=building_list, building_var=building_var, person_var=person_var,
                           bed_var=bed_var, bed_list=bed_list, car_var=car_var, car_list=car_list,
                           fire_var=fire_var, fire_list=fire_list, status_var=status_var, status_list=status_list,
                           rauchen_var=rauchen_var, rauchen_list=rauchen_list, wohnung_var=wohnung_var,
                           bus_var=bus_var, ablöse_var=abloese_var, dauer_var=dauer_var, stock_list=stock_list,
                           stock_var=stock_var, bool_list=bool_list, beruf_wg=beruf_wg, studenten_wg=studenten_wg,
                           spuelmaschine=spuelmaschine, parkett=parkett, keller=keller, balkon=balkon, aufzug=aufzug,
                           fliesen=fliesen, fussbodenheizung=fussbodenheizung, gemischt_wg=gemischt_wg,
                           waschmaschine=waschmaschine, teppich=teppich, kabel=kabel, dusche=dusche, pvc=pvc,
                           badewanne=badewanne, zweck_wg=zweck_wg, laminat=laminat, garten=garten, azubi_wg=azubi_wg,
                           satellit=satellit, terrasse=terrasse, dielen=dielen, frauen_wg=frauen_wg,
                           keine_zweck_wg=keine_zweck_wg, fahrradkeller=fahrradkeller,
                           gartenmitbenutzung=gartenmitbenutzung, haustiere=haustiere, straße_var=straße_var,
                           straße_list=straße_list

                           )


@app.route('/predict3', methods=['POST', "GET"])
def predict3():
    old_df = pd.read_csv("static/scrape_df.csv")
    model = pickle.load(open('static/model.pkl', 'rb'))
    values = []
    columns = []
    for key, value in request.form.items():
        if value == "":
            value = 0
        values.append(value)
        columns.append(key)
    values = [values]
    df_input = pd.DataFrame(values, columns=columns)
    df_input["m2_pro_pers"] = 0.0
    for x in old_df.columns:
        df_input[x] = df_input[x].astype(old_df[x].dtypes.name)
    df_input["m2_pro_pers"] = df_input.wohnung / df_input.personen
    df_input = df_input[list(old_df.columns)]
    save_shap_plot(model, df_input, "shap_values_new")
    return render_template('predict3.html',
                           tables=[df_input.to_html(classes='male', index=False)],

                           old_tables=[old_df.to_html(classes='male', index=False)])


@app.route('/predict_text', methods=['POST', "GET"])
def predict_text():
    return render_template('predict_text.html')


@app.route('/predict_text1', methods=['POST', "GET"])
def predict_text1():
    link = request.form["predict_text"]
    text = get_text(link)
    rank_pred = predict_prob(text)
    bar_values = [round(num, 2) for num in list(rank_pred)]
    max_value = max(bar_values) * 1.1
    point_est = svr_predict_rent(text)
    return render_template('predict_text1.html', link=link, labels=labels, values=bar_values, max=max_value,
                           point_est=point_est)


@app.route('/predict_text2', methods=['POST', "GET"])
def predict_text2():
    text = request.form["predict_custom_text"]
    rank_pred = predict_prob(text)
    bar_values = [round(num, 2) for num in list(rank_pred)]
    max_value = max(bar_values) * 1.1
    point_est = svr_predict_rent(text)
    return render_template('predict_text2.html', labels=labels, values=bar_values, max=max_value, point_est=point_est)


@app.route('/models')
def models():
    scatter_data = pd.read_csv("static/scatter.csv")
    scatter_data["diff"] = scatter_data.gesamtmiete - scatter_data.pred
    diff1 = list(scatter_data["diff"])
    pred = round(scatter_data["pred"], 1)
    real = scatter_data["gesamtmiete"]
    newlist = []
    for h, w in zip(pred, real):
        newlist.append({'x': h, 'y': w})
    data = str(newlist).replace('\'', '')
    scatter_data = pd.read_csv("static/scatter_svr.csv")
    scatter_data["diff"] = scatter_data.gesamtmiete - scatter_data.pred
    diff = list(scatter_data["diff"])
    pred = round(scatter_data["pred"], 1)
    real = scatter_data["gesamtmiete"]
    newlist = []
    for h, w in zip(pred, real):
        newlist.append({'x': h, 'y': w})
    data1 = str(newlist).replace('\'', '')
    return render_template('models.html', data=data, data1=data1, diff=diff, diff1=diff1)


@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == "__main__":
    app.run(debug=True)
