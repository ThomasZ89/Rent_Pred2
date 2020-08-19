import matplotlib
matplotlib.use('Agg')
from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
import pickle
import re
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
import shap
import datetime
import os

features = ['wohnung', 'zimmergröße', 'personen', 'car', 'buildings', 'stock', 'bus', 'bed', 'fire', 'rauchen', 'zweck_wg', 'keine_zweck_wg', 'beruf_wg', 'gemischt_wg', 'studenten_wg', 'frauen_wg', 'azubi_wg', 'straße', 'aufzug', 'balkon', 'fahrradkeller', 'garten', 'gartenmitbenutzung', 'haustiere', 'keller', 'spülmaschine', 'terrasse', 'waschmaschine', 'dielen', 'fliesen', 'fußbodenheizung', 'laminat', 'parkett', 'pvc', 'teppich', 'badewanne', 'dusche', 'kabel', "ablösevereinbarung", 'satellit', 'status', 'dauer', 'PLZ', "m2_pro_pers"]
cat_features = ["car", "buildings", "bed","stock","fire", "PLZ"]
num_feat = ['wohnung', 'zimmergröße', 'personen', 'bus', 'ablösevereinbarung', 'dauer']
num_and_bool_feat = list(set(features) - set(cat_features))


def remove_old_shap():
    for filename in os.listdir("static/"):
        if filename.startswith('shap_values'):
            os.remove("static/"+filename)


def get_latest_link_shap():
    shap_files = []
    for filename in os.listdir("static/"):
        if filename.startswith('shap_values_link'):
            shap_files.append(filename)
    shap_files.sort(reverse=True)
    return shap_files[0]


def save_shap_plot(model, X, name=""):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
    shap.force_plot(explainer.expected_value, shap_values[0, :], X.iloc[0, :], matplotlib=True, show=False)
    number = str(int(datetime.datetime.utcnow().timestamp()))
    name = name + number+'.png'
    save_name = 'static/'+ name
    plt.savefig(save_name, bbox_inches='tight')
    return name

def plot_all_errorbars(df, name):
    x = df.index
    y1 = df.parameter
    yerr1 = df.standard_error
    fig, ax = plt.subplots()
    fig.set_figheight(len(x))
    plt.rc('axes', labelsize=22)
    plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.2)
    trans1 = Affine2D().translate(-0.1, 0.0) + ax.transData
    trans2 = Affine2D().translate(+0.1, 0.0) + ax.transData
    er1 = ax.errorbar(y1, x, xerr=yerr1, marker="o", linestyle="none", transform=trans1)
    ax.axvline(x=0, color="black")
    ax.set_ylim(-0.1, len(df) - 1 + 0.1)
    return plt.savefig('static/'+name + '.png', bbox_inches='tight')

def plot_errorbars(df, name):
    x = [item.replace(name, '').replace('[', '').replace(']', '') for item in list(df.index)]
    y1 = df.parameter
    yerr1 = df.standard_error
    fig, ax = plt.subplots()
    fig.set_figheight(len(x))
    plt.rc('axes', labelsize=22)
    plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.2)
    trans1 = Affine2D().translate(-0.1, 0.0) + ax.transData
    trans2 = Affine2D().translate(+0.1, 0.0) + ax.transData
    er1 = ax.errorbar(y1, x, xerr=yerr1, marker="o", linestyle="none", transform=trans1)
    ax.axvline(x=0, color="black")
    ax.set_ylim(-0.1, len(df) - 1 + 0.1)
    return plt.savefig('static/'+name + '.png', bbox_inches='tight')


def clean_string2(liste):
    liste = re.sub(r"[\W\_]|\d+", ' ', liste)
    liste = " ".join(liste.split())
    liste = liste.lower()
    return liste

def get_main_text(soup):
    text = soup.find(id="ad_description_text").text
    text = clean_string2(text)
    cut_string = "ihr wg gesucht team"
    try:
        return text.split(cut_string, 1)[1]
    except:
        return text

def get_text(link):
    bs = get_bs_from_http(link)
    text = get_main_text(bs)
    return clean_string2(text)

def get_bs_from_html(html):
    return BeautifulSoup(html.text, "html.parser")

def get_text_from_clean(text, liste, direction="right"):
    pairs = []

    if direction == "right":
        for item in liste:
            try:
                if item in text:
                    pairs.append([item, text.split(item)[1].split()[0]])
                else:
                    pairs.append([item, "none"])
            except:
                pairs.append([item, "none"])
    if direction == "left":
        for item in liste:
            try:
                if item in text:
                    pairs.append([item, text.split(item)[0].split()[-1]])
                else:
                    pairs.append([item, "none"])
            except:
                pairs.append([item, "none"])

    return pairs


def clean_string(liste):
    liste = Flatten(liste)
    liste = " ".join(liste)
    liste = " ".join(liste.split())
    return liste


def get_text_from_html(bs, class_name):
    string_list = []
    soup = bs.find_all(class_=class_name)
    for entry in soup:
        string_list.append(entry.text)
    return string_list


def Flatten(ul):
    fl = []
    for i in ul:
        if type(i) is list:
            fl += Flatten(i)
        else:
            fl += [i]
    return fl


def get_all_data_from_site(bs, link):
    names = ["Wohnung", "Zimmergröße", "Sonstige", "Nebenkosten", "Miete", "Gesamtmiete", "Kaution",
             "Ablösevereinbarung"]
    my_list = get_text_from_html(bs, "col-sm-12 hidden-xs")
    my_list = clean_string(my_list)
    dict1 = dict(get_text_from_clean(my_list, names, "left"))

    names = ["frei ab: ", "frei bis: "]
    my_list = get_text_from_html(bs, "col-sm-3")
    my_list = clean_string(my_list)
    dict2 = dict(get_text_from_clean(my_list, names, "right"))

    names = [" Zimmer in "]
    my_list = get_text_from_html(bs, "col-sm-6")
    my_list = clean_string(my_list)
    dict3 = dict(get_text_from_clean(my_list, names, "right"))

    names = ["Malmännliche", "weiblich", 'height="17"']
    count = []
    for name in names:
        try:
            string = str(bs.find(
                class_="mr5 detail-list-fav-button display-inline-block hidden-xs create_favourite").next_sibling.next_sibling)
            count.append(string.count(name))
        except:
            count.append("none")
    dict4 = dict(zip(names, count))

    my_list = get_text_from_html(bs, "ul-detailed-view-datasheet print_text_left")
    my_list = [x.strip() for x in my_list]
    try:
        dict5 = dict(get_text_from_clean(my_list[1], ["zwischen"], "left"))
    except:
        dict5 = dict(get_text_from_clean(my_list, ["zwischen"], "left"))

    my_list = get_text_from_html(bs, "ul-detailed-view-datasheet print_text_left")
    my_list = [x.strip() for x in my_list]
    try:
        dict8 = dict(get_text_from_clean(my_list[1], ["Geschlecht"], "right"))
    except:
        dict8 = dict(get_text_from_clean(my_list, ["Geschlecht"], "right"))

    item_list = ["glyphicons glyphicons-bath-bathtub noprint",
                 "glyphicons glyphicons-wifi-alt noprint",
                 "glyphicons glyphicons-car noprint",
                 "glyphicons glyphicons-fabric noprint",
                 "glyphicons glyphicons-display noprint",
                 "glyphicons glyphicons-folder-closed noprint",
                 "glyphicons glyphicons-mixed-buildings noprint",
                 "glyphicons glyphicons-building noprint",
                 "glyphicons glyphicons-bus noprint",
                 "glyphicons glyphicons-bed noprint",
                 "glyphicons glyphicons-fire noprint"]
    data_list = []
    for item in item_list:
        try:
            data_list.append([item[22:-8], clean_string([bs.find(class_=item).next_sibling.next_sibling.next_sibling])])
        except:
            data_list.append([item[22:-8], "none"])
    dict6 = dict(data_list)

    liste = get_text_from_html(bs, "col-sm-4 mb10")
    adress_string = clean_string(liste).replace("Adresse ", "").replace("Umzugsfirma beauftragen1", "").replace(
        "Umzugsfirma beauftragen 1", "")
    dict7 = {"Adresse": adress_string, "Link": link}

    names = "Miete pro Tag: "
    my_list = get_text_from_html(bs, "col-sm-5")
    my_list = clean_string(my_list)
    if names in my_list:
        dict9 = {"taeglich": 1}
    else:
        dict9 = {"taeglich": 0}

    div_id = 'popover-energy-certification'
    try:
        cs = clean_string([bs.find(id=div_id).next_sibling])
        dict10 = {"baujahr": cs}
    except:
        dict10 = {"baujahr": "none"}

    rauchen = "Rauchen nicht erwünscht"
    nichrauchen = "Rauchen überall erlaubt"
    my_list = get_text_from_html(bs, "col-sm-6")
    my_list = clean_string(my_list)
    if rauchen in my_list:
        dict11 = {"rauchen": "raucher"}
    if nichrauchen in my_list:
        dict11 = {"rauchen": "nichtraucher"}
    if rauchen not in my_list and nichrauchen not in my_list:
        dict11 = {"rauchen": "keine_Angabe"}

    wg_list = ["Zweck-WG", "keine Zweck-WG", "Berufstätigen-WG", "gemischte WG", "Studenten-WG", "Frauen-WG",
               "Azubi-WG"]
    dict12 = []
    for wg in wg_list:
        my_list = get_text_from_html(bs, "col-sm-6")
        my_list = clean_string(my_list)
        if wg in my_list:
            dict12.append([wg, 1])
        else:
            dict12.append([wg, 0])
    dict12 = dict(dict12)

    dict_list = [dict1, dict2, dict3, dict4, dict5, dict8, dict6, dict7, dict7, dict9, dict10, dict11, dict12]
    for item in dict_list:
        dict1.update(item)
    return dict1


def get_bs_from_html(html):
    return BeautifulSoup(html.text, "html.parser")


def get_bs_from_http(link):
    html = requests.get(link)
    return BeautifulSoup(html.text, "html.parser")


def get_html_request(link):
    return requests.get(link)

def merge_dicts(dic1, dic2):
    try:
        dic3 = dict(dic2)
        for k, v in dic1.items():
            dic3[k] = Flatten([dic3[k], v]) if k in dic3 else v
        return dic3
    except:
        return dic1


def replace_viertel(x, viertel_liste):
    if x in viertel_liste:
        return x
    elif any([i in x for i in viertel_liste]):
        return [i for (i, v) in zip(viertel_liste, [i in x for i in viertel_liste]) if v][0]
    else:
        return x

def link_to_pandas(full_link, df_saved):
    stem = full_link[:57]
    link = full_link[57:]
    bs = get_bs_from_http(stem + link)
    data = get_all_data_from_site(bs, link)
    df = pd.DataFrame([data], columns=data.keys())
    df = df.apply(lambda x: x.astype(str).str.lower())
    # rename cols
    df.columns = ['wohnung', 'zimmergröße', 'sonstige', 'nebenkosten', 'miete', 'gesamtmiete', 'kaution',
                  'ablösevereinbarung', 'frei_ab', 'frei_bis', 'personen', 'männlich', 'weiblich', 'insgesamt',
                  'geschlecht', 'geschlecht2', 'bath-bathtub', 'wifi-alt', 'car',
                  'fabric', 'display', 'folder-closed', 'buildings', 'stock', 'bus', 'bed',
                  'fire', 'adresse', 'link', 'taeglich', 'baujahr', 'rauchen', 'zweck_wg', 'keine_zweck_wg', 'beruf_wg',
                  'gemischt_wg', 'studenten_wg', 'frauen_wg', 'azubi_wg']
    # remove common words from cols
    remove_list = ["m²", "€", r"n.a", "none", "(", ")", ". og", "minuten zu fuß entfernt", "minute zu fuß entfernt"]
    for col in list(df.columns):
        for item in remove_list:
            df[col] = df[col].str.replace(item, "", regex=False)

    # remove individual words from col
    df["personen"] = df["personen"].str.replace("er", "")

    df["straße"] = df.adresse.str.extract(pat="(.*)\d\d\d\d\d")
    df["straße"] = df.straße.str.replace("str\.", "straße")
    df["straße"] = df.straße.str.replace("str ", "straße")
    df["straße"] = df.straße.str.replace("strasse", "straße")
    df["straße"] = df.straße.str.replace("[^\w\d]", "")
    df["straße"] = df.straße.str.replace("[0-9]+", "")
    df["straße"] = df.straße.str.replace("ß", "ss")
    df.loc[~df["straße"].isin(
        list(pd.DataFrame(df_saved.straße.value_counts()).query("straße > 10").index)), "straße"] = "no_info_or_rare"


    # Vermutlich ist möbliert, teilmöbliert = teilmöbliert
    df["bed"] = df.bed.str.replace("möbliert, teilmöbliert", "teilmöbliert")

    df["geschlecht"] = df["geschlecht"] + df["geschlecht2"]
    df["geschlecht"] = df["geschlecht"].str.replace("egalegal", "egal")
    # drop unused cols
    # ,"Adresse"
    df = df.drop(columns=["geschlecht2"])

    df["folder-closed"] = df["folder-closed"].str.replace("haustiere erlaubt", "haustiere")
    df["bath-bathtub"] = df["bath-bathtub"].str.replace("eigenes bad", "eigenes_bad")
    df["bath-bathtub"] = df["bath-bathtub"].str.replace("gäste wc", "gäste_wc")

    class_list = [
        "aufzug, balkon, fahrradkeller, garten, gartenmitbenutzung, haustiere, keller, spülmaschine, terrasse, waschmaschine",
        "dielen, fliesen, fußbodenheizung, laminat, parkett, pvc, teppich",
        "badewanne, badmitbenutzung, dusche, eigenes_bad, gäste_wc",
        "kabel, satellit"]

    df = pd.concat([df, df], ignore_index=True)

    one_hot_cols = ["folder-closed", 'fabric', 'bath-bathtub', "display"]
    for i, col in enumerate(one_hot_cols):
        df.iloc[1, df.columns.get_loc(col)] = class_list[i]
        df2 = df[col].str.get_dummies(', ')
        df = pd.concat([df, df2.reindex(df.index)], axis=1)
    df = df.head(1)

    df = df.set_index("link")

    df = df.drop(columns=one_hot_cols)
    df["viertel"] = df.index.to_series().astype(str).str.extract(pat="(.*)\.\d\d\d\d\d\d")
    df["viertel"] = df["viertel"].str.lower()
    repl_viertel = ["--", "\d\d\d\d\d", "franfurter ", "frankfurt am main", "franfurt-am-main,""frankfurt-main-",
                    "frankfurtnord", "frankfurt-", "frankfurt", "bei-frankfurt", "naehe", "sudlich-von",
                    "u-bahn-station-", "-bei-ffm", "1-minute-from-", "am-main"]
    for word in repl_viertel:
        df["viertel"] = df["viertel"].str.replace(word, "")
    df["viertel"] = df["viertel"].str.strip('-')

    conditions = [
        (df["frei_ab"] == ""),
        ((df["frei_ab"] != "") & (df["frei_bis"] != "")),
        ((df["frei_ab"] != "") & (df["frei_bis"] == ""))]
    choices = ['inaktiv', 'befristet', 'unbefristet']
    df["status"] = np.select(conditions, choices)

    df["dauer"] = pd.to_datetime(df.frei_bis, format='%d.%m.%Y', errors='coerce') - pd.to_datetime(df.frei_ab,
                                                                                                   format='%d.%m.%Y',
                                                                                                   errors='coerce')
    df['dauer'] = df['dauer'] / np.timedelta64(1, 'D')
    df['dauer'].fillna(0, inplace=True)

    df["wohnung"] = df["wohnung"].str.replace("\.", "")
    df["m2_pro_pers"] = pd.to_numeric(df['wohnung'], errors='coerce') / pd.to_numeric(df['personen'], errors='coerce')

    # Replace uncommon places with common places if they are included in common places
    df["viertel_name"] = df.viertel.apply(replace_viertel, viertel_liste=freq_viertel)
    df_mapped = df
    df_mapped["baujahr"] = df_mapped.baujahr.str.extract(pat="baujahr (\d\d\d\d)")
    df_mapped["PLZ"] = df_mapped.adresse.str.extract("(\d\d\d\d\d)")

   # Replace uncommon PLZ with new value
    df_mapped.PLZ = df_mapped.PLZ.astype(str)
    df_saved.PLZ = df_saved.PLZ.astype(str)
    df_mapped.loc[~df_mapped["PLZ"].isin(
        list(pd.DataFrame(df_saved.PLZ.value_counts()).query("PLZ > 10").index)), "PLZ"] = 99999

    num_cols = ['wohnung', 'zimmergröße', 'sonstige', 'nebenkosten', 'miete', 'gesamtmiete', 'bus', 'männlich',
                'personen', 'weiblich', 'kaution', 'ablösevereinbarung', 'personen', 'bus', 'baujahr', "taeglich"]
    for col in num_cols:
        df_mapped[col] = df_mapped[col].astype(str)
        df_mapped[col] = df_mapped[col].str.extract('(\d+)', expand=False)
        df_mapped[col] = df_mapped[col].astype(float)


    # Replace na's
    feat_list1 = ["buildings", "stock", "bed", "car", "fire", "straße", "PLZ"]
    # median
    feat_list2 = ["wohnung", "bus", "personen", "m2_pro_pers"]
    # null setzen
    feat_list3 = ["baujahr", "ablösevereinbarung"]

    for feat in feat_list1:
        df_mapped[feat].fillna("no_info_or_rare", inplace=True)
        df_mapped[feat] = df_mapped[feat].replace(r'^\s*$', "no_info_or_rare", regex=True)

    for feat in feat_list2:
        df_mapped[feat].fillna(df_saved[feat].median(), inplace=True)
        df_mapped[feat] = df_mapped[feat].replace(r'^\s*$', df_saved[feat].median(), regex=True)

    for feat in feat_list3:
        df_mapped[feat].fillna(0, inplace=True)
        df_mapped[feat] = df_mapped[feat].replace(r'^\s*$', 0, regex=True)

    return df_mapped


def catboost_predict(dataframe, features= features):
    model = pickle.load(open('static/model.pkl', 'rb'))
    dataframe = dataframe[features]
    return model.predict(dataframe)

freq_viertel = ['sachsenhausen',
 'bockenheim',
 'bornheim',
 'nordend-ost',
 'nordend-west',
 'ostend',
 'innenstadt',
 'niederrad',
 'westend-nord',
 'dornbusch',
 'gallusviertel',
 'gallus',
 'bahnhofsviertel',
 'westend-sud',
 'roedelheim',
 'hoechst',
 'eschersheim',
 'gutleutviertel',
 'griesheim',
 'oberrad',
 'ginnheim',
 'heddernheim',
 'eckenheim',
 'hausen',
 'preungesheim',
 'flughafen',
 'fechenheim',
 'altstadt',
 'nied',
 'bergen-enkheim',
 'nieder-eschbach',
 'bonames',
 'praunheim',
 'sossenheim',
 'niederursel',
 'nordend',
 'offenbach',
 'seckbach',
 'berkersheim',
 'kelsterbach',
 'unterliederbach',
 'sindlingen',
 'neu-isenburg',
 'schwanheim',
 'westend',
 'kalbach',
 'er-berg',
 'europaviertel',
 'zeilsheim',
 'harheim',
 'eschborn',
 'riederwald',
 'riedberg',
 'bad-vilbel',
 'goldstein',
 'raunheimflughafen',
 'nieder-erlenbach',
 'oberursel',
 'maintal',
 'raunheim',
 'moerfelden-walldorf',
 'kaiserlei',
 'nordweststadt',
 'langen',
 'sachsenhausen-nord',
 'eschborn-bei',
 'bad-homburg',
 'nordend-bornheim',
 'rodgau']
