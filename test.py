import pandas as pd
x=0
if x==1:
    PLZ_coeff = pd.read_csv("static/housing_text.csv")
    PLZ_coeff.to_excel('static/housing_text.xlsx')
    PLZ_coeff = pd.read_excel('static/housing_text.xlsx')
    print(PLZ_coeff)


    filtered_df = pd.read_csv("static/filtered_df.csv")
    PLZ_coeff = pd.read_csv("static/filtered_df.csv")
    PLZ_coeff.to_excel('static/filtered_df.xlsx')


    filtered_df = pd.read_csv("static/svm_pred.csv")
    PLZ_coeff = pd.read_csv("static/svm_pred.csv")
    PLZ_coeff.to_excel('static/svm_pred.xlsx')
