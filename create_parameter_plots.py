import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from statsmodels.formula.api import ols
import statsmodels.regression.linear_model as sm
from functions import plot_all_errorbars, plot_errorbars, features, cat_features, num_feat

plot = 1
if plot ==1:
    filtered_df = pd.read_csv(r'static/filtered_df.csv')
    filtered_df.PLZ = filtered_df.PLZ.astype(str)
    mean = filtered_df.gesamtmiete.mean()
    dataframes = {}


    # Plot ctaegorical features
    for feat in cat_features:
        formula = 'gesamtmiete ~' + feat + " -1"
        fit = sm.OLS.from_formula(formula, data = filtered_df).fit()
        param_df = pd.DataFrame(fit.params, columns=["parameter"])
        param_df["standard_error"] = pd.DataFrame(fit.bse, columns=["standard_error"])["standard_error"]
        param_df["parameter"] = param_df["parameter"]-mean
        param_df = param_df.sort_values(by=['parameter'], ascending=False)
        dataframes[feat] = param_df

    for df in dataframes:
        plot_errorbars(dataframes[df], df)

    bool_feats = filtered_df.select_dtypes(include=['int64']).max()
    bool_feats = list(bool_feats[bool_feats <= 1].index)
    formula = 'gesamtmiete ~' + " + ".join(bool_feats) + " -1"
    fit = sm.OLS.from_formula(formula, data = filtered_df).fit()
    param_df = pd.DataFrame(fit.params, columns=["parameter"])
    param_df["standard_error"] = pd.DataFrame(fit.bse, columns=["standard_error"])["standard_error"]
    param_df["parameter"] = param_df["parameter"]
    param_df = param_df.sort_values(by=['parameter'], ascending=False)

    # Plot boolean values
    plot_errorbars(param_df, "boolean_parameters")

    # Plot values for multivariate regression with all values
    formula = 'gesamtmiete ~' + " + ".join(features)
    fit = sm.OLS.from_formula(formula, data = filtered_df).fit()
    param_df = pd.DataFrame(fit.params, columns=["parameter"])
    param_df["standard_error"] = pd.DataFrame(fit.bse, columns=["standard_error"])["standard_error"]
    param_df["parameter"] = param_df["parameter"]
    param_df = param_df.sort_values(by=['parameter'], ascending=False)
    param_df = param_df[~param_df.index.str.contains("straße")]
    param_df = param_df[~param_df.index.str.contains("PLZ")]
    param_df = param_df[~param_df.index.str.contains("Intercept")]
    param_df = param_df[~param_df.index.str.contains("kohleofen")]

    plot_all_errorbars(param_df, "all")

    # Plot numerical features
    formula = 'gesamtmiete ~' + " + ".join(num_feat)
    fit = sm.OLS.from_formula(formula, data = filtered_df).fit()
    param_df = pd.DataFrame(fit.params, columns=["parameter"])
    param_df["standard_error"] = pd.DataFrame(fit.bse, columns=["standard_error"])["standard_error"]
    param_df["parameter"] = param_df["parameter"]
    param_df = param_df.sort_values(by=['parameter'], ascending=False)
    param_df = param_df[~param_df.index.str.contains("Intercept")]

    plot_all_errorbars(param_df, "num_feat")


    # Save parameters
    formula = "gesamtmiete ~ PLZ -1"
    fit = sm.OLS.from_formula(formula, data=filtered_df).fit()
    param_df = pd.DataFrame(fit.params, columns=["parameter"])
    param_df["PLZ"] = [item.replace("PLZ", '').replace('[', '').replace(']', '') for item in list(param_df.index)]
    param_df.to_csv("static/PLZ_param.csv")
