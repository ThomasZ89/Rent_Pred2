import pandas as pd
from pandas_profiling import ProfileReport

from functions import features

features2 = features+["gesamtmiete"]

filtered_df = pd.read_csv(r'filtered_df.csv')
filtered_df.PLZ = filtered_df.PLZ.astype(str)
filtered_df = filtered_df[features2]
filtered_df = filtered_df.reindex(sorted(filtered_df.columns), axis=1)
profile = ProfileReport(filtered_df, title="Pandas Profiling Report")
profile.to_file("templates/Feature_Exploration.html")
