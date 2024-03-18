import pandas as pd
import numpy as np
from numpy import save

from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


def preprocess_data():
    print('preprocessing data')

    df = pd.read_csv('raw_data.csv')

    df.dropna(subset=['RainTomorrow'], inplace=True)

    df["TodayRain"] = pd.Series(np.where(df.RainToday.values == 'Yes', 1, 0), df.index)
    df["target"] = pd.Series(np.where(df.RainTomorrow.values == 'Yes', 1, 0), df.index)

    df["Timestamp"] = pd.to_datetime(df.Date)
    df["year"] = df.Timestamp.dt.year
    df["month"] = df.Timestamp.dt.month
    df["dayofyear"] = df.Timestamp.dt.dayofyear
    df["weekofyear"] = df.Timestamp.dt.isocalendar().week

    df = df.drop(columns=["RainToday", "RainTomorrow", "Date", "Timestamp"])

    temp = df.copy()
    for cat_col in df.columns:
        if df[f"{cat_col}"].dtype == "object":
            encoder = OneHotEncoder()
            one_hot_array = encoder.fit_transform(temp[[f'{cat_col}']]).toarray()

            # create new dataframe from numpy array
            one_hot_df = pd.DataFrame(one_hot_array, columns=encoder.get_feature_names_out())

            df = pd.concat([df, one_hot_df], ignore_index=False, sort=False, axis=1)
            df = df.reindex(temp.index)
            df = df.drop(columns=[f"{cat_col}"])

    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    df = pd.DataFrame(imp_mean.fit_transform(df), columns = df.columns)

    df = df.sample(frac=1)
    
    print(df["target"].value_counts())
    
    rain_df = df[df["target"] == 1]
    no_rain_df  = df[df["target"] == 0]

    no_rain_df_downsampled = resample(no_rain_df,
                 replace=True,
                 n_samples=len(rain_df),
                 random_state=42)
    
    df_downsampled = pd.concat([no_rain_df_downsampled, rain_df])

    print(df_downsampled["target"].value_counts())

    
    X_train, X_eval, y_train, y_eval = train_test_split(df_downsampled.drop(columns=["target"]), df_downsampled["target"], test_size=0.2, random_state=42)

    save("X_train.npy", X_train)
    save("y_train.npy", y_train)
    save("X_eval.npy", X_eval)
    save("y_eval.npy", y_eval)


if __name__ == '__main__':
    preprocess_data()