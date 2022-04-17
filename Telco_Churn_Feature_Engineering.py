import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns
def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")
def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
#########################################################
# Görev 1 : Keşifçi Veri Analizi
#########################################################

# Adım 1 : Genel Resmi İnceleyiniz. (Başka şeylere de bakılabilir.)

def load():
    data = pd.read_csv("datasets/Telco-Customer-Churn.csv")
    return data

df = load()
df.head()
df.shape
df.describe().T
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"],errors="coerce")
# errors="coerce" ile verideki boş değerler NaN değerlere çevrilmiştir.
df.dtypes

# Adım 2 : Numerik ve kategorik değişkenleri yakalayınız.

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Adım 3 : Numerik ve kategorik değişkenlerin analizini yapınız.

df[num_cols].describe().T

for col in cat_cols:
    cat_summary(df, col)

# Adım 4 : Hedef değişken analizi yapınız. (Kategorik değişkenlere göre hedef değişkenin ortalaması, hedef değişkene göre
# numerik değişkenlerin ortalaması)

df.groupby("Churn")[num_cols].mean()
# Kategorik değişkenler için işlem Adım 3'te yapılmıştır.

# Adım 5 : Aykırı gözlem analizi yapınız.

for col in num_cols:
    print(col, check_outlier(df, col)) # Aykırı gözlem bulunmamaktadır.

# Adım 6 : Eksik gözlem analizi yapınız.

missing_values_table(df)
df["tenure"][df[df["TotalCharges"].isnull()].index]
df["Churn"][df[df["TotalCharges"].isnull()].index]
# 11 TotalCharges değeri eksik bulunmuştur. Bu index'teki tenure(Müşterinin şirkette kaldığı ay sayısı)
# değeri 0, ve Churn değerlerine bakıldığında hepsinin "No" olduğu görülmüştür, muhtemelen bu kişiler
# yeni müşteridir. Bu sebepten TotalCharges değişkeninin NaN değerleri 0'a çevrilecektir.(Görev 2 Adım 1)

# Adım 7 : Korelasyon analizi yapınız.

corr_matrix = df.corr() # TotalCharges ile tenure arasında yüksek korelasyon saptanmıştır.

#########################################################
# Görev 2 : Feature Engineering
#########################################################

# Adım 1 : Eksik ve aykırı değerler için gerekli işlemleri yapınız.

df["TotalCharges"].fillna(0, inplace=True)
missing_values_table(df)
# Aykırı değer bulunmadığı için herhangi bir işlem yapılmamıştır.

# Adım 2 : Yeni değişkenler oluşturunuz.

df.loc[(df["PhoneService"] == "Yes") &
       (df["InternetService"] != "No") &
       (df["StreamingTV"] == "Yes") &
       (df["StreamingMovies"] == "Yes"),
       ["New_Using_All_Services"]] = "Yes"
df["New_Using_All_Services"].fillna("No", inplace=True)

df.loc[(df["OnlineSecurity"] == "Yes") &
       (df["OnlineBackup"] == "Yes") &
       (df["DeviceProtection"] == "Yes"),
       ["New_Safe_Customer"]] = "Yes"
df["New_Safe_Customer"].fillna("No",inplace=True)

# Adım 3 : Encoding işlemlerini gerçekleştiriniz.

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]
for col in binary_cols:
    label_encoder(df, col)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
df = one_hot_encoder(df, ohe_cols)
df.head()
num_cols = [col for col in num_cols if "customerID" not in col]

# Adım 4 : Numerik değişkenler için standartlaştırma yapınız.

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df[num_cols].head()
df.head()
# Adım 5 : Model oluşturunuz.

y = df["Churn"]
X = df.drop(["customerID", "Churn"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)

