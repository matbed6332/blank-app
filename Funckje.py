from sklearn.metrics import root_mean_squared_error,mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def percent_missing(df):
    percent_nan = 100 * df.isnull().sum() / len(df)
    percent_nan = percent_nan[percent_nan > 0].sort_values()

    return percent_nan

def regresive_model_split(df,X):
    '''
    Wczesniej musi byc posortowane Y
    X to co ktory zwiazek ma trafic do testu
    '''
    split=[]
    for i in range(len(df)):
        if i == 0 or i == len(df)-1:
            split.append('Train'),
        
        elif i % X ==0:  ## dajemy co ktory ma byc w test. Teraz jest co 3
            split.append('Test'),
        
        else:
            split.append('Train')

    df['split'] = split

    return df

def run_model(model,X_train,y_train,X_test,y_test):
    '''
    Do importu
    from sklearn.metrics import root_mean_squared_error,mean_absolute_error
    '''
    model.fit(X_train,y_train)

    y_test_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)

    results = {"RMSE_train": root_mean_squared_error(y_train, y_train_pred),
        "MAE_train": mean_absolute_error(y_train, y_train_pred),
        "RMSE_test": root_mean_squared_error(y_test, y_test_pred),
        "MAE_test": mean_absolute_error(y_test, y_test_pred)}

    return model, results, y_test_pred, y_train_pred