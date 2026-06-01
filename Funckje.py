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