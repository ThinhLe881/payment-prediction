import pandas as pd
import numpy as np


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def preprocessing(infile, outfile):
    # Import
    df = pd.read_table(infile)

    # Drop irrelevant cols
    df = df.drop(columns=['ANUMMER_01', 'ANUMMER_02', 'ANUMMER_03',
                 'ANUMMER_04', 'ANUMMER_05', 'ANUMMER_06', 'ANUMMER_07', 'ANUMMER_08', 'ANUMMER_09', 'ANUMMER_10'])

    # Fill out missing value aka '?'
    df['B_BIRTHDATE'] = np.where((df['B_BIRTHDATE'] == '?'), 0, 1)

    df['Z_CARD_ART'] = df['Z_CARD_ART'].replace({'?': 'unknown'})

    df['Z_LAST_NAME'] = df['Z_LAST_NAME'].replace({'?': 'unknown'})

    df['TIME_ORDER'] = df['TIME_ORDER'].replace(
        {'?': df['TIME_ORDER'].mode()[0]})
    df['TIME_ORDER'] = df['TIME_ORDER'].apply(lambda data: int(
        data.split(':')[0]) + int(data.split(':', 1)[1]) / 60)

    df['MAHN_AKT'] = df['MAHN_AKT'].replace({'?': -1})
    df['MAHN_AKT'] = pd.to_numeric(df['MAHN_AKT'])

    df['MAHN_HOECHST'] = df['MAHN_HOECHST'].replace({'?': -1})
    df['MAHN_HOECHST'] = pd.to_numeric(df['MAHN_HOECHST'])

    df['DATE_LORDER'] = np.where((df['DATE_LORDER'] == '?'), 0, 1)

    # One hot encode categorical data cols
    Z_METHODE = pd.get_dummies(df['Z_METHODE']).add_prefix('Z_METHODE_')
    df = df.join(Z_METHODE)
    df = df.drop('Z_METHODE', axis=1)

    Z_CARD_ART = pd.get_dummies(df['Z_CARD_ART']).add_prefix('Z_CARD_ART_')
    df = df.join(Z_CARD_ART)
    df = df.drop('Z_CARD_ART', axis=1)

    Z_LAST_NAME = pd.get_dummies(df['Z_LAST_NAME']).add_prefix('Z_LAST_NAME_')
    df = df.join(Z_LAST_NAME)
    df = df.drop('Z_LAST_NAME', axis=1)

    WEEKDAY_ORDER = pd.get_dummies(
        df['WEEKDAY_ORDER']).add_prefix('WEEKDAY_ORDER_')
    df = df.join(WEEKDAY_ORDER)
    df = df.drop('WEEKDAY_ORDER', axis=1)

    # Encode categorical data cols
    for col in df:
        if col == 'CLASS':
            continue
        df = df.replace({col: {'yes': 1, 'no': 0}})

    # Normalize continuous data cols
    for col in df:
        if col in ['VALUE_ORDER', 'TIME_ORDER', 'AMOUNT_ORDER', 'Z_CARD_VALID', 'SESSION_TIME', 'AMOUNT_ORDER_PRE',
                   'VALUE_ORDER_PRE', 'MAHN_AKT', 'MAHN_HOECHST']:
            df[col] = NormalizeData(df[col])

    # Move CLASS col to the last
    if 'CLASS' in df.columns:
        df['CLASS'] = df.pop('CLASS')

    # Export
    df.to_csv(outfile, index=False)


preprocessing('risk-train.txt', 'risk-train-preprocessed.csv')
