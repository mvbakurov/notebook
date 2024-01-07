"""
функции, с помощью которых преобразуется DataFrames
необходимые константы
"""

import pandas as pd



# Кортеж с целевыми действиями
target_action = (
    'sub_car_claim_click',
    'sub_car_claim_submit_click',
    'sub_open_dialog_click',
    'sub_custom_question_submit_click',
    'sub_call_number_click',
    'sub_callback_submit_click',
    'sub_submit_success',
    'sub_car_request_submit_click')


# Заполняет пустоты значением 'other'
def df_fill_na(df):
    df = df.copy()
    columns = [
        'utm_source',
        'utm_campaign',
        'device_brand',
        'utm_adcontent']
    df[columns] = df[columns].fillna('other')
    return df

#
def short_screen(df):
    df = df.copy()
    df = df[df.device_screen_resolution != '(not set)']
    df[['height', 'width']] = df.device_screen_resolution.str.split('x', expand=True)
    df.height = pd.to_numeric(df.height)
    df.width = pd.to_numeric(df.width)
    return df


# Удаляет столбцы в df_sessions
def filter_data(df):
    columns_to_drop = [
        'device_model',
        'utm_keyword',
        'device_os',
        'device_screen_resolution',
        'session_id',
        'client_id',
        'visit_date',
        'visit_time'
    ]
    return df.drop(columns_to_drop, axis=1)
