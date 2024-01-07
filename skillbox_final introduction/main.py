import dill

from func import *
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

# Загрузил df_hits
# Нужно выделить target из этой таблицы. Чтобы обучать модель
df_hits = pd.read_csv('data/ga_hits.csv')
# Добавил столбец target_action 1 - целевое действие - иначе 0
df_hits['target'] = df_hits.event_action.apply(lambda x: 1 if x in target_action else 0)

# В result обавляю 2 столбца ['session_id', 'target'],
# за одну сессию может быть несколько целевых действий
# Если в сессии было целевое действие, то в target 1, иначе 0
result = df_hits[['session_id', 'target']].groupby(by='session_id').sum()
result.target = result.target.apply(lambda x: 1 if x > 0 else 0)
result = result.reset_index()

# Загрузил df_sessions
df_sessions = pd.read_csv('data/ga_sessions.csv', low_memory=False)

# Объединяются таблицы. Удаляются строки, в которых нет target
data = df_sessions.merge(result, how = 'left', on='session_id')
data = data[data.target.notna()]


# Делим данные на входы и выходы
X = data.drop('target', axis=1)
y = data['target']

# Выбираем числовые и категориальные колонки
numerical_features = make_column_selector(dtype_include=['int64', 'float64'])
categorical_features = make_column_selector(dtype_include=object)

# стандартизуем числовые данные
numerical_transformer = Pipeline(steps=[
  #  ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# преобразование категориальных данных в OneHotEncoder
categorical_transformer = Pipeline(steps=[
  #  ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Запихиваем преобразования в один колумнтранформер
column_transformer = ColumnTransformer(transformers=[
    ('numerical', numerical_transformer, numerical_features),
    ('categorical', categorical_transformer, categorical_features)
])

# Pipeline.
# Заполняем пустоты
# Экран бьем на 2 колонки
# удаляем все ненужное
# и преобразум колонки
preprocessor = Pipeline(steps=[
    ('fill_na', FunctionTransformer(df_fill_na)),
    ('screen_size', FunctionTransformer(short_screen)),
    ('drop_columns', FunctionTransformer(filter_data)),
    ('column_trnsformer', column_transformer)
])

pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=20, max_depth=50, n_jobs=-1))
    ])

pipe.fit(X,y)

print(roc_auc_score(y, pipe.predict_proba(X)[:,1]))
print(confusion_matrix(y, pipe.predict(X)))

with open('model.pkl', 'wb') as file:
    dill.dump(pipe, file)



