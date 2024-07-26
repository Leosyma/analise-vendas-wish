# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 19:14:30 2024

@author: leoja
"""

#%% Bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#%% Leitura do arquivo
df_products = pd.read_csv(r'C:\Users\leoja\OneDrive\Documentos\Asimov Academy\Aulas\Trilha Data Science & Machine Learning\10 - Análise de vendas do marketplace Wish\Análise de Vendas de um Marketplace\summer-products-with-rating-and-performance_2020-08.csv')

# Colunas mais importantes
[i for i in df_products.columns]
cols = ['title',
 'price',
 'retail_price',
 'currency_buyer',
 'units_sold',
 'uses_ad_boosts',
 'rating',
 'rating_count',
 'badges_count',
 'badge_product_quality',
 'badge_fast_shipping',
 'tags',
 'product_color',
 'product_variation_size_id',
 'product_variation_inventory',
 'shipping_is_express',
 'countries_shipped_to',
 'inventory_total',
 'has_urgency_banner',
 'origin_country',
 'merchant_rating_count',
 'merchant_rating']

#%% Qualidade dos dados
df_products = df_products[cols]
df_products.info()
df_products.describe()

# Dados nulos
df_products.isna().sum()

# Tratamento dos dados nulos
df_products['product_color'] = df_products['product_color'].fillna('')
df_products['product_variation_size_id'] = df_products['product_variation_size_id'].fillna('')
df_products['has_urgency_banner'] = df_products['has_urgency_banner'].fillna(0)
df_products['origin_country'] = df_products['origin_country'].fillna('')

# Separar variaveis numéricas e categóricas
numerical_cols = df_products.describe().columns
categorial_cols = [i for i in df_products.columns if i not in numerical_cols]

#%% Análise exploratória
# Variáveis categóricas
for col in categorial_cols:
    if col not in ['title_origin','tags']:
        fig, ax = plt.subplots(figsize=(15,6))
        sns.countplot(data=df_products,x=col,order=df_products[col].value_counts().index)
        plt.xticks(rotation=90)
        plt.show()
        
# Variáveis numéricos
for col in numerical_cols:
    f, axes = plt.subplots(1,1,figsize=(18,4))
    sns.histplot(x=col, data=df_products)
    plt.xticks(rotation=90)
    plt.suptitle(col,fontsize=20)
    plt.show()

# Substituir unidades vendidas menores que 10
df_products.loc[df_products['units_sold'] < 10, 'units_sold'] = 10
df_products['units_sold'].value_counts()
df_products['units_sold'].median()
df_products['units_sold'].mean()
sns.distplot(df_products['units_sold'])

# Faturamento
df_products['income'] = df_products['price'] * df_products['units_sold']
sns.distplot(df_products['income'])
df_products['income'].median()
df_products['income'].mean()

# Tratamento das 'tags'
from wordcloud import WordCloud, STOPWORDS
word_string = ' '.join(df_products['tags'].str.lower())
wordcloud = WordCloud(stopwords=STOPWORDS).generate(word_string)
plt.subplots(figsize=(15,15))
plt.clf()
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

# Sucesso para faturamento maior que 7000
df_products['sucess'] = 0
df_products.loc[df_products['income'] > 7000, 'sucess'] = 1
df_products['sucess'].value_counts()

#%% Produtos com maior diferença entre 'retail_price' e 'price' vende mais?
# Análise do desconto
df_products['discount'] = df_products['retail_price'] - df_products['price']
fig, ax = plt.subplots(figsize=(15,6))
sns.distplot(df_products.loc[df_products['sucess'] == 1, 'discount'], label='1')
sns.distplot(df_products.loc[df_products['sucess'] == 0, 'discount'], label='0')
plt.legend()

# Ad boost aumentam as vendas?
df_products.loc[df_products['sucess'] == 0, 'uses_ad_boosts'].value_counts() / df_products.loc[df_products['sucess'] == 0, 'uses_ad_boosts'].value_counts().sum()
df_products.loc[df_products['sucess'] == 1, 'uses_ad_boosts'].value_counts() / df_products.loc[df_products['sucess'] == 1, 'uses_ad_boosts'].value_counts().sum()

# Avaliações aumentam as vendas?
fig, ax = plt.subplots(figsize=(15,6))
sns.distplot(df_products.loc[df_products['sucess'] == 1, 'rating'], label='1')
sns.distplot(df_products.loc[df_products['sucess'] == 0, 'rating'], label='0')
plt.legend()

df_products.loc[df_products['sucess'] == 1, 'rating'].mean()
df_products.loc[df_products['sucess'] == 0, 'rating'].mean()

df_products.loc[df_products['sucess'] == 1, 'rating'].median()
df_products.loc[df_products['sucess'] == 0, 'rating'].median()

# Badges importam?
df_products.groupby(['sucess','badges_count']).count()[['title']].pivot_table(index='sucess',columns='badges_count')
df_products.groupby(['sucess','badge_product_quality']).count()[['title']].pivot_table(index='sucess',columns='badge_product_quality')
df_products.groupby(['sucess','badge_fast_shipping']).count()[['title']].pivot_table(index='sucess',columns='badge_fast_shipping')

# Quantidade de tags auxiliam vendas?
df_products['tags_count'] = df_products['tags'].apply(lambda x: len(x.split(',')))
df_products['tags_count']
fig, ax = plt.subplots(figsize=(15,6))
sns.distplot(df_products.loc[df_products['sucess'] == 1, 'tags_count'], label='1')
sns.distplot(df_products.loc[df_products['sucess'] == 0, 'tags_count'], label='0')
plt.legend()

#%% Implementação de Machine Learning
from sklearn.model_selection import train_test_split

# Colunas para o modelo
model_cols = ['price', 'retail_price', 
       'uses_ad_boosts', 'rating', 'badges_count',
       'badge_product_quality', 'badge_fast_shipping', 'product_variation_inventory',
       'shipping_is_express', 'countries_shipped_to', 'inventory_total',
       'has_urgency_banner', 
       'merchant_rating', 'discount', 'tags_count']

x = df_products[model_cols]
y = df_products['sucess']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# Modelo Random Forest
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Escolher os melhores parametros para o modelo
param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestClassifier()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, 
                           return_train_score=True)

grid_search.fit(x_train, y_train)
grid_search.best_params_
rf_model = grid_search.best_estimator_

# Análise do modelo
from sklearn.metrics import classification_report, confusion_matrix

y_pred = rf_model.predict(x_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Importância das features
feature_importances = pd.DataFrame(rf_model.feature_importances_,
                                   index = x.columns,
                                    columns=['importance']).sort_values('importance', ascending=True)

fig, ax = plt.subplots(figsize=(20, 8))
feature_importances.plot(kind="barh", ax=ax)

# Explicação das features
import shap

explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(x)
shap.summary_plot(shap_values[:, :, 1], x)






