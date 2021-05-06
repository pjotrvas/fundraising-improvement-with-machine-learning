import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import linear_model

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score

import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

st.set_option('deprecation.showPyplotGlobalUse', False)

st.write("""
# Fundraising analiza 
""")


data=pd.read_csv('./dataset.csv')
#print(data.head())
#st.write(data.head())
st.dataframe(data)

data=data.dropna()
#print(data.head())

#fig, ax = plt.subplots() #solved by add this line 
data.boxplot('IZNOSI','KATEGORIJE',rot = 30,figsize=(5,6))
st.pyplot()

#fig, ax = plt.subplots() #solved by add this line 
data.boxplot('IZNOSI','RADIONICE',rot = 30,figsize=(5,6))
st.pyplot()

sns.set_theme(style="darkgrid")
fig, ax = plt.subplots() #solved by add this line 
ax = sns.countplot(x=data['RADIONICE'], data=data)
st.pyplot(fig)
fig, ax = plt.subplots() #solved by add this line 
ax=sns.countplot(x=data['RADIONICE'], hue=data['KATEGORIJE'], data=data)
st.pyplot(fig)

workshop=pd.get_dummies(data['RADIONICE'],drop_first=True)
st.write("Radionice nakon pretvorbe u dummy varijable:")
st.write(workshop.head())
#print(workshop.head())

categories=pd.get_dummies(data['KATEGORIJE'],drop_first=True)
st.write("Kategorije nakon pretvorbe u dummy varijable:")
st.write(categories.head())
#print(categories.head())

data=pd.concat([data,workshop,categories],axis=1)
data=data.drop(['RADIONICE','KATEGORIJE'],axis=1)
st.write(data.head())
print(data.head())



X=data.drop("IZNOSI",axis=1)
y=data["IZNOSI"] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = linear_model.LinearRegression()
model.fit(X_train,y_train)
predictions = model.predict(X_test)
r2 = model.score(X_train, y_train)

print('R^2 = ', r2)
st.write('R^2 = ', r2)

fig, ax = plt.subplots() #solved by add this line 
ax=sns.regplot(x=y_test, y=predictions, ci=68, truncate=False)
st.pyplot(fig)

data = data.iloc[:, 1:-1]

corr = data.corr(method='spearman')

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
fig, ax = plt.subplots(figsize=(6, 5))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True, sep=100)

#fig, ax = plt.subplots() #solved by add this line 
# Draw the heatmap with the mask and correct aspect ratio
ax=sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0, linewidths=.5)


fig.suptitle('Correlation matrix of features', fontsize=15)
ax.text(0.77, 0.2, 'Matea ZI', fontsize=13, ha='center', va='center',
         transform=ax.transAxes, color='grey', alpha=0.5)

fig.tight_layout()
st.pyplot(fig)
