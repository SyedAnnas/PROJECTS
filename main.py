import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
warnings.filterwarnings('ignore')
import numpy as np
# Mount the Google Drive to Google Colab
df1 = pd.read_csv("prevalence-by-mental-and-substance-use-disorder _AI.csv")
df2 = pd.read_csv("mental-and-substance-use-as-share-of-disease -AI.csv")
df1.head()
df2.head(10)
data = pd.merge(df1, df2)
data.head(10)
# Missing Values from the Dataset(Cleaning Data)
data.isnull().sum()
# Drop the Column
data.drop('Code', axis=1, inplace=True)
# View the data
data.head(10)
# size=row*column,shape=tuple of array dimension(row,column)
data.size, data.shape
# Column Set
data.columns = ['Country', 'Year', 'Schizophrenia', 'Bipolar disorder', 'Eating disorders', 'Anxiety',
                'Drug_Usage', 'depression', 'alcohol', 'Mental_fitness']
data.head(10)
plt.figure(figsize=(12, 6))
numeric_columns = data.select_dtypes(include='number')
correlation_matrix = numeric_columns.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='Blues')
plt.plot()

sns.pairplot(data, corner=True)
plt.show()
mean_value = data['Mental_fitness'].mean()
print(mean_value)
fig = px.pie(data, values='Mental_fitness', names='Year')
fig.show()
fig.write_html("Main_output.html")
fig = px.line(data, x='Year', y='Mental_fitness', color='Country',
              color_discrete_sequence=['red', 'blue'], template='plotly_dark')
fig.show()
df = data
df.head(10)
df.info()
l = LabelEncoder()
for i in df.columns:
    if df[i].dtype == 'object':
        df[i]=l.fit_transform(df[i])
df.shape
X = df.drop(columns='Mental_fitness',axis=1)
Y = df.drop(columns='Mental_fitness')
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=20, random_state=2)
# print("xtrain: ",xtrain.shape)
# print("xtest: ",xtest.shape)
# print("ytrain: ",ytrain.shape)
# print("ytest: ",ytest.shape)
lr = LinearRegression()
lr.fit(xtrain,ytrain)
ytrain_pred = lr.predict(xtrain)
mse = mean_squared_error(ytrain,ytrain_pred)
rmse = (np.sqrt(mean_squared_error(ytrain,ytrain_pred)))
r2 = r2_score(ytrain,ytrain_pred)
print("The Linear Regression model performance for training set")
print(2*"---------------------------------")
print(f"Mse is {mse}")
print(f"RMSE is {rmse}")
print(f"R2 is {r2}")
