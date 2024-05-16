import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 1
df = pd.read_csv('bikes_rent.csv')

# 2
X = df['weathersit'].values.reshape(-1, 1)
y = df['cnt'].values

model = LinearRegression()
model.fit(X, y)

plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.xlabel('Weather Situation')
plt.ylabel('Count of Bikes Rented')
plt.title('Linear Regression Model')
plt.show()

# 3
weathersit = 2
pred_value = model.predict([[2]])
print(pred_value)

# 4
X = df[['temp', 'hum']]
y = df['cnt']

pca = PCA(n_components=2)
X_2D = pca.fit_transform(X)
model = LinearRegression()
model.fit(X_2D, y)

plt.scatter(X_2D[:,0], X_2D[:,1], c=y, cmap='coolwarm')
plt.xlabel('temp')
plt.ylabel('hum')
plt.title('Predicted cnt in 2D space')
plt.colorbar(label='cnt')
plt.show()

# 5
X = df.drop(columns=['cnt'])
y = df['cnt']
lasso_model = Lasso(alpha=1.0)
lasso_model.fit(X, y)
index = abs(lasso_model.coef_).argmax()
max_value = df.columns[index]
print(f'The feature with the most influence on cnt: {max_value}')

