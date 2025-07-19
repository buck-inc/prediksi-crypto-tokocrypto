from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df['target'] = df['close'].shift(-1)  # prediksi harga close berikutnya
df.dropna(inplace=True)

X = df[['open', 'high', 'low', 'volume']]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
