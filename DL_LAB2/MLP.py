import numpy as np
import plotext as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0])

model = Sequential([
    Dense(4,activation='relu',input_shape=(2,)),
    Dense(1,activation='sigmoid')
])

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

history = model.fit(X,y,epochs=100,verbose=0)
print("completed")
loss,acc=model.evaluate(X,y,verbose=0)
print("loss:",loss)
print("accuracy:",acc)

pred = model.predict(X)
print("Predictions")
print(pred)


plt.plot(history.history['loss'])
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.show()
