{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 81,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "B    357\n",
      "M    212\n",
      "Name: diagnosis, dtype: int64\n",
      "Model: \"sequential_71\"\n",
      "_________________________________________________________________\n",
      "Layer (type)                 Output Shape              Param #   \n",
      "=================================================================\n",
      "dense_145 (Dense)            (None, 20)                620       \n",
      "_________________________________________________________________\n",
      "dense_146 (Dense)            (None, 1)                 21        \n",
      "=================================================================\n",
      "Total params: 641\n",
      "Trainable params: 641\n",
      "Non-trainable params: 0\n",
      "_________________________________________________________________\n",
      "None\n",
      "143/143 [==============================] - 0s 2ms/step\n",
      "[0.1752929748068531, 0.9440559148788452]\n",
      "\n",
      "----------Q.3 After Normalization---------------\n",
      "\n",
      "Model: \"sequential_71\"\n",
      "_________________________________________________________________\n",
      "Layer (type)                 Output Shape              Param #   \n",
      "=================================================================\n",
      "dense_145 (Dense)            (None, 20)                620       \n",
      "_________________________________________________________________\n",
      "dense_146 (Dense)            (None, 1)                 21        \n",
      "=================================================================\n",
      "Total params: 641\n",
      "Trainable params: 641\n",
      "Non-trainable params: 0\n",
      "_________________________________________________________________\n",
      "None\n",
      "143/143 [==============================] - 0s 77us/step\n",
      "[0.4570644114400957, 0.811188817024231]\n"
     ]
    }
   ],
   "source": [
    "from keras.models import Sequential\n",
    "from keras.layers.core import Dense\n",
    "\n",
    "# load dataset\n",
    "import pandas as pd\n",
    "dataset = pd.read_csv(\"breastcancer.csv\")\n",
    "\n",
    "\n",
    "X = dataset.iloc[:, 2:32].values\n",
    "y = dataset.iloc[:, 1].values\n",
    "\n",
    "print(dataset.iloc[:, 1].value_counts())\n",
    "\n",
    "# Encoding categorical data\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "labelencoder_X_1 = LabelEncoder()\n",
    "y = labelencoder_X_1.fit_transform(y) # Fit label encoder and return encoded labels M=1, B=0\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)\n",
    "\n",
    "\n",
    "my_first_nn = Sequential() # create model\n",
    "my_first_nn.add(Dense(20, input_dim=30, activation='relu')) # hidden layer\n",
    "my_first_nn.add(Dense(1, activation='sigmoid')) # output layer\n",
    "my_first_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])\n",
    "my_first_nn_fitted = my_first_nn.fit(X_train, y_train, epochs=100, verbose=0, initial_epoch=0)\n",
    "\n",
    "print(my_first_nn.summary())\n",
    "print(my_first_nn.evaluate(X_test, y_test))\n",
    "\n",
    "# --------------Q.3 Data is normalize here ----------------------#\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "sc = StandardScaler()\n",
    "\n",
    "X_train = sc.fit_transform(X_train)\n",
    "X_test = sc.transform(X_test)\n",
    "\n",
    "my_first_nn1 = Sequential() # create model\n",
    "my_first_nn1.add(Dense(20, input_dim=30, activation='relu')) # hidden layer\n",
    "my_first_nn1.add(Dense(1, activation='sigmoid')) # output layer\n",
    "my_first_nn1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])\n",
    "\n",
    "\n",
    "my_first_nn_fitted1 = my_first_nn1.fit(X_train, y_train, epochs=100, verbose=0, initial_epoch=0)\n",
    "\n",
    "print(\"\\n----------Q.3 After Normalization---------------\\n\")\n",
    "print(my_first_nn.summary())\n",
    "print(my_first_nn.evaluate(X_test, y_test))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
