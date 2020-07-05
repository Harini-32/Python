{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"sequential_6\"\n",
      "_________________________________________________________________\n",
      "Layer (type)                 Output Shape              Param #   \n",
      "=================================================================\n",
      "dense_11 (Dense)             (None, 20)                180       \n",
      "_________________________________________________________________\n",
      "dense_12 (Dense)             (None, 2)                 42        \n",
      "_________________________________________________________________\n",
      "dense_13 (Dense)             (None, 1)                 3         \n",
      "=================================================================\n",
      "Total params: 225\n",
      "Trainable params: 225\n",
      "Non-trainable params: 0\n",
      "_________________________________________________________________\n",
      "None\n",
      "192/192 [==============================] - 0s 854us/step\n",
      "[0.6475343902905782, 0.6145833134651184]\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "import numpy as np\n",
    "from keras.models import Sequential\n",
    "from keras.layers.core import Dense, Activation\n",
    "\n",
    "dataset = pd.read_csv(\"diabetes.csv\", header=None).values\n",
    "\n",
    "X_train, X_test, Y_train, Y_test = train_test_split(dataset[:,0:8], dataset[:,8],\n",
    "                                                    test_size=0.25, random_state=87)\n",
    "np.random.seed(155)\n",
    "my_first_nn = Sequential() # create model\n",
    "my_first_nn.add(Dense(20, input_dim=8, activation='relu')) # hidden layer\n",
    "my_first_nn.add(Dense(2, activation='softmax')) #added layer\n",
    "my_first_nn.add(Dense(1, activation='sigmoid')) # output layer\n",
    "my_first_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])\n",
    "my_first_nn_fitted = my_first_nn.fit(X_train, Y_train, epochs=100,verbose=0,initial_epoch=0)\n",
    "                                     \n",
    "print(my_first_nn.summary())\n",
    "print(my_first_nn.evaluate(X_test, Y_test))"
   ]
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
