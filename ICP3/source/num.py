import numpy as np
list1 = np.random.randint(1, 20, 20)
print(list1)
array1 = list1.reshape((4, 5))
print(array1)
indexes = np.arange(array1.shape[0]), np.argmax(array1, axis=1)
print(indexes)
array1[indexes] = 0
print(array1)