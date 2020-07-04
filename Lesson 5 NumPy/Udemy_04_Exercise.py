# Ex 1
import numpy as np

# Ex 2: Create an array of 10 zeros
print('Ex 2')
arr = np.zeros(10)
print(arr)
print('')

# Ex 3: Create an array of 10 ones
print('Ex 3')
arr1 = np.ones(10)
print(arr1)

# Ex 4 Create an array of 10 fives
print('Ex 4')
arr2 = np.ones(10) * 5
print(arr2)
print('')

# Ex 5: Create an array of the integers from 10 to 50
print('Ex 5')
arr3 = np.arange(10, 51)
print(arr3)
print('')

# Ex 6: Create an array of all the even integers from 10 to 50
print('Ex 6')
arr4 = np.arange(10, 51, 2)
print(arr4)
print('')

# Ex 7: Create a 3x3 matrix with values ranging from 0 to 8
print('Ex 7')
arr5 = np.arange(0, 9).reshape(3, 3)
print(arr5)
print('')

# Ex 8: Create a 3x3 identity matrix
print('Ex 8')
arr6 = np.eye(3)
print(arr6)
print('')

# Ex 9: Use NumPy to generate a random number between 0 and 1
print('Ex 9')
arr7 = np.random.rand()  # If we don't specify, by default 1 number.
print(arr7)
print('')

# Ex 10: Use NumPy to generate an array of 25 random numbers sampled from a standard normal distribution
print('Ex 10')
arr8 = np.random.randn(25)
print(arr8)
print('')

# Ex 11: Create the following matrix

# array([[ 0.01,  0.02,  0.03,  0.04,  0.05,  0.06,  0.07,  0.08,  0.09,  0.1 ],
#       [ 0.11,  0.12,  0.13,  0.14,  0.15,  0.16,  0.17,  0.18,  0.19,  0.2 ],
#       [ 0.21,  0.22,  0.23,  0.24,  0.25,  0.26,  0.27,  0.28,  0.29,  0.3 ],
#       [ 0.31,  0.32,  0.33,  0.34,  0.35,  0.36,  0.37,  0.38,  0.39,  0.4 ],
#       [ 0.41,  0.42,  0.43,  0.44,  0.45,  0.46,  0.47,  0.48,  0.49,  0.5 ],
#       [ 0.51,  0.52,  0.53,  0.54,  0.55,  0.56,  0.57,  0.58,  0.59,  0.6 ],
#       [ 0.61,  0.62,  0.63,  0.64,  0.65,  0.66,  0.67,  0.68,  0.69,  0.7 ],
#       [ 0.71,  0.72,  0.73,  0.74,  0.75,  0.76,  0.77,  0.78,  0.79,  0.8 ],
#       [ 0.81,  0.82,  0.83,  0.84,  0.85,  0.86,  0.87,  0.88,  0.89,  0.9 ],
#       [ 0.91,  0.92,  0.93,  0.94,  0.95,  0.96,  0.97,  0.98,  0.99,  1.  ]])

print('Ex 11')
arr9 = np.arange(0.01,1.01,0.01).reshape(10,10)
print(arr9)
print('')
# Ex 12: Create an array of 20 linearly spaced points between 0 and 1
print('Ex 12')
arr10 = np.linspace(0,1,20)
print(arr10)
print('')

# Ex 13 Numpy Indexing and Selection
arr11 = np.arange(1,26).reshape(5,5)
print(arr11)
print('')
print('Ex 13')
print(arr11[2:,1:])
print('')
print('Ex 14')
print(arr11[3,4])
print('')
print('Ex15')
print(arr11[:3,1].reshape(3,1))
print('')
print(arr11[:3,1:2])
print('')
print('Ex 16')
print(arr11[4,:])
print('')
print('Ex 16.2')
print(arr11[3:,:])
print('')

# Ex 17: Get the sum of all the values in mat
print('Ex 17')
arr11_sum1 = 0
for row in arr11:
    for index in row:
        arr11_sum1 += index
print(arr11_sum1)
print(arr11.sum())
print('')
# Ex 18: Get the standard deviation of the values in mat
print('Ex 18')
print(np.std(arr11))
print(arr11.std())
print('')
# Ex 19: Get the sum of all the columns in mat
print('Ex 19')
arr11_sum_column = 0
for index in arr11:
    arr11_sum_column += index
print(arr11_sum_column)
print(arr11.sum(axis = 0))



