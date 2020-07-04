import numpy as np

arr = np.arange(0, 11)
print(arr)
print(arr[2:8])
#print(arr[:6])
#print(arr[:])
arr[0:5] = 100
print(arr)

arr =np.arange(0,11)
print(arr)
slice_of_arr = arr[0:6]
print(slice_of_arr)
slice_of_arr[:] = 99
print(slice_of_arr)
print(arr)
arr_copy = arr.copy()
arr_copy[:] = 100
print(arr_copy)
print(arr)

# MATRIX INDEX
arr_2d = np.array([[5, 10, 15],
                   [20, 25, 30],
                   [35, 40, 45]])

print(arr_2d[2][1])
print(arr_2d[1,0])
print(arr_2d[:2, 1:])


arr1 = np.arange(1, 11)
print(arr1)
bool_arr1 = arr1 > 5
print(bool_arr1)
print(arr1[bool_arr1])
print(arr1[arr1 > 5])
print(arr1[arr1<3])


arr1_2d = np.arange(50).reshape(5,10)
print(arr1_2d)
print(arr1_2d[1:3, 3:5])
print(arr1_2d[arr1_2d >=35].reshape(3,5))
print(arr1_2d[2:5,6:8])
print(arr1_2d[1:5, 7:])
print(arr1_2d[:4, :4])
print(arr1_2d)
print(arr1_2d[:4, 2:7])
print(arr1_2d[1:,1:9])