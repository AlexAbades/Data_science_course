# make a data frame
arr = np.arange(0, 41)
print(arr)
arr2= []
for element in arr:
    if element%2 == 0:
        arr2.append('Male')
    else:
        arr2.append('Famale')
arr3 = np.random.randn(41)
print(arr3)

data = pd.DataFrame([arr3, arr2], ['sex', 'tips'], arr).transpose()
# data = pd.Series([arr3, arr2],np.arange(0, 40), ['sex', 'tips'])
print(data)