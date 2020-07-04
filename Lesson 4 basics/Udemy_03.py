list = list(range(10))
print(list)
list1 = []

for num in list:
    list1.append(num**2)

print(list1)

list2 = [num**2 for num in list1]
print(list2)

def my_func(num):
    #Hello this is a probe
    return num**2


output = my_func(2)

print(output)

