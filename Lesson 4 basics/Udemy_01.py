print(3**3)
num= 24
name = "Alex"

print("My number is {} and my name is {}".format(num,name))

print('hello world')

print("My name is {} and my name is {}, moreover I'm also {} years old".format(num, name, num))
print("Hello, my name is {one}, and i'm {two} years old, and I live in the {two} of C/Buenos aires street".format(one=name, two= num))
print(name[3])
my_list= ['a', 'b', 'c', 'd']
print(my_list)

my_list.append("b", "c")

print(my_list)

print(my_list[3])

for letter in my_list:
    print(letter)

print("next")

my_list1= ['a', 'b', 'c', [1, 2], ['target', 'Hello', 'mimau'], [1, 2, 3, 4, 'a']]

print(my_list1[3])

print("next")

for item in my_list1:
    for index in item:
        print(index)

print(my_list1[5][4])

