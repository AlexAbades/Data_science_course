seq = [1, 2, 3, 4, 5, 6]
out = []


def times2(var):
    return var * 2


print(times2(2))

print(list(map(times2, seq)))

print("Next")

for num in seq:
    num1 = times2(num)
    out.append(num1)

print(out)

t = lambda var: var ** 2

print(t(6))

print(t)

list3 = list(map(lambda num: num * 3, seq))
print(list3)

print(4 % 2 == 0)

list4 = list(filter(lambda num: num % 2 == 0, seq))

print(list4)

list5 = list(map(lambda num: num % 2 == 0, seq))
print(list5)

s = 'Hello my name is Alex'

print(s.split())

tweet = 'Go sports! #Sports'

print(tweet.split())

print(tweet.split("#")[1])

d = {
    'k1': 1,
    'k2': 2,
}


s1 = [1, 2, 3, 'x']
print('x' in s1)

s2 = [(1, 2), (3, 4), (4, 6)]

print(s2[1])

for item in s2:
    for index in item:
        print(index)