# 1
print('Ex 1')
print(7**4)

print(pow(7,4))

print("")

# 2
print('Ex 2')
s = 'Hi there Sam'
print(s.split())
print('')
# 3
print('Ex 3')
planet = "Earth"
diameter = 12742
print('The diameter of {} is {}'.format(planet, diameter))

# 4
print('Ex 4')
lst = [1,2,[3,4],[5,[100,200,['hello']],23,11],1,7]

print(lst[3][1][2][0])
print('')
# 5
print('Ex 5')
d = {
    'k1':[1,2,3,{
        'tricky':['oh','man','inception',{
            'target':[1,2,3,'hello']
        }]
    }]
}

print(d['k1'][3]['tricky'][3]['target'][3])
print('')
# 6
print('Ex 6')
def domain(var):
    s = var
    lst1 = s.split('@')
    return lst1[1]

print(domain("alex@gmail.com"))

print('')

# 7
print('Ex 7')

def dog(text):
    if 'dog' in text:
        return True
    else:
        return False
#
def findDog(s1):
    return 'dog' in s1.lower().split()
print(findDog('is dog here'))
#
t = lambda d : 'dog' in d.lower().split()

print(t('is dog here'))
# 8
print('Ex 8')
def count_dog(phrase):
    count = 0
    lst = phrase.split()
    for word in lst:
        if word == 'dog':
            count += 1
        else:
            count = count
    return count
print(count_dog('This dog runs faster than the other dog dude!'))
print('')
print('else')
def countdog(str1):
    coun = 0
    for word in str1.lower().split():
        if word == 'dog':
            coun +=1
    return coun
print(countdog('This dog runs dog faster than the other dog dude!'))




# 9
print('Ex 9')

seq = ['soup','dog','salad','cat','great']
print(seq)
t = list(filter(lambda word : word[0]=='s', seq))
print(t)
print('')
# 10
print('Ex 10')
"""
You are driving a little too fast, and a police officer stops you. Write a function
to return one of 3 possible results: "No ticket", "Small ticket", or "Big Ticket". 
If your speed is 60 or less, the result is "No Ticket". If speed is between 61 
and 80 inclusive, the result is "Small Ticket". If speed is 81 or more, the result is "Big Ticket". Unless it 
is your birthday (encoded as a boolean value in the parameters of the function) -- on your birthday, 
your speed can be 5 higher in all cases.
"""
def caught_speed(speed, is_birthday):
    while is_birthday:
        if speed <= 65 :
            return "No Ticket"
        elif 65 < speed <= 85:
            return "Small Ticket"
        else:
            return "Big Ticket"
    if speed <= 60:
        return 'No ticket'
    elif 60 < speed <= 80:
        return 'Small Ticket'
    else:
        return 'Big Ticket'
print(caught_speed(81,False))

def speedy(speed, is_birthday):
    if is_birthday:
        speeding = speed - 5
    else:
        speeding = speed
    if speeding >80:
        return 'Big Ticket'
    elif speeding > 60 :
        return 'Small Ticket'
    else:
        return 'No Ticket'
print(speedy(65, True))

# Instead of equal 5 to the speed limit, it subtract 5 to the velocity
