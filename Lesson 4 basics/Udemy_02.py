# Nested Dictionaries
d = {
    'numb': {
        'k11': [1, 2, 3],
        'k12': [4, 5, 6],
        'k13': [7, 8, 9],
    },
    'name': {
        'k21': ['alex', 'Santi', 'Pau'],
        'k22': ['Ari', 'Amy', 'Laia'],
        'k23': ['Erik', 'Manel', 'Roger']

    }
}
d2 = {'name': {'name1': ['Alex', 'Alejandro', 'Ari'], 'name2': ['Pau', 'Pol', 'Pablo'],
               'name3': ['Roger', 'Robert', 'Raul']},
      'ages': {'age1': [1, 2, 3], 'age2': [10, 20, 30], 'age3': [40, 50, 60]}}

print(d['numb']['k11'][2], d['name']['k21'][2])

print(d2['name']['name2'][1] + " is " + str(d2['ages']['age3'][0]))

print(d2['ages']['age3'])

d3 = {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3}
print(d3)
print('next')
d4 = set([1, 1, 1, 1, 2, 2, 2, 3, 3, 3])
print(d4)

d3.add(5)
print(d3)

print(1 < 2 and 2 < 3)



