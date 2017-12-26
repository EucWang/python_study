import sys

def printList(items):
    for item in items:
        print(item,end='')
        if item != items[len(items) - 1]:
            print('\t',end='')
        else:
            print('\n')


cars = ['bmw', 'audi', 'toyota', 'subaru']


print('before sort, cars :')
#printList(cars)
print(cars)
cars.sort()
print('after sort, cars : ')
#printList(cars)
print(cars)

print('sort again, but reverse')
cars.sort(reverse=True)
print(cars)

print('then temp sorted again, but donot changed last time')
print(sorted(cars))
print('show you the cars:')
print(cars)