motorcycles = ['honda', 'yamaha', 'suzuki']
print(motorcycles)
motorcycles.insert(0, 'ducati')
print('after insert some at the index 0,then there are :')
print(motorcycles)

print('pop last item')
last = motorcycles.pop()
print('last item is ' + last)

print('pop the second item')
second = motorcycles.pop(1) 
print('second item is ' + second)