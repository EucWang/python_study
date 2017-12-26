
for value in range(1, 5):
    print(value, end='\t')

print('\n')
for value in range(1,6):
    print(value, end='\t')
print('\n')

numbers = list(range(1,6))
print(numbers)

even_numbers = list(range(2,11, 2))
print(even_numbers)

squares = []
for value in range(1,11):
    square = value**2
    squares.append(square)
print(squares)

digits = list(range(1,10))
digits.append(0)
print(digits)
# 获取数字列表的最大值,最小值,总和
print('min of digits :', min(digits))
print('max of digits :', max(digits))
print('sum of digits:', sum(digits))