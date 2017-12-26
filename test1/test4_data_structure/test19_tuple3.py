
dimensions = (200 ,50)
print('Original dimensions :')
for dimension in dimensions:
    print(dimension)

# 元祖的元素值时不能修改的,
# 但是可以给存储元祖的变量赋值
# 就是说可以重新定义这个元祖
dimensions = (400, 100)
print('\nModified dimensions:')
for dimension in dimensions:
    print(dimension)