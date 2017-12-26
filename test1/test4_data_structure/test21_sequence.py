shoplist = ['apple', 'mango', 'carrot', 'banana']

print('shoplist is :',end='\t')
for i in shoplist:
    print(i, end=',')

# 通过index获取序列的某一个元素
print('\nItem 0 is', shoplist[0])
print('Item 1 is', shoplist[1])
print('Item 2 is', shoplist[2])
print('Item 3 is', shoplist[3])

# index为负数,标示倒着数,最后一个元素就是-1,依次类推
print('Item -1 is', shoplist[-1])
print('Item -2 is',shoplist[-2])

#切片, 如果指定了结束的index,那么切片之后的序列不包含结束index的元素
print('Item 1 to 3 is', shoplist[1:3])
print('Item 2 to end is', shoplist[2:])
print('Item 1 to -1 is', shoplist[1:-1])
print('Item start to end is', shoplist[:])

#字符串也是序列,一样的使用切片功能
name = 'swaroop'
print('characters 1 to 3 is', name[1:3])
print('characters 2 to end is ', name[2:])
print('characters 1 to -1 is ', name[1:-1])
print('characters start to end is',name[:])