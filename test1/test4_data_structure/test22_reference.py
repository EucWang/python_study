
print('Simple Assignment')
shoplist=['apple','mango','carrot','banana']

# 通过 = 赋值, 则会将对应的引用赋值给新的变量
mylist=shoplist

del shoplist[0]
print('shoplist is', shoplist)
print('mylist is', mylist)

print('Copy by making a full slice')

#  通过 切片功能赋值,则会生成一个新的序列,然后将这个序列的引用赋值给新的变量
mylist = shoplist[:]    
del mylist[0]   

print('shoplist is', shoplist)  
print('mylist is', mylist)