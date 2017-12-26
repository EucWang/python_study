
# 'ab' 地址簿

#生成一个字典数据
ab = {
    'Swaroop':'swaroopch@byteofpython.info',
    'Larry':'larry@wall.org',
    'Matsumote':'matz@ruby-lang.org',
    'Spammer':'spammer@hotmail.com'
    }

# 通过key获取字典里对应的value值
print("Swaroop's address is %s"%ab['Swaroop'])

# 添加一条道字典里去
ab['Guido'] = 'guido@python.org'

#从字典里删除某一条
del ab['Spammer']
print('delete spammer and add guido, then \n')

#循环遍历字典
for name, email in ab.items():
    print("Contact %s at %s"%(name, email))

# 判断一个key是否在字典中
print('is spammer in ab ', dict.__contains__(ab, 'Spammer'))
print('is guido in ab ', dict.__contains__(ab, 'Guido'))
