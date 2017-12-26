name = 'swaroop'

# a.startwith(b) 判断字符串是否以某个字符串开头
if name.startswith('swar'):
    print('Yes, the string starts with "Swa"')
# in操作符 判断某个字符串在另外一个字符串中出现
if 'a' in name:
    print('Yes, it contains the string "a"')
# a.find(b) 在某个字符中查找指定字符串
if name.find('war') != -1:
    print('Yes, it contains the stirng "war"')

delimiter = '_*_'
mylist = ['Brazil', 'Russia', 'India', 'China']

# a.join(listB) :通过join将一个列表所有字符串元素使用指定字符串拼接连接成一个的字符串
print(delimiter.join(mylist))