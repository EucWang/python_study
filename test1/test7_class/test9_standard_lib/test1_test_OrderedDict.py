'''
collections.OrderedDict类 的实例的行为几乎和字典相同.
区别只在于记录了键值对的添加顺序
'''
from collections import OrderedDict

from random import randint

favorite_language = OrderedDict()
# 使用{} 来创建一个 空的 字典
# 使用 [] 来创建一个 空的 列表
# 使用 ()  来创建一个 空的 元祖
#favorite_language = {}  
favorite_language['jen'] = 'python'
favorite_language['sarah'] = 'c'
favorite_language['edward'] = 'ruby'
favorite_language['phil'] = 'python'   

for name, language in favorite_language.items():
    print(name.title() + "'s favorite language is", language.title() + ".")

x = randint(1, 6)
print('\nrandint , then x', str(x))