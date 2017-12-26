
pets = ['dog','cat','dog','goldfish', 'cat', 'rabbit','cat']

print(pets)

# 判断列表中是否还有某个元素, 如果有,则继续移除该元素
while 'cat' in pets:
    pets.remove('cat')

print(pets)