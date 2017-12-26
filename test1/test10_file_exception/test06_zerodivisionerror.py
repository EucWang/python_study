'''异常使用 try-except 代码块来处理'''
try:
    print(5/0)
except ZeroDivisionError as e:
    print(e)