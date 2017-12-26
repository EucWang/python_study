'''
编写一个存储一组数字的程序,  json.dump() 方法存储数字
再编写一个将这些数字读取到内存中的程序.  json.load()加载数字
'''

import json

dir = 'test10_file_exception'
filename = 'numbers.json'
numbers = [2,3,5,7,11,13]
try:
    # 以写的方式打开文件
    with open(dir + "/" + filename, 'w') as f_obj:
        # 使用json.dump()将 numbers列表中的内容写入到文件中存储,存储格式为json格式
        json.dump(numbers, f_obj)
except BaseException as e:
    print(e)
else:
    # 完成之后打印一个成功信息
    print("save number to file '" + filename + "' success.")


print("\nthen read file use json.")

read_number = []
try:
    with open(dir + "/" + filename, 'r') as f_obj:
        read_number = json.load(f_obj)
except BaseException as e:
    print(e)
else:
    print("read numbers from file '" + filename + "' success.")
    print("and the numbers are : ", read_number)
