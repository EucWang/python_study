'''FileNotFoundError  在python找不到要打开的文件时创建的异常.'''
filename = 'alice.txt'

try:
    with open(filename) as f_obj:
        contents = f_obj.read()
except FileNotFoundError:
    msg = "Sorry, the file '" + filename + "' does not exist."
    print(msg)

