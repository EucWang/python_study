'''尝试读取文件,如果文件不存在
则让用户输入用户名
如果文件存在, 则问候用户'''

import json

dir = "test10_file_exception"
filename = "username.json"

try:
    with open(dir + "/" + filename, 'r') as f_obj:
        username = json.load(f_obj)
except FileNotFoundError:
    username = input("What is your name?")
    with open(dir + "/" + filename, 'w') as f_obj:
        json.dump(username, f_obj)
        print("I willl remember you when you come back", username + "!")
except BaseException as e:
    pass
else:
    print("Welcome back,", username)
