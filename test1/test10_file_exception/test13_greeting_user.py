'''对test12_remember_user.py进行重构
将其拆分为2个函数'''

import json

# 文件夹
dir = 'test10_file_exception'
# 文件名称
filename = 'username.json'


# 从文件中获取用户名称
def get_stored_user():
    username = ''
    try:
        with open(dir + "/" + filename) as f_obj:
            username = json.load(f_obj)
    except FileNotFoundError:
        return None
    else:
        return username

# 如果能从文件中获取名称,则 问候用户
# 如果从文件中没有获取到用户名称,则提示用户输入用户名,然后保存用户名称,然后提示用户
def greet_user():
    username = get_stored_user()
    if username:
        print("Welcome back,", username + "!")
    else:
        username = input("What is your username?")
        try:
            with open(dir + "/" + filename, 'w') as f_obj:
                json.dump(username, f_obj)
        except BaseException as e:
            print("Non except error :", e)
        else:
            print("We'll remember you when you come back,", username + "!")

# 调用方法
greet_user()  