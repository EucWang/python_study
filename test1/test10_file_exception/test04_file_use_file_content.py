#filename = './test10_file_exception/pi_digits.txt'
filename = './test10_file_exception/pi_million_digits.txt'
lines = []
try:
    with open(filename) as file_object:
        # 逐行读取文件内容
        #for line in file_object:
         #   print(line.rstrip())
         # 获取每一行内容的列表传递给一个外部列表
         lines = file_object.readlines()
except Exception as e:
    print(e)

pi_string = ''

for line in lines:
    pi_string += line.strip()

birthday = input("Enter your birthday, in the form mmddyy: ")
if birthday in pi_string:
    print("Your birthday appears in the first million digits of pi!")
else:
    print("Your birthday does not appear in the first million digits of pi.")