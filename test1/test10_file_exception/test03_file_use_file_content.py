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

#print("read file content from out of with statement:")

pi_string = ''
for line in lines:
    #print(line.rstrip())
    # 去掉 前后的所有空格 strip()
    pi_string += line.strip()

print(pi_string)