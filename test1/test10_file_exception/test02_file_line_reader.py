filename = './test10_file_exception/pi_digits.txt'
try:
    with open(filename) as file_object:
        # 逐行读取文件内容
        for line in file_object:
            print(line.rstrip())
except Exception as e:
    print(e)
