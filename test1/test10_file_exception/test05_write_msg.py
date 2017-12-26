filename = "./test10_file_exception/programming.txt"

# 先将一些内容写入到指定文件中
try:
    # 第二个实参'w' 告诉python ,会以写入的方式打开文件
    # 'w'   写入, 如果文件不存在,则自动创建文件, 如果存在, 会清空文件
    # 'r'   只读 (默认)
    # 'a'   追加 , 如果文件不存在,自动创建, 如果存在, 在文件末尾添加需要写入的内容
    # 'r+'  能够读取和写入文件
    #with open(filename, 'w') as file_object:
    with open(filename, 'a') as file_object:
        # write() 方法不会自动在文本信息上追加 换号符, 需要文本信息自带
        #file_object.write("I love programming.\n")
        #file_object.write("I love creating new games.\n")
        file_object.write("I alse love finding meaning in large datasets.\n")
except Exception as e:
    print(e)

# 然后再从这个文件中间内容读取出来
print("\nthen read file content after write something.\n")
try:
    with open(filename) as file_object:
        print(file_object.read())
except Exception as e:
    print(e)