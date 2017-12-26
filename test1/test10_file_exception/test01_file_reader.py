try:
    # 因为当前py文件位于某一个模块下,所以会从模块开始位置为默认搜索位置
    with open("./test10_file_exception/pi_digits.txt") as file_object:
        contents = file_object.read()
        print(contents.rstrip()) # rstrip() 方法会删除掉字符串末尾的空白
except Exception as e:
    print(e)