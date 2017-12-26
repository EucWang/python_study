def count_words(filename):
    '''计算一个文件中大约包含多少个单词'''
    try:
        with open(filename) as f_obj:
            contents = f_obj.read()
    except FileNotFoundError:
        #print("The file '" + filename + "' does not exist.")
        # 使用pass 关键字, 让python继续执行下面的内容,当前什么也不做
        # pass 语句还可以充当 占位符
        pass
    else:
        #计算文件内容包含多少个单词
        words = contents.split()
        num_words = len(words)
        print("The file '" + filename + "' has about", str(num_words), "words")

dir = "./test10_file_exception"
filenames = ['alice.txt', 'little_women.txt', 'moby_dick.txt', 'siddhartha.txt']

for filename in filenames:
    count_words(dir + '/' + filename)