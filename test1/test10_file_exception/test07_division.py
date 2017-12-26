'''创建一个只执行除法运算的简单计算器
try-except-else  工作原理:
尝试执行try 代码块
出现异常则中断try代码块的执行,进入到except代码块执行
try代码块成功执行之后会去执行else代码块中的代码
'''

print("Give me two numbers, and I'll divide them.")

print("Enter 'q' to quit.")

while True:
    first_number = input("\nFirst Number:")
    if first_number == 'q':
        break
    second_number = input("Second Number:")
    if second_number == 'q':
        break
    try:
        answer = int(first_number) / int(second_number)
    except ZeroDivisionError:
        print("You can't divide by 0!")
    except TypeError:
        print("You can't use non-number to divide.")
    except BaseException as e:
        print("Other exception.", e)
    else:
        print("the answer is :", str(answer))
