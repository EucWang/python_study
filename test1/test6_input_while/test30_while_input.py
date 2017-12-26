
prompt = r'''Tell me something, and I will repeat it back to you:
Enter 'quit' to end the program.'''

message = ''
flag = True

#while message != 'quit':
while flag:
    message = input(prompt)
    if message == 'quit':
        flag = False
    else:
        print(message)