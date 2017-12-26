
responses = {}

flag = True 

while flag:
    name = input('\nWhat is your name?\n')
    response = input('Which mountain would you like to climb someday?\n')
    responses[name] = response

    repeat = input('Would you like to let another person respond?(yes/no)')
    if repeat == 'no':
        flag = False
    else:
        flag = True
print('\n-----Poll Results-----')
for name, response in responses.items():
    print(name, ' would like to climb ', response, ".")