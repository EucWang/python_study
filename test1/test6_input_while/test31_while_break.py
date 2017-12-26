
prompt = r'''Please enter the name of a city you have visited:
(Enter 'quit' when you are finished.)'''

flag = True
while flag:
    city = input(prompt)
    if city=='Tokyo':
        print('I do not like japan city, continue next.')
        continue
    if city=='quit':
        break
    else:
        print('I\'d love to go to ', city.title(), '!')