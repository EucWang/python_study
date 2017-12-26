
age = int(input('How old are you?\n'))
if age>=18:
    print('you are adult.')
else:
    print('you are juveniles')


number = input('Enter a number, and I\'ll tell you if it\'s even or odd:\n')
number = int(number)

if number % 2 == 0:
    print('\nThe number', number, "is even.")
else:
    print('\n The number', number, 'is odd.')