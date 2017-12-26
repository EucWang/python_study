def printMax(x,y):
    '''prints the maximum of two numbers
    The two values must be integers.'''

    x = int(x)
    y = int(y)

    if(x>y):
        print(x, 'is maximum')
    elif(x == y):
        print(x, 'is equal to', y)
    else:
        print(y, 'is maximum')

printMax(3,6)
printMax(5,5)
print(printMax.__doc__)
