def make_pizza(*toppings):
    '''接受任意数量的实参,在实参前面加上星号
    形参名*toppings 中的星号让Python创建一个名为toppings的空元祖,
    并且将接受到的所有值都封装到这个元祖中.
    '''
    #print(toppings)
    print('\nMaking a pizza with the following toppings:')
    for topping in toppings:
        print('- ', topping)

make_pizza('pepperoni')
make_pizza('mushrooms', 'green peppers', 'extra cheese')
