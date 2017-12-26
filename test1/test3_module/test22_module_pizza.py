
def make_pizza(size, *toppings):
    '''接受制作pizza的原材料'''
    print("\nMaking a " + str(size) + "-inch pizza with the following toppings:")
    for topping in toppings:
        print("- " + topping)