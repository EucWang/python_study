class Restaurant(object):
    """饭馆的模拟类"""
    def __init__(self, name, type):
        self.restaurant_name = name #饭馆名称
        self.cuisine_type = type #烹调风格

    def describe_restaurant(self):
        print(self.restaurant_name.title() + "'s cuisine type is " + self.cuisine_type.title())

    def open_restaurant(self):
        print("it's time to open the restaurant : " + self.restaurant_name.title())

my_restaurant = Restaurant("wild duck restaurant", "hu nan cuisine")
my_restaurant.describe_restaurant()
my_restaurant.open_restaurant()

