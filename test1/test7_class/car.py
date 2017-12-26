class Car(object):
    """Car class"""

    def __init__(self, **kwargs):
        #return super().__init__(**kwargs)
        self.make = kwargs['make']
        self.model = kwargs['model']
        self.year = kwargs['year']
        self.odometer_reading = 0

    def get_descriptive_name(self):
        long_name = str(self.year) + ' ' + self.make + ' ' + self.model
        return long_name.title()

    def read_odometer(self):
        print("This car has " + str(self.odometer_reading) + " miles on it.")

    def update_odometer(self, mileage):
        if mileage >= self.odometer_reading:
            self.odometer_reading = mileage
        else:
            print("You can't roll back an odometer.")

my_car = Car(make='Audi',model='a4',year=2016)
print(my_car.get_descriptive_name())
#my_car.odometer_reading = 101
my_car.update_odometer(32)
my_car.update_odometer(23)
my_car.read_odometer()

