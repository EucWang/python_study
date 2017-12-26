from electricCar import ElectricCar

my_telsa = ElectricCar('tesla', 'model s', 2018)
print(my_telsa.get_descriptive_name())
my_telsa.battery.describe_battery()
my_telsa.battery.get_range()
