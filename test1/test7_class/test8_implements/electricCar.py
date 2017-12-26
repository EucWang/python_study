import car
import battery

class ElectricCar(car.Car):
    """implement from Car
    电动汽车ElectricCar  继承自 Car"""

    def __init__(self, make, model, year):
        '''super() 方法 用于使用父类的方法'''
        super().__init__(make, model,year)
        '''增加子类中自己的属性'''
        #self.battery_size = 70
        # 将电池的属性信息单独抽取出来作为一个类,将电池类作为一个属性
        self.battery = battery.Battery()

    #def describe_battery(self):
    #    '''控制台输出一条描述电瓶容量的消息'''
    #    print("This car has a", str(self.battery_size)+ "-kWh battery.")

    def fill_gas_tank(self):
        print("This car does not need a gas tank!")

#my_tesla = ElectricCar('tesla', 'model s', 2016)
#print(my_tesla.get_descriptive_name())
#my_tesla.describe_battery()
#my_tesla.fill_gas_tank()

#my_tesla.battery.describe_battery()
#my_tesla.battery.get_range()

