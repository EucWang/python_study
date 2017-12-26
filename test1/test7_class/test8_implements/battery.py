class Battery(object):
    """电动车电瓶"""

    def __init__(self, battery_size=70):
        self.battery_size = battery_size

    def describe_battery(self):
        print("This car has a", str(self.battery_size) + "-kWh battery.")

    def get_range(self):
        '''控制台输出一条信息, 指出电瓶的续航里程'''
        if self.battery_size == 70:
            range = 240
        elif self.battery_size == 85:
            range = 270

        print("This car can go approximately", str(range), "miles on a full charge.")



