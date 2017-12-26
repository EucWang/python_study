import unittest  #引入单元测试模块

# 引入被测试的python模块
from test11_test_code.get_formatted_name import get_formatted_name

#创建一个类继承 unittest.TestCase类
# 必须继承这个unittest.TestCase这个类
class NamesTestCase(unittest.TestCase):
    '''测试get_formatted_name.py'''

    def test_first_last_name(self):
        '''单元测试方法, 方法必须以test开头'''
        formatted_name = get_formatted_name('janis', 'joplin')
        # 断言方法
        #判断函数返回的结果是否是期望的值
        self.assertEqual(formatted_name, 'Janis Joplin')

    def test_first_middle_last_name(self):
        '''测试包含中间名称的情况'''
        formatted_name = get_formatted_name('wolfgang', 'mozart','amadeus')
        self.assertEqual(formatted_name, 'Wolfgang Amadeus Mozart')

# 让Python开始进行一项测试
unittest.main()

