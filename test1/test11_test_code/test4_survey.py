import unittest

from test11_test_code.survey import AnonymousSurvey

class TestAnonymousSurvey(unittest.TestCase):

    def setUp(self):
        '''setUp()方法会在其他单元测试方法执行之前执行,
        然后可以在setUp()中初始化一些对象,方便其他单元测试使用
        '''
        question = "What language did you first speak?"
        self.responses = ['English','Spanish','Mandarin']
        self.my_survey = AnonymousSurvey(question)


    def test_single_response(self):
        self.my_survey.store_response(self.responses[0])
        self.assertIn(self.responses[0], self.my_survey.responses)

    def test_store_three_response(self):
        for response in self.responses:
            self.my_survey.store_response(response)

        for response in self.responses:
            self.assertIn(response, self.my_survey.responses)

unittest.main()
