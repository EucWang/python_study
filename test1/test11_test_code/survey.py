class AnonymousSurvey(object):
    """手机匿名调查问卷的答案"""

    def __init__(self, question):
        self.question = question
        self.responses = []

    def show_question(self):
        '''显示调查的问题'''
        print(self.question)

    def store_response(self, new_response):
        self.responses.append(new_response)

    def show_result(self):
        print('Survey results:')
        for response in self.responses:
            print('-', response)


