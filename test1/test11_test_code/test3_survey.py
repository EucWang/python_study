'''简单使用survey.py中的类'''

#import unittest

from test11_test_code.survey import AnonymousSurvey

question = 'What language did you first learn to speak?'
my_survey = AnonymousSurvey(question)

my_survey.show_question()
print("Enter 'q' at any time to quit.\n")
while True:
    response = input("Language:")
    if response == 'q':
        break
    my_survey.store_response(response)

#显示调查结果
print('\nThank you to everyone whe participated in the survey!')
my_survey.show_result()

