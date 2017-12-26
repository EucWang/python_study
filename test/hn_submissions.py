from operator import itemgetter

import pygal
import requests

url = 'https://hacker-news.firebaseio.com/v0/topstories.json'
r = requests.get(url)
print('Status code:', r.status_code)

submission_ids = r.json()
submission_dicts = []

for submission_id in submission_ids[:30]:
    url = 'https://hacker-news.firebaseio.com/v0/item/' + str(submission_id) + '.json'

    try:
        submission_r = requests.get(url)
        print("submission : ", submission_id, ", status code : " , submission_r.status_code)
        response_dict = submission_r.json()

        submission_dict = {
            'label': response_dict['title'],
            'xlink': 'http://news.ycombinator.com/item?id=' + str(submission_id),
            'value': response_dict.get('descendants', 0)
        }
        submission_dicts.append(submission_dict)
    except BaseException as e:
        print(e)

submission_dicts = sorted(submission_dicts, key=itemgetter('value'), reverse=True)

#for submission_dict in submission_dicts:
##    print('\nTitle', submission_dict['title'])
#    print('Discussion link', submission_dict['link'])
#    print('Comments', submission_dict['comments'])

submission_titles = []
for submission_dict in submission_dicts:
#    print('\nTitle', submission_dict['title'])
    submission_titles.append(submission_dict['label'])

# submission_titles = [title for title in submission_dicts['title']]

bar = pygal.Bar()
bar.add('', submission_dicts)
bar.x_labels  = submission_titles
bar.render_to_file("hn_submission.svg")
