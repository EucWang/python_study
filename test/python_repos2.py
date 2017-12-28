import pygal
# import requests

from pygal.style import LightColorizedStyle as LCS, LightenStyle as LS

# url = 'https://api.github.com/search/repositories?q=language:python&sort=starts'
# r = requests.get(url)   # 请求url, r就是返回的响应

# print('Status code:', r.status_code)  # 获取响应的状态码

# response_dict = r.json()  # 获取响应的json数据

# print(response_dict.keys())

# repo_dicts = response_dict['items']

#print("Repositories returned:", len(repo_dicts))

#repo_dict = repo_dicts[0]
#print('\nKeys:', len(repo_dict))

#for key in sorted(repo_dict.keys()):
#    print(key)

repo1 = {'name': 'mina_s1', 'count': '8051', 'faildTotal': '279', 'desc': '3.47%'}
repo2 = {'name': 'mina_s2', 'count': '4028', 'faildTotal': '124', 'desc': '3.08%'}
repo3 = {'name': 'mina_s3', 'count': '1192', 'faildTotal': '48', 'desc': '4.03%'}
repo4 = {'name': 'mina_s4', 'count': '1319', 'faildTotal': '297', 'desc': '22.52%'}

repo5 = {'name': 'hwgw_s1', 'count': '4786', 'faildTotal': '455', 'desc': '9.51%'}
repo6 = {'name': 'hwgw_s2', 'count': '2192', 'faildTotal': '138', 'desc': '6.30%'}
repo7 = {'name': 'hwgw_s3.1.0.16', 'count': '628', 'faildTotal': '77', 'desc': '12.26%'}
repo8 = {'name': 'hwgw_s4', 'count': '104', 'faildTotal': '5', 'desc': '4.81%'}
repo_dicts = [repo1, repo2, repo3, repo4,
              repo5, repo6, repo7, repo8]

names = []
stars = []
stars2 = []
plot_dicts = []

for repo_dict in repo_dicts:
    names.append(repo_dict['name'])
    stars.append({
        'value': int(repo_dict['count']),
        'label': str("请求总数" + repo_dict['count']),  # 根据label 显示 鼠标悬停 的显示
        'color': 'green'
        # 'xlink': str(repo_dict['html_url']) #})    # 根据 xlink ,处理 鼠标点击条目
    })
    stars2.append({
        'value': int(repo_dict['faildTotal']),
        'label': str("失败率" + repo_dict['desc']),  # 根据label 显示 鼠标悬停 的显示
        'color': 'red',
        # 'xlink': str(repo_dict['html_url']) #})    # 根据 xlink ,处理 鼠标点击条目
    })

# x_label_rotation =45 x轴上的标签 旋转45度
# show_legend=False  隐藏图例

    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    # plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# pygal_bar = pygal.Bar(style=LS('#336699', base_style=LCS), x_label_rotation=45, show_legend=False)
# pygal_bar = pygal.Bar(x_label_rotation=45, print_values=True, value_formatter=lambda x: '{}'.format(x))
pygal_bar = pygal.Bar(x_label_rotation=45, print_values=True)
pygal_bar.title = '按照mima服务器分类统计'
pygal_bar.x_labels = names
pygal_bar.add('总的请求数', stars)
# faid_count = pygal_bar.add('失败的请求数', stars2, formatter=lambda x: '%s' % x)
faid_count = pygal_bar.add('失败的请求数', stars2)
# faid_count.config.css
pygal_bar.render_to_file('python_repos2.svg')
