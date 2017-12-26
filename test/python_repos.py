import pygal
import requests

from pygal.style import LightColorizedStyle as LCS, LightenStyle as LS

url = 'https://api.github.com/search/repositories?q=language:python&sort=starts'
r = requests.get(url)   # 请求url, r就是返回的响应

print('Status code:', r.status_code)  # 获取响应的状态码

response_dict = r.json()  # 获取响应的json数据

# print(response_dict.keys())

repo_dicts = response_dict['items']

#print("Repositories returned:", len(repo_dicts))

#repo_dict = repo_dicts[0]
#print('\nKeys:', len(repo_dict))

#for key in sorted(repo_dict.keys()):
#    print(key)

names = []
stars = []
plot_dicts = []

for repo_dict in repo_dicts:
    names.append(repo_dict['name'])
    stars.append({
        'value': int(repo_dict['stargazers_count']),
        'label': str(repo_dict['description']),  # 根据label 显示 鼠标悬停 的显示
        'xlink': str(repo_dict['html_url'])})    # 根据 xlink ,处理 鼠标点击条目

# x_label_rotation =45 x轴上的标签 旋转45度
# show_legend=False  隐藏图例
pygal_bar = pygal.Bar(style=LS('#336699', base_style=LCS), x_label_rotation=45, show_legend=False)
pygal_bar.title = 'Most-Starred Python Projects on GitHub'
pygal_bar.x_labels = names
pygal_bar.add('', stars)
pygal_bar.render_to_file('python_repos.svg')
