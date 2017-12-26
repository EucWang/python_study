def build_profile(first, last, **user_info):
    '''创建一个字典,其中包含我们知道的有关用户的一切
    第三个参数,接受任意数量的键值对数据
    '''
    profile ={}
    profile['first_name']=first
    profile['last_name']=last
    for key, value in user_info.items():
        profile[key] = value
    return profile

user_profile=build_profile('albert', 'einstein', location='princeton', field='physics')

print(user_profile)
