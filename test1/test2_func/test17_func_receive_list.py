def greet_users(names):
    '''向列表中的每一个用户发出简单的问候'''
    for name in names:
        msg = 'Hello, ' + name.title() + '!'
        print(msg)

usernames = ['hanah', 'ty', 'margot']
greet_users(usernames)