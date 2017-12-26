banned_users = ['andrew', 'carolina', 'david']

#user = 'marie'
user = 'carolina'

if user not in banned_users:
    print(user.title() + ', you can post a response if you wish.')
else:
    print(user.title() + ', you banned.')


requested_toppings = ['mushrooms', 'green peppers', 'extra cheese']

for requested_topping in requested_toppings:
    print('Adding ' + requested_topping, end=',')
print('\nFinished making your pizza!')