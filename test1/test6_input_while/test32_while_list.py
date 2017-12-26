
unconfirmed_users = ['alice','brian','candace']
confirmed_users = []

#将一个列表的元素取出来,然后传递给另外一个列表
while unconfirmed_users:
#while len(unconfirmed_users)>0:
    current_user = unconfirmed_users.pop()
    print('Verfiying user:', current_user)
    confirmed_users.append(current_user)
print('\nThe following users have been confirmed:')
for confirmed_user in confirmed_users:
    print(confirmed_user)