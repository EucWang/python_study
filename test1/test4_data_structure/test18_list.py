shoplist =['apple', 'mango','carrot','banana']

print('I have',len(shoplist),'items to purchase')

print("Thses items are:")
for item in shoplist:
    print(item,end=',')

print('\n\nI alse have to buy rice')
shoplist.append('rice')
print('My shopping list is now', shoplist)

print('\nI will sort my list now')
shoplist.sort()
print('Sorted shopping list is', shoplist)

print('\nThe first item I will buy is', shoplist[0])
olditem = shoplist[0]
del shoplist[0]
print('\nI bought the', olditem)
print('\nMy shopping list is now', shoplist)
