
players = ['charles', 'martina', 'michael', 'florence', 'eli']

print('Here are the first three players on my team:')
for player in players[:3]:
    print(player.title(), end='\t')
print('\n')

my_foods = ['pizza', 'falafel', 'carrot cake']

friend_foods = my_foods[:]  
friend_foods.append('pizza')
print(my_foods)
print(friend_foods)