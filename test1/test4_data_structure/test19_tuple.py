zoo = ('wolf','elephant','penguin')

print('Number of animals in the zoo is', len(zoo))
print("these are:")
for item in zoo:
    print(item, end=",")

new_zoo = ('monkey','dolphin',zoo)
print('\n\nNumber of animals in the new zoo is', len(new_zoo))
for item in new_zoo:
    print(item, end=",")

print('\n\nAll animals in new zoo are', new_zoo)

print('Animals brought from old zoo are', new_zoo[2])

print('Last animal brought from old zoo is',new_zoo[2][2])
