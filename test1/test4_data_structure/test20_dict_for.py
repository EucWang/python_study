
favorite_languages = {
    'jen':'python',
    'sarah':'c',
    'edward':'ruby',
    'phil':'python',
    }

for name, language in favorite_languages.items():
    print(name, "\'s favorite language is ", language)

print('\nall the people are :')
#for name in favorite_languages.keys():
for name in favorite_languages:
    print(name, end=',')
print('\n')

if 'erin' not in favorite_languages.keys():
    print('Erin, please take our poll!')

print('\n#sorted the dict#')
for name in sorted(favorite_languages.keys()):
    print(name.title() + ", thank you for taking the poll.")

print('\nThe following languages have been mentioned:')
for language in favorite_languages.values():
    print(language, end='\t')
print('\n')


