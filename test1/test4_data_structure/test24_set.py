
favorite_languages = {
    'jen':'python',
    'sarah':'c',
    'edward':'ruby',
    'phil':'python',
    'eric':'java',
    'tom':'c'
    }

print('\nThe following languages have been mentioned:')
for language in set(favorite_languages.values()):
    print(language, end='\t')
print('\n')