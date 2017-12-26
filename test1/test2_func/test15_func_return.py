
def get_formatted_name(first_name, last_name, middle_name=''):
    # 非空字符会自动被解读为True
    if middle_name:
        full_name = first_name.title() + ' '+ middle_name.title() + ' ' + last_name.title()
    else:
        full_name = first_name.title() + ' ' + last_name.title()

    return full_name

musician = get_formatted_name('jimi', 'hendrix')
print(musician)
musician = get_formatted_name('john', 'hooker', 'lee')
print(musician)