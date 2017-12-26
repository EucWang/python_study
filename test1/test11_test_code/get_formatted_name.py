def get_formatted_name(first, last, middle=''):
    '''根据用户输入的名和姓,然后返回用户的全名'''
    if middle:
        full_name = first +  ' ' + middle + ' ' + last
    else:
        full_name = first + " " + last
    return full_name.title()
