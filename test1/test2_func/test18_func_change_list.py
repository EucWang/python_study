def print_module(unprinted_designs, completed_models):
    '''将第一个列表元素打印出来,移除元素到第二个列表'''
    while unprinted_designs:
        current_design = unprinted_designs.pop()
        print('Printing model:', current_design)
        completed_models.append(current_design)

def show_complted_models(completed_models):
    print('\nThe following models have been printed:')
    for completed_model in completed_models:
        print(completed_model)


unprinted_designs = ['iphone case', 'robot pendant', 'dodecahedron']
completed_models = []

#print_module(unprinted_designs, completed_models)
#给函数传给参数时,使用列表的切片,则不会修改原有的列表数据
print_module(unprinted_designs[:], completed_models)
show_complted_models(completed_models)
print('the unprinted_designs are:\n', unprinted_designs)
