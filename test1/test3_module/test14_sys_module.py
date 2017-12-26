import sys
import test3_module.test15_module_name

print('The command line arguments are:')
for i in sys.argv:
    print(i)

print('\nThe PYTHON PATH is:')
for i in sys.path:
    print(i)

