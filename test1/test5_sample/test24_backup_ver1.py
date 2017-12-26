import os
import time

source=['E:\\apks','E:\\key']
target_dir = 'E:\\OneDriveTemp\\'
target = target_dir + time.strftime('%Y%m%d%H%M%S') + '.zip'
zip_command = "zip -qr %s %s"%(target, ' '.join(source))
print(zip_command,'\n...',end='')
if os.system(zip_command) == 0:
    print('\nsuccessful backup to', target)
else:
    print('\nBackup FAILED')

