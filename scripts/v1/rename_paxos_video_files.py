import sys
import os
import re

BASE_PATH = 'Camp K'
BASE_LETTER = BASE_PATH[-1] 

files = os.listdir(BASE_PATH)

for filename in files:
	if len(filename) <= 4:
		continue
	print(f'Renaming file {filename}')
	num = int(re.findall(r'\d+', filename)[0])
	new_name = BASE_LETTER + f'0{num:02d}'
	os.rename(os.path.join(BASE_PATH, filename), os.path.join(BASE_PATH, new_name))

print(f'Finished renaming files in {BASE_PATH}')


