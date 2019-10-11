import sys
import os
import re

BASE_PATH = 'Camp P'
BASE_LETTER = BASE_PATH[-1] 

files = os.listdir(BASE_PATH)

for filename in files:
	print(f'Going into folder {filename}')
	mov_files = os.listdir(os.path.join(BASE_PATH, filename))
	working_files = []
	for mov_file in mov_files:
		if not mov_file.endswith('.MOV'):
			print(f'Skpping file {mov_file} in folder {filename}')
			continue
		file_num = int(re.findall(r'\d+', filename)[0])
		working_files.append((mov_file, file_num))
	if len(working_files) != 2:
		print(f'Skipping folder {filename}, too many files ({len(mov_files)})')
		continue
	smaller_idx = working_files[0][0] if working_files[0][1] <= working_files[1][1] else working_files[1][0]
	bigger_idx = working_files[0][0] if working_files [0][1] > working_files[1][1] else working_files[1][0]

	print(f'Renaming file {smaller_idx} to {filename + "R.MOV"}')
	print(f'Renaming file {bigger_idx} to {filename + "L.MOV"}')
	os.rename(os.path.join(BASE_PATH, filename, smaller_idx), os.path.join(BASE_PATH, filename, filename + 'R.MOV'))
	os.rename(os.path.join(BASE_PATH, filename, bigger_idx), os.path.join(BASE_PATH, filename, filename + 'L.MOV'))


print(f'Finished renaming files in {BASE_PATH}')


