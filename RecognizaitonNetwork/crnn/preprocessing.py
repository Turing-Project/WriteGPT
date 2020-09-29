'''
360万中文训练集标签修改
'''

# chinese characters dictionary for 3.6 million data set.
with open('../char_std_5990.txt', 'rb') as file:
	char_dict = {num : char.strip().decode('gbk','ignore') for num, char in enumerate(file.readlines())}

# processing output
with open('../test.txt') as file:
	value_list = ['%s %s'%(segment_list.split(' ')[0], ''.join([char_dict[int(val)] for val in segment_list[:-1].split(' ')[1:]])) for segment_list in file.readlines()]

# final output
with open('test.txt', 'w', encoding='utf-8') as file:
	[file.write(val+'\n') for val in value_list]

'''
orginal version
'''

# with open('../char_std_5990.txt', 'rb') as file:
# 	char_dict = {num : char.strip().decode('gbk','ignore') for num, char in enumerate(file.readlines())}

#  value_list = []
# with open('../test.txt') as file:
# 	label_list = file.readlines()
# 	for segment_list in label_list:
# 		key = segment_list.split(' ')[0]
# 		segment_list = segment_list[:-1].split(' ')[1:]
# 		temp = [char_dict[int(val)] for val in segment_list]
# 		value_list.append('%s %s'%(key, ''.join(temp)))
	
# with open('test.txt', 'w', encoding='utf-8') as file:
# 	[ file.write(val+'\n') for val in value_list]