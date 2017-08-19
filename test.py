#TEST#
import re

# from tokenizer_utilities import *
# test = '<h1>List of past NHL scoring leaders</h1><h2>NHL goal scoring leaders</h2><p>#N##N##N##N#Wayne Gretzky#N#, five-time leading scorer#N##N##N##N##N##N##N##N##N##N##N##N##N#</p>'
# test_tokenizer = Table_Tokenizer(10,20)
# test = '<table border="1" cellpadding="5">'
# test =  re.sub(r'cellpadding="[0-9]*"', '',test)
# print re.sub(r'border="[0-9]*"', '',test)
test = [1,2,3,4]
def test_it(arr):
	for item in arr:
		yield item,item+1

def test_it2(iter):
	for item in iter:
		new_list = [item] * 5
		yield new_list

test2 = zip(test_it(test),test_it(test))

for item in test2:
	print item[0]