#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import os
import numpy as np
import operator
import math
import re
##Tokenization Utilities. This should be kept in sync with the C# tokenizer

class Tokenizer:

	def __init__(self, vocab_len, sentence_len, right_pad = True, right_trun =True):
		self.vocab = {}
		self.vocab_len = vocab_len
		self.max_sentence_len = sentence_len
		self.right_pad = right_pad
		self.right_trun = right_trun

	def load_dict(word_dict):
		self.vocab = word_dict


	def load_dict_file(self, vocab_file_path):

		with open(vocab_file_path, "r") as ins:
			index = 0
			for line in ins:
				self.vocab[line.rstrip()] = index
				index += 1

	def remove_invalid_chars(self, str):
		blacklist_chars = '!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r–’≥®²‘“”°'
		str_list = []
		for char in str:
			if char not in blacklist_chars:
				str_list.append(char)
			else: ##BUG BUG : new change
				str_list.append(' ')

		return ''.join(str_list)

	def tokenize(self, str):
		if self.isNaN(str):
			str =''
		lowered_str = str.lower()
		cleaned_str = self.remove_invalid_chars(lowered_str)
		splitted = cleaned_str.split(' ') ## Still need to remove white spaces
		tokens = []
		start = 0
		token_counter = 0

		if self.right_trun == False:
			start = max(len(splitted) - self.max_sentence_len, 0)

		for index in range(start, len(splitted)):
			current_token = splitted[index]
			if current_token == '': #BUGBUG: new change
				continue
			if current_token in self.vocab:
				if self.vocab[current_token] < self.vocab_len:
					tokens.append(self.vocab[current_token])
					token_counter += 1

				if token_counter >= self.max_sentence_len:
					break

		zeros_list = [0] * (max(self.max_sentence_len - token_counter, 0))

		if self.right_pad:
			tokens = tokens + zeros_list
		else:
			tokens = zeros_list + tokens
		return tokens

	def tokenize_texts(self, chunks, column):
		for chunk in chunks:
			tokenized_text = []
			for text in chunk[column]:
				tokenized = self.tokenize(text)
				tokenized_text.append(tokenized)
			yield tokenized_text

	def gen_dict(self, chunks, column):
		word_dict = {}
		for chunk in chunks:
			for text in chunk[column]:
				lowered_str = text.lower()
				cleaned_str = self.remove_invalid_chars(lowered_str)
				splitted = cleaned_str.split(' ')

				for token in splitted:
					if token not in word_dict:
						word_dict[token] = 1

					else:
						word_dict[token] += 1

		sorted_words = sorted(word_dict.items(), key=operator.itemgetter(1), reverse=True)

		sorted_dict = dict()
		for index in range(len(sorted_words)):
			sorted_dict[sorted_words[index][0]] = index

		self.vocab = sorted_dict

		return sorted_dict

	def isNaN(self,num):
		return num != num


class Table_Tokenizer:

	def __init__(self, vocab_len, sentence_len, right_pad = True, right_trun =True):
		self.vocab = {}
		self.vocab_len = vocab_len
		self.max_sentence_len = sentence_len
		self.right_pad = right_pad
		self.right_trun = right_trun
		self.struct_labels = {'<td>':' TDATSTART ','</td>':' TDATEND ', '<tr>':' TROWSTART ', '</tr>':' TROWEND ','<th>':' THEADERSTART ', '</th>':' THEADEREND ','<h1>':' THEAD1START ', '</h1>':' THEAD1END ',
		 				'<h2>':' THEAD2START ', '</h2>':' THEAD2END ', '<h3>':' THEAD3START ', '</h3>':' THEAD3END ', '<h4>':' THEAD4START ', '</h4>':' THEAD4END ','<p>':' PARAGSTART ', '</p>':' PARAGEND ',
		 				'#n#':'','#r#':'','#tab#':'','<table':'TABLESTART', '</table':'TABLEEND'}

	def load_dict(word_dict):
		self.vocab = word_dict


	def load_dict_file(self, vocab_file_path):

		with open(vocab_file_path, "r") as ins:
			index = 0
			for line in ins:
				self.vocab[line.rstrip()] = index
				index += 1
	def replace_table_struct_labels(self, text):
		for key in self.struct_labels:
			text = text.replace(key,self.struct_labels[key])
		return text

	def remove_invalid_chars(self, str):
		blacklist_chars = '!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r–’≥®²‘“”°'
		str_list = []
		for char in str:
			if char not in blacklist_chars:
				str_list.append(char)
			else: ##BUG BUG : new change
				str_list.append(' ')

		return ''.join(str_list)

	def tokenize(self, str):
		if self.isNaN(str):
			str =''
		lowered_str = str.lower()
		lowered_str =  re.sub(r'cellpadding="[0-9]*"', '',lowered_str)
		lowered_str = re.sub(r'border="[0-9]*"', '',lowered_str)

		cleaned_str = self.replace_table_struct_labels(lowered_str)
		cleaned_str = self.remove_invalid_chars(cleaned_str)

		splitted = cleaned_str.split(' ') ## Still need to remove white spaces
		tokens = []
		start = 0
		token_counter = 0

		if self.right_trun == False:
			start = max(len(splitted) - self.max_sentence_len, 0)

		for index in range(start, len(splitted)):
			current_token = splitted[index]
			if current_token == '': #BUGBUG: new change
				continue
			if current_token in self.vocab:

				if self.vocab[current_token] < self.vocab_len:
					tokens.append(self.vocab[current_token])
					token_counter += 1

				if token_counter >= self.max_sentence_len:
					break

		zeros_list = [0] * (max(self.max_sentence_len - token_counter, 0))

		if self.right_pad:
			tokens = tokens + zeros_list
		else:
			tokens = zeros_list + tokens
		return tokens

	def tokenize_texts(self, chunks, column):
		for chunk in chunks:
			tokenized_text = []
			for text in chunk[column]:
				tokenized = self.tokenize(text)
				tokenized_text.append(tokenized)
			yield tokenized_text

	def gen_dict(self, chunks, column):
		word_dict = {}
		for chunk in chunks:
			for text in chunk[column]:
				lowered_str = text.lower()
				cleaned_str = self.remove_invalid_chars(lowered_str)
				splitted = cleaned_str.split(' ')

				for token in splitted:
					if token not in word_dict:
						word_dict[token] = 1

					else:
						word_dict[token] += 1

		sorted_words = sorted(word_dict.items(), key=operator.itemgetter(1), reverse=True)

		sorted_dict = dict()
		for index in range(len(sorted_words)):
			sorted_dict[sorted_words[index][0]] = index

		self.vocab = sorted_dict

		return sorted_dict


	def isNaN(self,num):
		return num != num