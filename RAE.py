# -*- coding: utf-8 -*-

'''
输入   一个句子的向量矩阵，每个词都是一个向量  sentence_input[[1,2,3,..,n],[],...]
       编码器参数  w1(n行*2n列),b1偏差 （n行1列）
       解码器参数  w2(n行*2n列),b2偏差 （n行1列）
       每个词的向量长度 n
输出一个向量，表示该句子  sentence_output[1,2,3...,n]
'''

'''
样例解释：
“深度”的向量表示为[0.3 0.1 0.6]T，“学习”的向量表示为[0.2 0.5 0.7]T
假定“深度 学习”是“深度”、“学习”的父节点p ，“深度”是第一个子节点C1 ，“学习”是第二个子节点C2
得到父节点[0.8 0.2 0.7]T
对整个句子递归使用这种神经网络结构，使用贪心算法每次选取两个节点结合之后父节点得分最高的节点组合，最终可
以得到整个句子的n 维向量表示
'''
from numpy import *
import math

#全局变量
sentence_input = []
w1 = []
b1 = []
w2 = []
b2 = []
n = 0


#两个点的父节点P
#每个词处理成列向量c
def get_p(c1, c2, w1, b1):
	#联接两个向量
	c_joint = concatenate((c1, c2))
	x = dot(w1, c_joint) + b1
	#双曲正切函数
	x = x.T.tolist()[0]
	for i in range(len(x)):
		x[i] = math.tanh(x[i])
		# print item
	p = mat(x).T
	return p

#通过P还原出的叶子节点 c1',c2'  ,分别记为c11,c22
def get_c11_c22_from_p(p, w2, b2, n):
	c11_joint = dot(w2, p) + b2
	c11 = c11_joint[0:n]
	c22 = c11_joint[n:2*n]
	return c11,c22


#两个点的E,不加权重的情况下，c1,c2的重构误差
def get_E(c1, c2, c11, c22):
	c1_minus = (c1 - c11).T.tolist()[0]
	c2_minus = (c2 - c22).T.tolist()[0]
	sumE = 0
	for item in c1_minus:
		sumE += item**2
	for item in c2_minus:
		sumE += item**2
	return sumE

#计算所有的E,P,更新sentence_input
def get_new_sentence_input(sentence_input):
	E = []
	Ps = []
	for i in range(len(sentence_input)- 1):
		c1 = sentence_input[i]
		c2 = sentence_input[i+1]
		p = get_p(c1, c2, w1, b1)
		Ps.append(p)
		c11,c22 = get_c11_c22_from_p(p, w2, b2, n)
		E_samp = get_E(c1, c2, c11, c22)
		E.append(E_samp)
	#找出最小的E所对应的sentence_input的下标
	# E_temp = sorted(E)
	#item < 1
	min_E = 100
	for item in E:
		if item < min_E:
			min_E = item
	#min_index存储E最小的那对的第一个index
	min_index = -1
	for i in range(len(E)):
		if E[i] == min_E:
			min_index = i
			break
	#更新sentence_input,用p代替这两个点
	#p代替左孩子，去掉右孩子
	sentence_input[min_index] = Ps[min_index]
	sentence_input.remove(sentence_input[min_index + 1])
	return sentence_input


#贪心计算树形最优结构，树中的根节点即为句子的向量表示形式
def RAE_main(sentence_input):
	# 每个词处理成向量形式
	for i in range(len(sentence_input)):
		sentence_input[i] = mat(sentence_input[i]).T
	while(len(sentence_input) > 1):
		sentence_input = get_new_sentence_input(sentence_input)
	#len(sentence_input) = 1 时，得到sentence_input的根节点，即sentence_output
	sentence_output = sentence_input
	return sentence_output


if __name__ == '__main__':
	main()