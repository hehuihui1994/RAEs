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
#每个词的向量长度 n
n = 0




#两个点的父节点P
#每个词处理成列向量c
#隐藏层所有的父节点需要归一化
def get_p(c1, c2, w1, b1, is_root):
	#联接两个向量
	c_joint = concatenate((c1, c2))
	x = dot(w1, c_joint) + b1
	#双曲正切函数
	x = x.T.tolist()[0]
	#判断是否为根节点，不是根节点的需要归一化该父节点
	if is_root != true:
		#归一化
		#向量的模
		x_m = 0
		for i in range(len(x)):
			x[i] = math.tanh(x[i])
			x_m += x[i]*x[i]
		x_m = sqrt(x_m)
		for i in range(len(x)):
			x[i] = x[i] / x_m
	else:
		for i in range(len(x)):
			x[i] = math.tanh(x[i])

	p = mat(x).T
	return p

#通过P还原出的叶子节点 c1',c2'  ,分别记为c11,c22
def get_c11_c22_from_p(p, w2, b2, n):
	c11_joint = dot(w2, p) + b2
	c11 = c11_joint[0:n]
	c22 = c11_joint[n:2*n]
	return c11,c22


#两个点的E,不加权重的情况下，c1,c2的重构误差
# def get_E(c1, c2, c11, c22):
# 	c1_minus = (c1 - c11).T.tolist()[0]
# 	c2_minus = (c2 - c22).T.tolist()[0]
# 	sumE = 0
# 	for item in c1_minus:
# 		sumE += item**2
# 	for item in c2_minus:
# 		sumE += item**2
# 	return sumE

##两个点的E,加权重的情况下，c1,c2的重构误差
def get_E_weight(c1, c2, c11, c22, n1, n2):
	c1_minus = (c1 - c11).T.tolist()[0]
	c2_minus = (c2 - c22).T.tolist()[0]
	sumE1 = 0
	for item in c1_minus:
		sumE1 += item**2
	sumE1 = ( n1*1.0/(n1+n2) ) * sumE1
	sumE2 = 0
	for item in c2_minus:
		sumE2 += item**2
	sumE2 = ( n2*1.0/(n1+n2) ) * sumE2
	sumE = sumE1 + sumE2
	return sumE


#计算所有的E,P,更新sentence_input
def get_new_sentence_input(sentence_input, Ps_children, is_root):
	#重构误差
	E = []
	#生成的父节点
	Ps = []
	#父节点覆盖的词数Ps_children
	# Ps_children = []
	for i in range(len(sentence_input)- 1):
		c1 = sentence_input[i]
		c2 = sentence_input[i+1]
		#判断是否为隐藏层父节点
		p = get_p(c1, c2, w1, b1, is_root)
		Ps.append(p)
		c11,c22 = get_c11_c22_from_p(p, w2, b2, n)
		n1 = Ps_children[i]
		n2 = Ps_children[i+1]
		#不加权重时
		# E_samp = get_E(c1, c2, c11, c22)
		#加权重
		E_samp = get_E_weight(c1, c2, c11, c22, n1, n2)
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
	#更新Ps_children, 左孩子对应的覆盖词数+1,去掉右孩子
	Ps_children[min_index] = Ps_children[min_index] + 1
	Ps_children.remove(Ps_children[min_index + 1])
	return sentence_input, Ps_children


#贪心计算树形最优结构，树中的根节点即为句子的向量表示形式
def RAE_main(sentence_input):
	# 每个词处理成向量形式
	for i in range(len(sentence_input)):
		sentence_input[i] = mat(sentence_input[i]).T
	#处理w1,b1,w2,b2成为向量形式
	w1 = mat(w1)
	b1 = mat(b1).T
	w2 = mat(w2)
	b2 = mat(b2)
	#每个节点覆盖的词数
	Ps_children = [ 1 for i in range(len(sentence_input))]
	#是否为根节点
	is_root = false
	if len(sentence_input) == 2:
		is_root = true
	while(len(sentence_input) > 1):
		sentence_input, Ps_children = get_new_sentence_input(sentence_input, Ps_children, is_root)
		if len(sentence_input) == 2:
			#最后两个点生成根节点，不需要归一化父节点了，只有隐藏层需要
			is_root = true
	#len(sentence_input) = 1 时，得到sentence_input的根节点sentence_input[0]，即sentence_output
	sentence_output = sentence_input[0].T.tolist()[0]
	#sentence_output为数组形式
	return sentence_output




if __name__ == '__main__':
	RAE_hhh()