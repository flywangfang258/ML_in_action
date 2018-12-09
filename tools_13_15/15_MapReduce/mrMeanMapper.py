#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'WF'

import sys
from numpy import mat, mean, power


def read_inpiut(file):
    for line in file:
        yield line.rstrip()

input = read_inpiut(sys.stdin)

input = [float(line) for line in input]
numInputs = len(input)
input = mat(input)
sqInput = power(input, 2)
print('%d\t%f\t%f' % (numInputs, mean(input), mean(sqInput)))
# print(sys.stderr, 'report: still alive')


# python-conda3 mrMeanMapper.py < inputFile.txt
# =>
# 100     0.509570        0.344439
# <_io.TextIOWrapper name='<stderr>' mode='w' encoding='cp936'> report: still alive