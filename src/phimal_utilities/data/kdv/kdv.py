import torch
import numpy as np


def soliton(x, t, c, a):
    '''Single soliton solution of kdv'''
    u = 1/2 * c / torch.cosh(np.sqrt(c) / 2 * (x - c * t - a))**2
    return u


def KdVSoliton(x, t, c, a, b):
    '''Solution of the kortweg de vries for two solitons. kdv is nonlinear,
    but solitons can be superpositioned :). c is wavespeed, a and b initial position
    of each soliton.

    Source: https://en.wikipedia.org/wiki/Korteweg%E2%80%93de_Vries_equation'''
    u = soliton(x, t, c, a)
    return u
