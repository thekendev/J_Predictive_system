import pymongo
import numpy as np
import pandas as pd


#Reading the csv file
from pymongo pd.read_csv('heart_disease_data.csv')

#saving all entities in heart disease data as a class
class HeartDiesease(object):
    def __init__(self, cp, age, sex, ttb, chol, fbs, rg, tch, exec, op, sp, ca, ha, tg):
        self.chestpain = cp
        self.age = age
        self.sex = sex
        self.trestbps = ttb
        self.chol = chol
        self.fbs = fbs
        self.restecg = rg
        self.thalach = tch
        self.exang = exec
        self.oldpeak = op
        self.slope = sp
        self.ca = ca
        self.thal = ha
        self.target = tg


chest_pain = HeartDiesease()
age = HeartDiesease()
sex = HeartDiesease()
trestbps = HeartDiesease()
chol = HeartDiesease()
fbs = HeartDiesease()
restecg = HeartDiesease()
thalach = HeartDiesease()
exang = HeartDiesease()
oldpeak = HeartDiesease()
slope = HeartDiesease()
ca = HeartDiesease()
thal = HeartDiesease()
target HeartDiesease()

