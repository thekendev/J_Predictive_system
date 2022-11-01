#importing the dependencies
import pandas as pd
import json
import pymongo
import pprint
from pymongo import MongoClient


client = MongoClient()
db = client['newdata']
collection = db["allData"]

heart_data = pd.read_csv("heart_disease_data.csv")
data = heart_data.to_dict(orient="records")
post = collection.heart_data.insert_many(data)







        #ALTERNATIVE WAY
#class Database(object):
   # URI = "mongodb://localhost:27017"
  #  DATABASE = "None"

   # @staticmethod
   # def initialize():
    #    client = pymongo.MongoClient(Database.URI)
    #    Database.DATABASE = client["heart_prediction"]
   # @staticmethod
  #  def insert(collection, data):
   #     Database.DATABASE[collection].insert(data)
   # @staticmethod
  #  def find(collection, query):
     #   return  Database.DATABASE[collection].find(query)
    #@staticmethod
   # def find_one(collection, query):
       # return  Database.DATABASE[collection].find_one(query)




#Userprofile
#password- ****
#username- patient
