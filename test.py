import pymongo

test = pymongo.MongoClient("mongodb://localhost:27017/")

db = test['test_db']

collection = db['test_collection']

mydict = {"name": "John", "address": "Highway 37"}

mydict2 = [
    {"name": "Amy", "address": "Apple st 652"},
    {"name": "Hannah", "address": "Mountain 21"}]

collection.insert(mydict2)
print([x['address'] for x in collection.find()])
