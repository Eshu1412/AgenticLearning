from typing import TypedDict

class Person(TypedDict):
    name:str
    age:int

new_person=Person({'name':'Tushar','age':56})

for v,k in new_person.items():
    print(v,k)
