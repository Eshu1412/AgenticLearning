from pydantic import BaseModel,EmailStr,Field
from typing import Optional

class Student(BaseModel):
    name:str
    age:Optional[int]=None
    email:EmailStr
    cgpa:float=Field(lt=10,gt=0,description='Student CGPA')

students={'name':'Tushar Maurya',
'age':24,
'email':'mauryatushar115@gmail.com',
'cgpa':8.95
}

stu=Student(**students)
student_json=stu.model_dump_json()
print(student_json)
print(type(student_json))