from pydantic import BaseModel, Field
from typing import Optional


class student(BaseModel):
    name : str
    age : float

new_student = {"name" : "bhavisya", "age" :45 }

student = student(**new_student)

print(student.name)
print(student.age)



