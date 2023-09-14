import numpy
import pandas
from src.utils import load_house_price_dataset
df_house = load_house_price_dataset()


def questao1():
    count = df_house['price'].count()
    avg = df_house['price'].mean()
    std = df_house['price'].std()
    cv = std/avg * 100
    return count, avg, std, cv

count, avg, std, cv = questao1()
print(count)
print(avg)
print(std)
print(cv)
def questao2():



    return None
def questao3():
    return None

print("Questao1: ",questao1())
print("Questao2: ",questao2())
print("Questao3: ",questao3())



