import math
import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm

from tools import Modular_DB_creator, modular_DB_opener

# algoritam će naći najbolje poklapajuću sumu standardnih devijacija facial značajki od meana,
# ovaj algoritam računa na to da će postojati vrlo slične fotografije radi količine fotografija u datasetu
# prednost ove metode nad metodom uspoređivanja suma je ta što je zbog prosjeka i std devijaije centar centriran na 0
# te su odstupanja malnja i još negativna i pozitina, stoga su sume reprezentativnije pri usporeedbi,
# dodatno, to daje mali time boost

# IMPLEMENTACIJA
# svi feature podaci su u bazi, uz to nalazi se i paragraf sa sortiranim devijacijama od prosjeka.
# KOMPLEKSNIJI BACKUP NAČINI
# istrenirati NN da prepoznaje osobe sa slika
# Napraviti alogritam koji računa euklideanske usaljenosti između značajki te tako radi drugačiju sumu



# PREPROCESSING
dat = open("list_landmarks_standardized_celeba.txt", "r")
dat2 = open("labels_qc.csv","r").readlines()[1:]
dat3 = open("list_attr_celeba.txt","r").readlines()[1:]
print(open("list_attr_celeba.txt","r").readlines()[0].split(" ").index("Male"))
DataArray = []
linesums = np.asarray([0] * 10)
lines = dat.readlines()
print(lines[1].split())
print(lines[0:3])
for rl in lines[1:]:
    DataArray.append((np.asarray(list(map(int, rl.split(",")[1:]))), rl.split(",")[0]))
    linesums += np.asarray(DataArray[-1][0])

avgs = linesums / len(DataArray)
d = {-1:0,1:1}
print(avgs)
for i in tqdm(range(len(DataArray))):
    DataArray[i] = list(DataArray[i])
    DataArray[i].append(np.sum(DataArray[i][0] - avgs).item())
TableFeeder= []
for i in tqdm(DataArray):
    new = [round(j) for j in i[0]]
    new.insert(0,i[-1])
    id = int(i[-2].split(".")[0])-1
    raceLabs = dat2[id].split(",")
    genLab = dat3[id].replace("  "," ").split(" ")
    gender =int(genLab[21])
    age = int(genLab[-1].replace("\n",""))
    new.append(int(raceLabs[-2]))
    new.append(int(raceLabs[-1][0]))
    new.append(d[age])
    new.append(d[gender])
    new.append(i[-2])
    TableFeeder.append(new)
TableFeeder = sorted(TableFeeder)
def binary_search(list, item):
    s = 0
    e = len(list) / 2
    while e - s > 0:
        middle = (s + e) // 2
        if item >= list[middle][0]:
            s = middle
            continue
        else:
            e = middle
            continue
    return list[s]

avgs = list(avgs)
avgs.insert(0,0)
for i in range(5):
    avgs.append(0)
base = Modular_DB_creator.Assembler("feat_base")
base.create_tables("features_with_mean",["mean",'lefteye_x', 'lefteye_y', 'righteye_x', 'righteye_y', 'nose_x',
                                    'nose_y', 'leftmouth_x', 'leftmouth_y', 'rightmouth_x', 'rightmouth_y',
                   "inappropriate","dark_skin","age","gender",'img_src'])
base.AddToTable("features_with_mean",avgs,many=False)
base.AddToTable("features_with_mean",TableFeeder,many=True)

##Spremi podatke u bazu, table u koji se sprema se zove feat_holder

# Izvlačitelj iz baze
class MatcherDataset(modular_DB_opener.Opener):

    def __init__(self, name, path="../"):
        super().__init__(name, path)
        self.MemoryContainer = super().FetchAll(table="feat_holder")
        # averages se dodaju kada se izračunaju
        self.avgs = np.asarray(self.MemoryContainer[0])
        self.MemoryContainer = self.MemoryContainer[1:]

    def calc_dev(self, items):
        assert items.__class__ == np.array
        return np.sum(items - self.avgs[1:-1]).item()

    def __call__(self, **kwargs):
        if not "feats" in kwargs:
            raise BaseException("please provide list of features of size 10")
        deviation = self.calc_dev(items=kwargs["feats"])
        # možda će biti problema sa binary searchem
        search_res = binary_search(self.MemoryContainer, deviation)
        # razlika između suma devijacija
        match_diff = abs(deviation - search_res)
        match = self.MemoryContainer[search_res]
        return match, match_diff
