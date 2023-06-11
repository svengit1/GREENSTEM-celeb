import numpy as np
import pandas as pd
from tqdm import tqdm

from tools import Modular_DB_creator

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
standardized_landmarks = pd.read_csv("custom/list_landmarks_standardized_celeba.txt")
quality_control_landmarks = pd.read_csv("custom/list_attr_quality_control.txt")
attribute_landmarks = pd.read_csv("existing/list_attr_celeba.txt",delim_whitespace=True)

DataArray = []
line_sums = np.asarray([0] * 10)

for row,line in standardized_landmarks.iterrows():
    line = line.tolist()
    line_array = np.asarray(line[1:])
    DataArray.append([line_array, line[0]])
    line_sums += np.asarray(line_array)

average_features = line_sums / len(DataArray)
for i in tqdm(range(len(DataArray))):
    DataArray[i].append(np.sum(DataArray[i][0] - average_features).item())

TableFeeder = []
for i in tqdm(DataArray):
    row_to_feed = i[0].tolist()
    row_to_feed.insert(0, i[-1])
    id = int(i[-2].split(".")[0]) - 1

    quality_control_labels = quality_control_landmarks.iloc[id].tolist()
    inappropriate = quality_control_labels[2]
    race = quality_control_labels[3]

    gender_age_labels = attribute_landmarks.iloc[id].tolist()
    gender = int(gender_age_labels[21])
    age = int(gender_age_labels[-1])

    row_to_feed.append(inappropriate)
    row_to_feed.append(race)
    row_to_feed.append(age)
    row_to_feed.append(gender)
    row_to_feed.append(i[-2])

    TableFeeder.append(row_to_feed)

# Sortiranje po devijaciji od prosječnih značajki
TableFeeder = sorted(TableFeeder)


average_features = list(average_features)
average_features.insert(0, 0)
for i in range(5):
    average_features.append(0)

base = Modular_DB_creator.Assembler("feat_base")
base.create_tables("features_with_mean", ["mean", 'lefteye_x', 'lefteye_y', 'righteye_x', 'righteye_y', 'nose_x',
                                          'nose_y', 'leftmouth_x', 'leftmouth_y', 'rightmouth_x', 'rightmouth_y',
                                          "inappropriate", "dark_skin", "age", "gender", 'img_src'])
base.AddToTable("features_with_mean", average_features, many=False)
base.AddToTable("features_with_mean", TableFeeder, many=True)


#INFO: Selektor je implementran u zasebnoj datoteci u tools