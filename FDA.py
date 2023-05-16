import math
import re
import numpy as np

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


from tools import modular_DB_opener



def binary_search(list, item):
    s = 0
    e = len(list)
    while e - s > 1:
        middle = int((s + e) // 2)
        if item >= list[middle][0]:
            s = middle
            continue
        else:
            e = middle
            continue
    return s

class MatcherDataset(modular_DB_opener.Opener):

    def __init__(self, name, path="../"):
        super().__init__(name, path)
        self.MemoryContainer = super().FetchAll(table="features_with_mean")
        # averages se dodaju kada se izračunaju
        self.avgs = np.asarray(self.MemoryContainer[0])
        self.MemoryContainer = self.MemoryContainer[1:]

    def calc_dev(self, items):
        #assert items.__class__ == np.array()
        return np.sum(items - self.avgs[1:-1]).item()

    def __call__(self, **kwargs):
        if not "feats" in kwargs:
            raise BaseException("please provide list of features of size 10")
        deviation = self.calc_dev(items=kwargs["feats"])
        # možda će biti problema sa binary searchem
        search_res = binary_search(self.MemoryContainer, deviation)
        # razlika između suma devijacija
        match_diff = abs(deviation - self.MemoryContainer[search_res][0])
        match = self.MemoryContainer[search_res]
        return match, match_diff
