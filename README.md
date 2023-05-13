# GREENSTEM-project

This project uses a CNN to tell you which celebrity you look like! 

Epoha 30 - mse 3.8
(ako neće biti dobart, proći će kroz još 30 epoha)

# NEW 
Model v2, kada se učitava, treba postojati testing folder sa cnn_class.py datotekom i CNN modulom u njoj!
Epoha 9 - mse val: 1.1, test: 2 - bolje ne može bez overfita
plotting datoteka- podaci za prikaz u matplotlibu
# CHANGED
Većina testing i utility datoteka premještena u bbox_component datoteku, 
kada se trebaju koristiti, da bi se avoidao circular import, vratiti ih u root datoteku