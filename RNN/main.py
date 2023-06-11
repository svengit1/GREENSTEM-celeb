import os
from collections import deque
import time
from SqlConnect import *
from univ_rnn_predictor import sample

class Queue(deque):
    def enqueue(self, item):
        self.append(item)

    def deque(self):
        if self:
            return self.popleft()
        return None


RequestsForBot = Queue()
currentRow = 1
is_standby = False

check_time = 2

Login_info = {"host": 'petagimnazija.hr',
              "database": 'petagimnazijahr_AI',
              "user": 'petagimnazijahr_AI',
              "password": 'AIpeta99%'}

conn_src = connect(Login_info["host"],
                   Login_info["database"],
                   Login_info["user"],
                   Login_info["password"])

if not conn_src:
    raise BaseException("Database Error: cannot connect to Base")
conn_curs_1 = conn_src.cursor()

while True:
    if not is_standby:
        cmd = f"SELECT * from QA WHERE Done = 0"
        conn_curs_1.execute(cmd)
        x = conn_curs_1.fetchall()
        if not x:
            is_standby = True
            continue
        print("Found something")
        currentRow += 1
        for y in x:
            RequestsForBot.enqueue(y)

    else:

        print("Standby moment")
        time.sleep(check_time)
        os.system('cls')

        is_standby = False
    conn_src.commit()

    if len(RequestsForBot) > 0:
        print("fulfilling requests..")
        while len(RequestsForBot) > 0:
            A = RequestsForBot.deque()
            #ovo bi trebalo dodati na site
            type = "tadjanovic"
            Res = sample(type,A[2],len_generated_text=1000,scale_factor=1.0)
            conn_curs_1.execute(f"UPDATE QA SET Answer = '{Res}', Done = 1 WHERE QAID = {A[0]}")
            conn_src.commit()

