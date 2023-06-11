import sqlite3

class Opener:

    def __init__(self, name,root="../"):
        self.CONN = sqlite3.connect(f"{root}{name}.db")
        self.CUR = self.CONN.cursor()
        self.CUR.execute("SELECT * FROM all_tables ")
        tables = self.CUR.fetchall()

    def GetData(self, table, parameter): #wrapper oko Fetch funkcije
        command = f"SELECT * FROM {table} WHERE {parameter}"
        return self.Fetch(command).fetchall()

    def FetchAll(self,table):
        return self.Fetch(f"SELECT * FROM {table}").fetchall()

    def Fetch(self, command):
        self.CUR.execute(command)
        return self.CUR


