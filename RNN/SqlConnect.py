import pymysql
from pymysql import err


def connect(host, database, user, password):
    """ Connect to MySQL database """
    conn = None
    conn = pymysql.connect(host=host,
                           database=database,
                           user=user,
                           password=password)
    if conn:
        print('Connected to MySQL database')
        return conn
