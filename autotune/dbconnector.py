import mysql.connector 
import psycopg2
from abc import ABC, abstractmethod


class DBConnector(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def connect_db(self):
        pass

    @abstractmethod
    def close_db(self):
        pass

    @abstractmethod
    def fetch_results(self, sql, json=True):
        pass

    @abstractmethod
    def execute(self, sql):
        pass



class MysqlConnector(DBConnector):
    def __init__(self, host='localhost', port=3318, user='root', passwd='', name='tpcc', socket=''):
        super().__init__()
        self.dbhost = host
        self.dbport = port
        self.dbuser = user
        self.dbpasswd = passwd
        self.dbname = name
        self.sock = socket
        self.conn = self.connect_db()
        if self.conn:
            self.cursor = self.conn.cursor()

    def connect_db(self):
        conn = False
        if self.sock:
            conn = mysql.connector.connect(host=self.dbhost,
                                           user=self.dbuser,
                                           passwd=self.dbpasswd,
                                           db=self.dbname,
                                           port=self.dbport,
                                           unix_socket=self.sock)
        else:
            conn = mysql.connector.connect(host=self.dbhost,
                                           user=self.dbuser,
                                           passwd=self.dbpasswd,
                                           db=self.dbname,
                                           port=self.dbport)
        return conn

    def close_db(self):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()

    def fetch_results(self, sql, json=True):
        results = False
        if self.conn:
            self.cursor.execute(sql)
            results = self.cursor.fetchall()
            if json:
                columns = [col[0] for col in self.cursor.description]
                return [dict(zip(columns, row)) for row in results]
        return results

    def execute(self, sql):
        results = False
        if self.conn:
            self.cursor.execute(sql)

class PostgresqlConnector(DBConnector):
    def __init__(self, host='localhost', port=5432, user='root', passwd='', name='tpcc', socket=''):
        super().__init__()
        self.dbhost = host
        self.dbport = port
        self.dbuser = user
        self.dbpasswd = passwd
        self.dbname = name
        self.sock = socket
        self.conn = self.connect_db()
        if self.conn:
            self.cursor = self.conn.cursor()

    def connect_db(self):
        conn = False
        if self.sock:
            conn = psycopg2.connect(host=self.dbhost,
                                    user=self.dbuser,
                                    password=self.dbpasswd,
                                    database=self.dbname,
                                    port=self.dbport,
                                    unix_socket=self.sock)
        else:
            conn = psycopg2.connect(host=self.dbhost,
                                    user=self.dbuser,
                                    password=self.dbpasswd,
                                    database=self.dbname,
                                    port=self.dbport)
        return conn

    def close_db(self):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()

    def fetch_results(self, sql, json=True):
        results = False
        if self.conn:
            self.cursor.execute(sql)
            results = self.cursor.fetchall()
            if json:
                columns = [col[0] for col in self.cursor.description]
                return [dict(zip(columns, row)) for row in results]
        return results

    def execute(self, sql):
        results = False
        if self.conn:
            self.cursor.execute(sql)