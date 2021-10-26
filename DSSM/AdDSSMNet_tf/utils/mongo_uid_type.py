# -*- coding: UTF-8 -*-
import os
import sys
import pymongo

curdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(curdir)
from config import Config


def retry(func):
    def wrapper(server, *args, **kwargs):
        try:
            return func(server, *args, **kwargs)
        except:
            server.init()
            return func(server, *args, **kwargs)

    return wrapper


class MongoServer(object):
    def __init__(self, env, table, section='mongo'):
        self.section = section
        self.env = env
        self.table_name = table
        self.config = Config(type=self.env, section=self.section)
        self.init()

    def init(self):
        self.conn = pymongo.MongoClient(self.config.get('url'), minPoolSize=1)
        self.db = self.conn[self.config.get('db')]
        self.table = self.db[self.table_name]

    @retry
    def find(self, *args, **kwargs):
        cursor = self.table.find(*args, **kwargs)
        return cursor

    @retry
    def find_one(self, *args, **kwargs):
        data = self.table.find_one(*args, **kwargs)
        return data


if __name__ == '__main__':
    config = Config(type='35', section='mongo')
    # config.showAllSections()
    mongo_server = MongoServer(env='35', table='active_user')
    data = mongo_server.find_one({})
    print(data)
    pass
