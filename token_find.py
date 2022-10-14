class token_find:
    def __init__(self, rank=[0,2], group=[0],signal=False,count=[0],ok=[True]):
        self.world_list = group
        self.rank = rank
        self.group = group
        self.signal = signal
        self.count = count
        self.ok = ok

class select_find:
    def __init__(self, group=[0],signal=False,count=[0],ok=[True]):
        self.world_list = group
        self.ok = ok
        self.target= group
'''
from multiprocessing.managers import BaseManager
if __name__ == "__main__":
   sign = token_find()
   mgr = BaseManager(address=('127.0.0.1', 50000), authkey=b'password')
   mgr.register("getUser", callable=lambda :sign)
   # server永不关闭
   server = mgr.get_server()
   server.serve_forever()
'''