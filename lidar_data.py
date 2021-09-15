import math
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import os
import socket,threading
from create_obstacles_real import create_ob
from matplotlib.pyplot import MultipleLocator
from TCP import tcplink
from multiprocessing import Process,Pipe
show_animation = True
parent_p_TCP,child_p_TCP=Pipe()

def connect():
    # TCP连接下位机
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('127.0.0.1',9999))
    s.listen(5)
    print('Waiting for conneting')
    sock, addr = s.accept()
    t = threading.Thread(target=tcplink, args=(sock, addr, child_p_TCP))
    t.start()
    #parent_p_TCP.send(b'00')

def main():
    lidar_data=[]
    lidar_data=read_data()
    while True:
        print(lidar_data)
        i for i in range(10) if lidar_data[i][2]<0.5:
            parent_p_TCP.send(b'00000000')
        i for i in range(40,50) if lidar_data[i][2] < 0.5:
            parent_p_TCP.send(b'00003050')
        i for i in range(220,230) if lidar_data[i][2] < 0.5:
            parent_p_TCP.send(b'03000050')

if __name__ == '__main__':
    main()


