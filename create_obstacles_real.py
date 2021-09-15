import numpy as np
'''
obstacles_list=[(0.5, 2.5), (0.5, 3.5), (0.5, 4.5), (0.5, 5.5), (0.5, 6.5), (0.5, 7.5), (0.5, 8.5),
     (0.5, 9.5), (0.5, 10.5), (0.5, 11.5), (0.5, 12.5), (0.5, 13.5), (0.5, 14.5), (0.5, 15.5), (0.5, 16.5), (0.5, 17.5),
     (0.5, 18.5), (0.5, 19.5), (0.5, 20.5), (0.5, 21.5), (0.5, 22.5), (0.5, 23.5), (0.5, 24.5), (0.5, 25.5),
     (0.5, 26.5), (0.5, 27.5), (0.5, 28.5), (0.5, 30.5), (1.5, 30.5), (2.5, 30.5), (3.5, 30.5), (4.5, 30.5),
     (5.5, 30.5), (6.5, 30.5), (7.5, 30.5), (8.5, 30.5), (9.5, 30.5), (10.5, 30.5), (11.5, 30.5), (12.5, 30.5),
     (13.5, 30.5), (14.5, 30.5), (15.5, 30.5), (16.5, 30.5), (17.5, 30.5), (18.5, 30.5), (19.5, 30.5), (20.5, 30.5),
     (21.5, 30.5), (22.5, 30.5), (23.5, 30.5), (24.5, 30.5), (25.5, 30.5), (26.5, 30.5), (27.5, 30.5), (28.5, 30.5),
     (30.5, 30.5), (30.5, 29.5), (30.5, 28.5), (30.5, 27.5), (30.5, 26.5), (30.5, 25.5), (30.5, 24.5), (30.5, 23.5),
     (30.5, 22.5), (30.5, 21.5), (30.5, 20.5), (30.5, 19.5), (30.5, 18.5), (30.5, 17.5), (30.5, 16.5), (30.5, 15.5),
     (30.5, 14.5), (30.5, 13.5), (30.5, 12.5), (30.5, 11.5), (30.5, 10.5), (30.5, 9.5), (30.5, 8.5), (30.5, 7.5),
     (30.5, 6.5), (30.5, 5.5), (30.5, 4.5), (30.5, 3.5), (30.5, 2.5), (30.5, 0.5), (29.5, 0.5), (28.5, 0.5),
     (27.5, 0.5), (26.5, 0.5), (25.5, 0.5), (24.5, 0.5), (23.5, 0.5), (22.5, 0.5), (21.5, 0.5), (20.5, 0.5),
     (19.5, 0.5), (18.5, 0.5), (17.5, 0.5), (16.5, 0.5), (15.5, 0.5), (14.5, 0.5), (13.5, 0.5), (12.5, 0.5),
     (11.5, 0.5), (10.5, 0.5), (9.5, 0.5), (8.5, 0.5), (7.5, 0.5), (6.5, 0.5), (5.5, 0.5), (4.5, 0.5), (3.5, 0.5),
     (2.5, 0.5), (2.5, 2.5), (3.5, 2.5), (4.5, 2.5), (5.5, 2.5), (10.5, 10.5), (10.5, 11.5),
                                                                                 (10.5, 12.5), (10.5, 13.5),
                                                                                 (10.5, 14.5), (10.5, 15.5),
                                                                                 (10.5, 16.5), (10.5, 17.5),
                                                                                 (10.5, 18.5), (11.5, 18.5),
                                                                                 (12.5, 18.5), (13.5, 18.5),
                                                                                 (14.5, 18.5), (15.5, 18.5),
                                                                                 (16.5, 18.5), (17.5, 18.5),
                                                                                 (18.5, 18.5), (18.5, 17.5),
                                                                                 (18.5, 16.5), (18.5, 15.5),
                                                                                 (18.5, 14.5), (18.5, 13.5),
                                                                                 (18.5, 12.5), (18.5, 11.5),
                                                                                 (18.5, 10.5), (26.5, 9.5),
                                                                                                 (9.5, 22.5),
                                                                                                 (0.5, 4.5), (3.5, 2.5),
                                                                                                 (13.5, 26.5),
                                                                                                 (14.5, 6.5),
                                                                                                 (3.5, 12.5),
                                                                                                 (15.5, 2.5),
                                                                                                 (24.5, 2.5),
                                                                                                 (0.5, 22.5),
                                                                                                 (7.5, 23.5),
                                                                                                 (20.5, 11.5),
                                                                                                 (7.5, 16.5),
                                                                                                 (3.5, 5.5),
                                                                                                 (13.5, 9.5),
                                                                                                 (15.5, 5.5),
                                                                                                 (27.5, 12.5),
                                                                                                 (25.5, 12.5),
                                                                                                 (5.5, 7.5),
                                                                                                 (9.5, 16.5),
                                                                                                 (15.5, 13.5),
                                                                                                 (12.5, 19.5),
                                                                                                 (15.5, 28.5),
                                                                                                 (23.5, 28.5),
                                                                                                 (13.5, 24.5),
                                                                                                 (26.5, 29.5),
                                                                                                 (19.5, 9.5),
                                                                                                 (2.5, 27.5),
                                                                                                 (5.5, 5.5),
                                                                                                 (15.5, 21.5),
                                                                                                 (14.5, 0.5),
                                                                                                 (11.5, 5.5),
                                                                                                 (5.5, 9.5), (1.5, 6.5),
                                                                                                 (21.5, 22.5),
                                                                                                 (2.5, 3.5),
                                                                                                 (10.5, 10.5),
                                                                                                 (28.5, 12.5),
                                                                                                 (1.5, 18.5),
                                                                                                 (24.5, 11.5),
                                                                                                 (10.5, 22.5),
                                                                                                 (14.5, 27.5),
                                                                                                 (14.5, 20.5),
                                                                                                 (18.5, 2.5),
                                                                                                 (17.5, 5.5),
                                                                                                 (10.5, 17.5),
                                                                                                 (27.5, 28.5),
                                                                                                 (4.5, 24.5),
                                                                                                 (23.5, 8.5),
                                                                                                 (17.5, 3.5),
                                                                                                 (1.5, 6.5),
                                                                                                 (22.5, 17.5),
                                                                                                 (16.5, 26.5),
                                                                                                 (0.5, 21.5),
                                                                                                 (0.5, 19.5),
                                                                                                 (28.5, 26.5),
                                                                                                 (21.5, 23.5),
                                                                                                 (11.5, 15.5),
                                                                                                 (3.5, 6.5), (4.5, 9.5),
                                                                                                 (20.5, 18.5),
                                                                                                 (14.5, 12.5),
                                                                                                 (27.5, 23.5),
                                                                                                 (2.5, 28.5),
                                                                                                 (28.5, 1.5),
                                                                                                 (1.5, 11.5),
                                                                                                 (27.5, 29.5),
                                                                                                 (10.5, 22.5),
                                                                                                 (19.5, 1.5),
                                                                                                 (2.5, 4.5),
                                                                                                 (15.5, 18.5),
                                                                                                 (7.5, 10.5),
                                                                                                 (11.5, 29.5),
                                                                                                 (4.5, 20.5),
                                                                                                 (14.5, 19.5),
                                                                                                 (29.5, 4.5),
                                                                                                 (20.5, 17.5),
                                                                                                 (8.5, 5.5),
                                                                                                 (9.5, 11.5),
                                                                                                 (24.5, 18.5)]
'''
obstacles_list=[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (1, 7), (2, 7),
                  (3, 7), (4, 7), (5, 7),(6, 7), (7, 7), (8, 7), (9, 7), (9, 8), (9, 9), (10, 9),
                  (11, 9), (12, 9), (13, 9), (14, 9), (15, 9), (16, 9), (17, 9), (18, 9), (19, 9),
                  (20, 9), (21, 9), (22, 9), (23, 9), (24, 9), (25, 9), (26, 9), (27, 9), (28, 9),
                  (29, 9), (29, 8), (29, 7), (29, 6), (30, 6), (30, 5), (30, 4), (30, 3), (30, 2),
                  (30, 1), (30, 0), (30, -1), (30, -2), (30, -3), (29, -3), (28, -3), (27, -3),
                  (26, -3), (25, -3), (24, -3), (23, -3), (22, -3), (21, -3), (21, -2), (21, -1),
                  (21, 0), (20, 0), (19, 0), (18, 0), (17, 0), (16, 0), (15, 0), (14, 0), (13, 0),
                  (12, 0), (11, 0), (10, 0), (9, 0), (8, 0), (7, 0), (6, 0), (5, 0), (4, 0), (3, 0),
                  (2, 0), (1, 0), (0, 0),(11, 6), (11, 5), (11, 4), (12, 4), (13, 4), (14, 4),
                  (14, 5), (13, 5), (12, 5)]

length=len(obstacles_list)
print('length of obstacle_list:',len(obstacles_list))
ob=np.zeros((length,2))

def create_ob():
    for i in range(length):
        ob[i][0]=obstacles_list[i][0]
        ob[i][1]=obstacles_list[i][1]

    ob_txt=open('./ob.txt','w')
    ob_txt.write(str(ob))

    ob_txt.close()

    return ob