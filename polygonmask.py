import sqlite3
import numpy as np

def getPolygonCorners(db_file, pov_id):
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    cur.execute('SELECT PolygonCorners FROM CameraConfigurations')
    r = cur.fetchall()
    listofstr = []
    for cursor in range(len(r)):
        temp = ''.join(r[cursor])
        listofstr.append(temp)
    cor1 = []
    cor2 = []
    cor3 = []
    cor4 = []
    cor5 = []
    cor6 = []
    cor7 = []
    cor8 = []
    cor9 = []
    cor10 = []
    for sep in range(len(listofstr)):

        a = listofstr[sep].split(' ')
        if len(a) > 10:
            cor2.append(a[1])
            cor3.append(a[2])
            cor4.append(a[3])
            cor1.append(a[10] + a[0])
            cor5.append(a[4])
            cor6.append(a[5])
            cor7.append(a[6])
            cor8.append(a[7])
            cor9.append(a[8])
            cor10.append(a[9])
        elif len(a) > 9:
            cor2.append(a[1])
            cor3.append(a[2])
            cor4.append(a[3])
            cor1.append(a[9] + a[0])
            cor5.append(a[4])
            cor6.append(a[5])
            cor7.append(a[6])
            cor8.append(a[7])
            cor9.append(a[8])
            cor10.append('0')
        elif len(a) > 8:
            cor2.append(a[1])
            cor3.append(a[2])
            cor4.append(a[3])
            cor1.append(a[8] + a[0])
            cor5.append(a[4])
            cor6.append(a[5])
            cor7.append(a[6])
            cor8.append(a[7])
            cor9.append('0')
            cor10.append('0')
        elif len(a) > 7:
            cor2.append(a[1])
            cor3.append(a[2])
            cor4.append(a[3])
            cor1.append(a[7] + a[0])
            cor5.append(a[4])
            cor6.append(a[5])
            cor7.append(a[6])
            cor8.append('0')
            cor9.append('0')
            cor10.append('0')
        elif len(a) > 6:
            cor2.append(a[1])
            cor3.append(a[2])
            cor4.append(a[3])
            cor1.append(a[6] + a[0])
            cor5.append(a[4])
            cor6.append(a[5])
            cor7.append('0')
            cor8.append('0')
            cor9.append('0')
            cor10.append('0')
        elif len(a) > 5:
            cor2.append(a[1])
            cor3.append(a[2])
            cor4.append(a[3])
            cor1.append(a[5] + a[0])
            cor5.append(a[4])
            cor6.append('0')
            cor7.append('0')
            cor8.append('0')
            cor9.append('0')
            cor10.append('0')
        elif len(a) > 4:
            cor2.append(a[1])
            cor3.append(a[2])
            cor4.append(a[3])
            cor1.append(a[4] + a[0])
            cor5.append(a[4])
            cor5.append('0')
            cor6.append('0')
            cor7.append('0')
            cor8.append('0')
            cor9.append('0')
            cor10.append('0')
        elif len(a) > 3:
            cor2.append(a[1])
            cor3.append(a[2])
            cor1.append(a[3] + a[0])
            cor4.append('0')
            cor5.append('0')
            cor6.append('0')
            cor7.append('0')
            cor8.append('0')
            cor9.append('0')
            cor10.append('0')
        elif len(a) > 2:
            cor2.append(a[1])
            cor3.append(0)
            cor1.append(a[2] + a[0])
            cor4.append('0')
            cor5.append('0')
            cor6.append('0')
            cor7.append('0')
            cor8.append('0')
            cor9.append('0')
            cor10.append('0')
        elif len(a) > 1:
            cor2.append('0')
            cor3.append('0')
            cor1.append(a[1] + a[0])
            cor4.append('0')
            cor5.append('0')
            cor6.append('0')
            cor7.append('0')
            cor8.append('0')
            cor9.append('0')
            cor10.append('0')
        else:
            cor2.append('0')
            cor3.append('0')
            cor1.append('0')
            cor4.append('0')
            cor5.append('0')
            cor6.append('0')
            cor7.append('0')
            cor8.append('0')
            cor9.append('0')
            cor10.append('0')
    templist = []
    alllist = []
    tmpx=[0]
    tmpy =[0]
    xmins =[]
    xmaxs = []
    ymins = []
    ymaxs = []
    for i in range(103):
        a = []

        a.append(cor1[i])
        a.append(cor2[i])
        a.append(cor3[i])
        a.append(cor4[i])
        a.append(cor5[i])
        a.append(cor6[i])
        a.append(cor7[i])
        a.append(cor8[i])
        a.append(cor9[i])
        a.append(cor10[i])
        templist.append(a)
        for k in range(4):
            alllist.append(templist[i][k].split(';'))

    for i in range(len(alllist)):
        if len(alllist[i]) > 1:
            for k in range(2):
                alllist[i][k] = int(alllist[i][k])
        else:
            alllist[i][0] = int(alllist[i][0])

        if i % 4 is 0 and i > 0:
            xmins.append(min(tmpx))
            xmaxs.append((max(tmpx)))
            ymins.append(min(tmpy))
            ymaxs.append(max(tmpy))
            tmpy = []
            tmpx = []
            tmpx.append(alllist[i][0])
            if len(alllist[i]) > 1:
                tmpy.append(alllist[i][1])
            else:
                tmpy.append(0)
        elif (i + 3) > len(alllist):
            xmins.append(min(tmpx))
            xmaxs.append(max(tmpx))
            ymins.append(min(tmpy))
            ymaxs.append(max(tmpy))
        else:
            if len(alllist[i]) > 1:
                tmpy.append(alllist[i][1])
            else:
                tmpy.append(0)
            tmpx.append(alllist[i][0])

    polygonmask = []
    polygonmask.append((xmins[pov_id],xmaxs[pov_id],ymins[pov_id],ymaxs[pov_id]))
    return polygonmask[0]
