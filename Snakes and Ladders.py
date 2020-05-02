#!/usr/bin/env python
# coding: utf-8

# In[4]:


import random
import sys


# In[5]:


def play1(t):
    print(t)
    global s
    if t==1 and s==0:
        s=s+1
    else:
        if s>0 and s<=100:
            s=s+t
            if s==19:
                s=38
            if s==21:
                s=82
            if s==6:
                s=25
            if s==28:
                s=53
            if s==36:
                s=57
            if s==46:
                s=15
            if s==48:
                s=9
            if s==50:
                s=91
            if s==52:
                s=11
            if s==59:
                s=18
            if s==61:
                s=81
            if s==65:
                s=96
            if s==66:
                s=87
            if s==54:
                s=88
            if s==68:
                s=2
            if s==83:
                s=39
            if s==89:
                s=51
            if s==93:
                s=37
            if s==99:
                s=27
    if s>100:
        s=s-t
    print("player 1:",s)
    if s==100:
        print("player 1 is winner")
        sys.exit()
    else:
        p2()

def p1():
    x=str(input())
    if x=='tap':
        t=random.randint(1,6)
        play1(t)

def p2():
    y=str(input())
    if y=='tap':
        q=random.randint(1,6)
        play2(q)
        
def play2(q):
    print(q)
    global a
    if q==1 and a==0:
        a=1
    else:
        if a>0 and a<=100:
            a=a+q
            if a==19:
                a=38
            if a==21:
                a=82
            if a==6:
                a=25
            if a==28:
                a=53
            if a==36:
                a=57
            if a==46:
                a=15
            if a==48:
                a=9
            if a==50:
                a=91
            if a==52:
                a=11
            if a==59:
                a=18
            if a==61:
                a=81
            if a==65:
                a=96
            if a==66:
                a=87
            if a==54:
                a=88
            if a==68:
                a=2
            if a==83:
                a=39
            if a==89:
                a=51
            if a==93:
                a=37
            if a==99:
                a=27
    if a>100:
        a=a-q
    print("player 2:",a)
    if a==100:
        print("Player 2 is winner")
        sys.exit()
    else:
        p1()


a=0
s=0
p1()


# In[ ]:




