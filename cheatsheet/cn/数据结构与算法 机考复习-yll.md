

# 数据结构与算法 机考复习-yll

## 一  python入门

## 1模拟

模拟问题一般通过使用列表、字典等简单数据结构作为支撑，使用for循环，while循环，定义递归函数等基本的算法进行实现，其中也需要利用一些特殊的python内置函数与算法编写方式进行时间，来达到时间复杂度与空间复杂度的要求。该类题目与计概题目重合较多，更多地作为与前面学习的衔接内容，同时让自己对本来不熟悉的python内置函数的使用以及定义递归函数的时间空间复杂度的感知更加敏锐。

做这些题目的过程中，我经常需要在网上查找许多python基本内置函数的用法，多次练习之后，也加深了我对许多经典语句的使用熟练度，与此同时，本人拥有的数学竞赛背景也使我在解决一些特殊题目的时候可以更快的想出一些更简便的算法。

在复习过程中，我重点的会提炼一些重要的python内置函数的用法语句，加入到我的cheatsheet中，以免考试时忘记。

#### 01035 拼写检查  http://cs101.openjudge.cn/2024sp_routine/01035/

利用for循环模拟检查拼写的过程

```python
A=[]
B=[]
while True:
    voc=input()
    if voc!='#':
        A.append(voc)
    else:break
while True:
    voc=input()
    if voc!='#':
        B.append(voc)
    else:break
for voc2 in B: 
    if voc2 in A:
        print(voc2+' is correct')
    else:
        out=[]
        for voc1 in A:
            k=len(voc2)-len(voc1)
            if k==0:
                v=0
                for t in range(len(voc2)):
                    if voc2[t]!= voc1[t]:
                        v+=1
                if v<=1:
                    out.append(voc1)
            if k ==1:
                for t in range(len(voc2)):
                    if t==len(voc2)-1:
                        out.append(voc1)
                    elif voc2[t]!=voc1[t]:
                        voc3=voc2[:t]+voc2[t+1:]
                        if voc3==voc1:
                            out.append(voc1)
                        break
            if k ==-1:
                for t in range(len(voc2)+1):
                    if t==len(voc2):
                        out.append(voc1)
                    elif voc2[t]!=voc1[t]:
                        voc3=voc1[:t]+voc1[t+1:]                       
                        if voc2==voc3:
                            out.append(voc1)
                        break
        print(voc2+': '+' '.join(out))
```

#### 01426 Find The Multiple http://cs101.openjudge.cn/2024sp_routine/01426/

寻找被n整除的01数

我的代码使用了枚举法，非常垃圾，就不放在这里了

#### 01936 全在其中 http://cs101.openjudge.cn/2024sp_routine/01936/

判断两个字符串是不是互为子串，使用index函数+递归构造判断函数

```python
def f(s,t):
    if s=='':
        return 'Yes'
    else:
        try:a=t.index(s[0])
        except ValueError:return 'No'
        else:return f(s[1:],t[a+1:])
while True:
    try:
        s,t=input().split()
    except EOFError:
        break
    else:
        print(f(s,t))
```

#### 01941 The Sierpinski Fractal http://cs101.openjudge.cn/2024sp_routine/01941/

打出三角形，使用递归函数实现

```python
def draw(x):
    if x==1:
        return [' /\\','/__\\']
    else:
        net1=[]
        net2=[]
        t=0
        for j in draw(x-1):
            net1.append(' '*(2**(x-1))+j)
            net2.append(j+' '*(2**(x-1)-1-t)+j)
            t+=1
        return net1+net2
while True:
    x=int(input())
    if x:
        for j in draw(x):
            print(j)
        print(' ')
    else:
        break
```

#### 02039 反反复复 http://cs101.openjudge.cn/2024sp_routine/02039/

建立矩阵，重打数据

```python
i=int(input())
s=input()
output=[0]*len(s)
for j in range(len(s)):
    if j//i%2==0:
        output[(j//i)+(len(s)//i)*(j%i)]=s[j]
    else:
        output[(j//i)+(len(s)//i)*(i-1-j%i)]=s[j]
print(''.join(output))
```

#### 02733 判断闰年 http://cs101.openjudge.cn/2024sp_routine/02733/

利用python的%运算实现

```python
y=int(input())
if y%4==0:
    if y%100 !=0:
        print('Y')
    else:
        if y%400 != 0:
            print('N')
        else:
            if y %3200 != 0:
                print('Y')
            else:
                print('N')
else:
    print('N')
```

#### 02734 从十进制到八进制 http://cs101.openjudge.cn/2024sp_routine/02734/

利用进制转换函数oct()实现

```python
a=int(input())
b=oct(a)
c=b[2:]
print(c)
```

#### 02746 约瑟夫问题 http://cs101.openjudge.cn/2024sp_routine/02746/

使用队列模拟问题，从而实现

```python
t=True
while t:
    s=list(map(int,input().split()))
    if s==[0,0]:
        t=False
        break
    mon=[]
    for i in range(1,s[0]+1):
        mon.append(i)
    n=s[1]
    i=1
    while len(mon)>1:
        if i%n==0:
            mon.pop(0)
        else:
            mon.append(mon[0])
            mon.pop(0)
        i+=1
    print(mon[0])
```

#### 02760 数字三角形 http://cs101.openjudge.cn/2024sp_routine/02760/

对每个位置设置递归函数，因为堆层数设置递归函数导致了time limit exceed

```python
n=int(input())
numlst=[]
dp=[[-1]*n for _ in range(n)]
def sum(y,x):
    if dp[y][x]==-1:
        if y==n-1:
            tempsum=numlst[y][x]
            dp[y][x]=tempsum
            return tempsum
        else:
            tempsum=max(sum(y+1,x),sum(y+1,x+1))+numlst[y][x]
            dp[y][x]=tempsum
            return tempsum
    else:
        return dp[y][x]
for _ in range(n):
    numlst.append(list(map(int,input().split())))
print(sum(0,0))
```

#### 02783 Holiday Hotel http://cs101.openjudge.cn/2024sp_routine/02783/

找出最划算的那些酒店：不存在任何一个酒店，距离和价格同时小于它

利用sort函数+key=lambda函数实现排序，然后便可直接输出，其他方法容易time limited exceed 比较的时候也要注意>与>=的区别

```python
while True:
    t=int(input())
    if t==0:
        break
    else:
        hts=[]
        for j in range(t):
            ht=list(map(int,input().split()))
            hts.append(ht)
        hts.sort(key=lambda x:(x[0],x[1]))
        f=0
        for i in range(t):
            if i>0 and hts[i-f][1]>=hts[i-1-f][1]:
                hts.pop(i-f)
                f+=1
        print(len(hts))
```

#### 02808 校门外的树 http://cs101.openjudge.cn/2024sp_routine/02808/

使用列表模拟，解决问题

```python
tre=list(map(int,input().split()))
trees=[1]*(tre[0]+1)
for i in range(tre[1]):
    sub=list(map(int,input().split()))
    for j in range(sub[0],sub[1]+1):
        trees[j]=0
s=0
for t in trees:
    s+=t
print(s)
```

#### 02810 完美立方 http://cs101.openjudge.cn/2024sp_routine/02810/

使用枚举实现

```python
tris=dict()
nums=[]
n=int(input())
nums_1=[]
nums_2=dict()
for i in range(1,n+1):
    nums_1.append(i**3)
    nums_2[i**3]=i
for i in range(2,n):
    for j in range(i,n):
        for k in range(j,n):
            c=i**3+j**3+k**3
            if c<=n**3 and c in nums_1:
                y=nums_2[c]
                if y in tris.keys():
                    tris[y].append([i,j,k])
                else:
                    nums.append(y)
                    tris[y]=[[i,j,k]]
nums.sort()
for d in nums:
    for f in range(len(tris[d])):
        s='Triple = ('
        for h in range(len(tris[d][f])):
            if h ==0:
                s=s+str(tris[d][f][h])
            else:
                s=s+','+str(tris[d][f][h])
        s=s+')'
        s='Cube = '+str(d)+ ', '+s
        print(s)
```

#### 03253 约瑟夫问题 No.2 http://cs101.openjudge.cn/2024sp_routine/03253/

使用队列模拟实现

```python
f=True
while f:
    stus=list(map(int,input().split()))
    n=stus[0]
    p=stus[1]
    m=stus[2]
    if n==0 and p==0 and m==0:
        f=False
        break
    stu=[]
    pstu=[]
    a=0
    for i in range(n):
        stu.append(i+1)
    while len(stu)>0:
        a+=1
        if a%m==0:
            pstu.append(stu.pop(0))
        else:
            stu.append(stu[0])
            stu.pop(0)
    s=''
    for j in pstu:
        t=(j-2+p)%n+1
        s=s+','+str(t)
    print(s[1:])   
```

#### 04067 回文数字 http://cs101.openjudge.cn/2024sp_routine/04067/

通过对字符串遍历实现

```python
while True:
    try:
        s=input()
    except EOFError:
        break
    else:
        n=len(s)
        t=True
        for i in range((n+1)//2):
            if s[i] != s[n-1-i]:
                t=False
                break
        if t :
            print('YES')
        else:
            print('NO')
```

#### 04093 倒排索引查询 http://cs101.openjudge.cn/2024sp_routine/04093/

使用函数实现索引查询

```python
def f(x):
    output=[]
    for i in search.keys():
        f=True
        for j in range(a):
            if x[j]==-1 and search[i][j] == 1:
                f=False
            if x[j]==1 and search[i][j] == -1:
                f=False
        if f:output.append(i)
    output.sort()
    outputs=[]
    for k in output:
        outputs.append(str(k))
    if output:print(' '.join(outputs))
    else:print('NOT FOUND')
search=dict()
a=int(input())
notes=[]
finds=[]
for i in range(a):
    notes_0=list(map(int,input().split()))
    c=notes_0.pop(0)
    for j in notes_0:
        if j in search.keys():
            search[j][i]=1
        else:
            search[j]=[-1]*a
            search[j][i]=1
b=int(input())
for i in range(b):
    finds_0=list(map(int,input().split()))
    finds.append(finds_0)
for i in finds:
    f(i)
```

#### 04110 圣诞老人的礼物 http://cs101.openjudge.cn/2024sp_routine/04110/

经典的使用贪心算法实现的题目

```python
santa=list(map(int,input().split()))
w=santa[1]
can=dict()
money_w=dict()
money=[]
m=0
for i in range(santa[0]):
    canm=list(map(int,input().split()))
    can[i]=canm
    n=canm[0]/canm[1]
    if n in money_w.keys():
        money_w[n]+=canm[1]
    else:
        money_w[n]=canm[1]
        money.append(n)
money.sort(reverse=True)
for n in money:
        if w>money_w[n]:
            w-=money_w[n]
            m+=n*money_w[n]
        elif w>0:
            m+=n*w
            w=0
print(f'{m:.1f}')
```

#### 04137 最小新整数 http://cs101.openjudge.cn/2024sp_routine/04137/

简单思考过后得出，可以通过删去比后一位大的数字实现

```python
t = int(input())
for i in range(t):
    n,k = input().split()
    for j in range(int(k)):
        f=True
        for s in range(len(n)-1):
            if f and int(n[s+1])<int(n[s]):
                n=n[:s]+n[s+1:]
                f=False
        if f:
            n=n[:-1]
    print(n)
```

#### 04143 和为给定数 http://cs101.openjudge.cn/2024sp_routine/04143/

使用双指针实现查找

```python
def f(nums, target):
    nums.sort()
    i = 0
    j = 0
    l = []
    for k in range(0, (len(nums) - 1)):
        if nums[k] + nums[k + 1] >= target:
            i = k
            j = k + 1
            break
    while i >= 0 and j < len(nums):
        if nums[i] + nums[j] < target:
            j += 1
        elif nums[i] + nums[j] > target:
            i -= 1
        else:
            l.append([nums[i],nums[j]])
            i -= 1
            j += 1
    if l:
        return l[-1]
    else:
        return None
n = int(input())
l = list(map(int,input().split()))
m = int(input())
if f(l,m):
    print(f(l,m)[0],f(l,m)[1])
else:
    print('No')
```

#### 04146 数字方格 http://cs101.openjudge.cn/2024sp_routine/04146/

使用在模30意义下，可以直接通过数学规律实现问题

```python
n=int(input())
t=n%30
A=[0,1,2,3,4,6,8,9,11,12,13,14,15,16,17,18,19,21,23,24,26,28,29]
if t in A:
    s=(((t*3)//5))*5
else:
    s=(((t*3)//5)-1)*5
m=90*(n//30)+s
print(m)
```

#### 05343 用队列对扑克牌排序 http://cs101.openjudge.cn/2024sp_routine/05343/

直接暴力实现，相对简单

```python
Queue1=[]
Queue2=[]
Queue3=[]
Queue4=[]
Queue5=[]
Queue6=[]
Queue7=[]
Queue8=[]
Queue9=[]
QueueA=[]
QueueB=[]
QueueC=[]
QueueD=[]
n=int(input())
cards=input().split()
for k in range(n):
    card=cards[k]
    if card[1]=='1':Queue1.append(card)
    if card[1]=='2':Queue2.append(card)
    if card[1]=='3':Queue3.append(card)
    if card[1]=='4':Queue4.append(card)
    if card[1]=='5':Queue5.append(card)
    if card[1]=='6':Queue6.append(card)
    if card[1]=='7':Queue7.append(card)
    if card[1]=='8':Queue8.append(card)
    if card[1]=='9':Queue9.append(card)
print('Queue1:'+' '.join(Queue1))
print('Queue2:'+' '.join(Queue2))
print('Queue3:'+' '.join(Queue3))
print('Queue4:'+' '.join(Queue4))
print('Queue5:'+' '.join(Queue5))
print('Queue6:'+' '.join(Queue6))
print('Queue7:'+' '.join(Queue7))
print('Queue8:'+' '.join(Queue8))
print('Queue9:'+' '.join(Queue9))
cards_s=[]
for card in Queue1:
    if card[0]=='A':QueueA.append(card)
    if card[0]=='B':QueueB.append(card)
    if card[0]=='C':QueueC.append(card)
    if card[0]=='D':QueueD.append(card)
for card in Queue2:
    if card[0]=='A':QueueA.append(card)
    if card[0]=='B':QueueB.append(card)
    if card[0]=='C':QueueC.append(card)
    if card[0]=='D':QueueD.append(card)
for card in Queue3:
    if card[0]=='A':QueueA.append(card)
    if card[0]=='B':QueueB.append(card)
    if card[0]=='C':QueueC.append(card)
    if card[0]=='D':QueueD.append(card)
for card in Queue4:
    if card[0]=='A':QueueA.append(card)
    if card[0]=='B':QueueB.append(card)
    if card[0]=='C':QueueC.append(card)
    if card[0]=='D':QueueD.append(card)
for card in Queue5:
    if card[0]=='A':QueueA.append(card)
    if card[0]=='B':QueueB.append(card)
    if card[0]=='C':QueueC.append(card)
    if card[0]=='D':QueueD.append(card)
for card in Queue6:
    if card[0]=='A':QueueA.append(card)
    if card[0]=='B':QueueB.append(card)
    if card[0]=='C':QueueC.append(card)
    if card[0]=='D':QueueD.append(card)
for card in Queue7:
    if card[0]=='A':QueueA.append(card)
    if card[0]=='B':QueueB.append(card)
    if card[0]=='C':QueueC.append(card)
    if card[0]=='D':QueueD.append(card)
for card in Queue8:
    if card[0]=='A':QueueA.append(card)
    if card[0]=='B':QueueB.append(card)
    if card[0]=='C':QueueC.append(card)
    if card[0]=='D':QueueD.append(card)
for card in Queue9:
    if card[0]=='A':QueueA.append(card)
    if card[0]=='B':QueueB.append(card)
    if card[0]=='C':QueueC.append(card)
    if card[0]=='D':QueueD.append(card)
print('QueueA:'+' '.join(QueueA))
print('QueueB:'+' '.join(QueueB))
print('QueueC:'+' '.join(QueueC))
print('QueueD:'+' '.join(QueueD))
for card in QueueA:cards_s.append(card)
for card in QueueB:cards_s.append(card)
for card in QueueC:cards_s.append(card)
for card in QueueD:cards_s.append(card)
print(' '.join(cards_s))
```

#### 05345 位查询 http://cs101.openjudge.cn/2024sp_routine/05345/

用列表存储数字，使用for循环操作与查找

```python
s=list(map(int,input().split()))
nums=list(map(int,input().split()))
for j in range(s[1]):
    d=input().split()
    if d[0]== 'C':
        for i in range(s[0]):
            nums[i]+=int(d[1])
            if nums[i]>65536:
                nums[i]-=65536
    else:
        k=0
        for i in range(s[0]):
            if nums[i]%(2**(int(d[1])+1))!=nums[i]%(2**(int(d[1]))):
                k+=1
        print(k)
```



#### 06640 倒排索引 http://cs101.openjudge.cn/2024sp_routine/06640/

使用dict字典实现查询

```python
def f(x):
    if x in search.keys():
        output=[]
        for k in search[x]:
            output.append(str(k))
        print(' '.join(output))
    else:print('NOT FOUND')
search=dict()
a=int(input())
notes=[]
finds=[]
for i in range(a):
    notes_0=input().split()
    c=notes_0.pop(0)
    for j in notes_0:
        if j in search.keys():
            if search[j.lower()][-1] != i+1:
                search[j.lower()].append(i+1)
        else:
            search[j.lower()]=[i+1]
b=int(input())
for i in range(b):
    finds_0=input()
    f(finds_0)
```



#### 07745:整数奇偶排序 http://cs101.openjudge.cn/2024sp_routine/07745/

倒排：revers=True

```python
nums=list(map(int,input().split()))
a=[]
b=[]
for i in range(10):
    if nums[i]%2==0:
        b.append(nums[i])
    else:
        a.append(nums[i])
a.sort(reverse=True)
b.sort()
c=a+b
s=''
for j in range(10):
    if j == 0:
        s=str(c[j])
    else:s=s+' '+str(c[j])
print(s)
```

#### 12876:生理周期 http://cs101.openjudge.cn/2024sp_routine/12876/

通过整除函数实现

```python
i=True
d=0
while i:
    d+=1
    s=list(map(int,input().split()))
    if s ==[-1,-1,-1,-1]:
        break
    for j in range(s[3]+1,21253+s[3]):
        if (j-s[2])%33==0:
            if (j-s[1])%28==0:
                if (j-s[0])%23==0:
                    print('Case '+str(d)+': the next triple peak occurs in '+str(j-s[3])+' days.')
                    break
```

#### 18182:打怪兽 http://cs101.openjudge.cn/2024sp_routine/18182/

直接使用sort函数实现

```python
s=int(input())
for t in range(s):
    monster=input().split()
    attack=int(monster[0])
    attack_monster=[]
    total=int(monster[1])
    blood=int(monster[2])
    for j in range(attack):
        attack_monster.append(list(map(int,input().split())))
    attack_monster.sort(key=lambda x:(x[0],-x[1]))
    time=attack_monster[-1][0]
    n=0
    l=len(attack_monster)
    for x in range(l):
        if n<total:
            if x>=1 and attack_monster[x][0] != attack_monster[x-1][0]:
                n=1
            elif x==0:
                n=1               
            else:
                n+=1
            blood-= attack_monster[x][1]
        elif n==total:
            if x>=1 and attack_monster[x][0]!= attack_monster[x-1][0]:
                n=1
                blood-= attack_monster[x][1]
        if blood<=0:
            print(attack_monster[x][0])
            break
        elif x == l-1:
            print('alive')
```

#### 19963:买学区房 http://cs101.openjudge.cn/2024sp_routine/19963/

直接使用sort 函数实现

```python
i=int(input())
hou=input().split()
hp=list(map(int,input().split()))
p=[]
xjb=[]
p_x=[]
inp=0
for j in range(i):
    p.append(hp[j])
    h=hou[j].split(',')
    a=int(h[0][1:])
    b=int(h[1][:-1])
    x=(a+b)/hp[j]
    xjb.append(x)
    p_x.append([hp[j],x])
p.sort()
xjb.sort()
pm=1/2*(p[((i+1)//2)-1]+p[i//2])
xjbm=1/2*(xjb[((i+1)//2)-1]+xjb[i//2])
for t in p_x:
    if t[0]<pm and t[1]>xjbm:
        inp+=1
print(inp)
```

#### 20449:是否被5整除 http://cs101.openjudge.cn/2024sp_routine/20449/

直接使用python的运算%实现

```python
n=input()
num=0
out=''
for _ in n:
    num=num*2+int(_)
    if num%5==0:
        out+='1'
    else: out+='0'
print(out)
```

#### 21554:排队做实验 (greedy)v0.2 http://cs101.openjudge.cn/2024sp_routine/21554/

直接使用sort实现

```python
s=int(input())
stu=list(map(int,input().split()))
s_t=[]
for i in range(s):
    s_t.append([stu[i],i+1,0])
s_t.sort(key=lambda x :(x[0],x[1]))
inp=''
tim=0
for t in s_t:
    inp=inp+' '+str(t[1])
    tim+=t[2]
    for j in s_t:
        j[2]+=t[0]
print(inp[1:])
print('%.2f' % (tim/s))
```

#### 21759:P大卷王查询系统 http://cs101.openjudge.cn/2024sp_routine/21759/

使用字典实现

```python
sta=list(map(int,input().split()))
stu_cor=dict()
for i in range(sta[0]):
    stu=input().split()
    if stu[1] in stu_cor.keys():
        stu_cor[stu[1]][0]+=1
        stu_cor[stu[1]][1]+=int(stu[2])
    else:
        stu_cor[stu[1]]=[1,int(stu[2])]
num=int(input())
for j in range(num):
    stu_0=input()
    if stu_cor[stu_0][0]>= sta[1] and stu_cor[stu_0][1]/stu_cor[stu_0][0]>sta[2]:
        print('yes')
    else:
        print('no')
```

#### 22068:合法出栈序列 http://cs101.openjudge.cn/2024sp_routine/22068/

直接使用出栈过程判断

```python
s_0=input()
s=s_0
l=len(s_0)
while True:
    try:
        t = input()
    except EOFError:
        break
    else:
        x=[]
        s=s_0
        f=True
        for i in range(2*l):
            if len(x)>0:
                if len(t)>0 and x[-1] == t[0]:
                    k=x.pop()
                    t=t[1:]
                elif len(t)==0:
                    f=False
                else:
                    if len(s)==0:
                        f=False
                    else:
                        x.append(str(s[0]))
                        s=s[1:]     
            else:
                if len(s)==0:
                        f=False
                else:
                    x.append(str(s[0]))
                    s=s[1:]
        if f:
            print('YES')
        else:
            print('NO')
```

#### 22271:绿水青山之植树造林活动 http://cs101.openjudge.cn/2024sp_routine/22271/

直接使用字典+排序实现

```python
n=int(input())
nums={}
for _ in range(n):
    tree=input()
    if tree in nums:
        nums[tree]+=1
    else:
        nums[tree]=1
nums=sorted(nums.items())
for tree in nums:
    t=(tree[1]/n)*100
    print(tree[0],"{:.4f}".format(t)+'%')
```

#### 22359:Goldbach Conjecture http://cs101.openjudge.cn/2024sp_routine/22359/

逐个枚举，判断是否满足条件

```python
s=int(input())
c=True
for a in range(2,s+1):
    if c:
        b=s-a
        d=True
        for i in range(2,s+1):
            if d:
                if a%i == 0 and a > i:
                    d=False
                elif b%i == 0 and b > i:
                    d=False
                if i == s:
                    c=False
    else:
        print(a-1,s+1-a)
        break
```

#### 23007:版本号排序 http://cs101.openjudge.cn/2024sp_routine/23007/

直接使用排序函数实现

```python
nums=[]
m=int(input())
for k in range(m):
    num=list(map(int,input().split('.')))
    for j in range(len(num),100):
        num.append(-1)
    nums.append(num)
nums.sort()
for k in nums:
    s=''
    while k[0]!= -1:
        s+='.'+str(k[0])
        k=k[1:]
    print(s[1:])
```

#### 23563:多项式时间复杂度 http://cs101.openjudge.cn/2024sp_routine/23563/

使用字符串相关知识实现

```python
s=input()
net=s.split('+')
c=0
for a in net:
    if a[0]=='0':
        net.remove(a)
for a in net:
    b=a.split('^')
    d=int(b[1])
    if d > c:
        c=d
t='n^'+str(c)
print(t)
```

#### 24510:网站网页访问统计 http://cs101.openjudge.cn/2024sp_routine/24510/

def 一个时间函数计算时间，然后使用字典输出

```python
def time(a,b):
    t_a=list(map(int,a.split(':')))
    t_b=list(map(int,b.split(':')))
    ta=0
    tb=0
    for i in range(3):
        if i ==0:
            ta+=t_a[i]*3600
            tb+=t_b[i]*3600
        if i ==1:
            ta+=t_a[i]*60
            tb+=t_b[i]*60
        if i ==2:
            ta+=t_a[i]
            tb+=t_b[i]
    return tb-ta
m=int(input())
web_t=dict()
t_max=0
for j in range(m):
    web,x,y=input().split()
    if web in web_t.keys():
        web_t[web]+=time(x,y)
    else:
        web_t[web]=time(x,y)
    if web_t[web]>=t_max:
        t_max=web_t[web]
for k in web_t.keys():
    if web_t[k]==t_max:
        print(k)
```

#### 24684:直播计票 http://cs101.openjudge.cn/2024sp_routine/24684/

直接使用排序函数实现

```python
net=input().split(' ')
l=len(net)
s=0
p=[]
m=[]
n=len(m)
for j in range(l):
    if net[j] not in m:
        m.append(net[j])
for j in m:
    t=0
    for i in net:
        if i== j:
            t+=1
    if t>s:
        s=t
        p=[int(j)]
    elif t==s:
        p.append(int(j))
p.sort()
q=str(p[0])
k=len(p)
if k >=2:
    for t in range(1,k):
        q=q+' '+str(p[t])
print(q)
```

#### 25301:生日相同 http://cs101.openjudge.cn/2024sp_routine/25301/

使用sort + key=lambda实现

```python
stu=int(input())
stu_bir=dict()
birs=[]
for i in range(stu):
    s=input().split()
    num=s[0]
    bir=(int(s[1]),int(s[2]))
    if bir in stu_bir.keys():
        stu_bir[bir].append(num)
    else:
        stu_bir[bir]=[num]
        birs.append(bir)
birs.sort(key=lambda x:(x[0],x[1]))
for x in birs:
    if len(stu_bir[x])>=2:
        y=str(x[0])+' '+str(x[1])
        for j in stu_bir[x]:
            y=y+' '+str(j)
        print(y)
```

#### 25655:核酸检测统计 http://cs101.openjudge.cn/2024sp_routine/25655/

def一个判断是否通过核酸检测的函数，然后逐个判断

```python
def f(a):
    if '1' not in a:
        return False
    else:
        for j in range(1,8):
            if str(j) not in a:
                if str(j+1) not in a:
                    if str(j+2) not in a:
                        return False
        return True
yuan_stu=dict()
stu_hs=dict()
zongrenshu=0
max_ren=0
output=0
n=int(input())
m=int(input())
for i in range(n):
    stu,yuan=input().split()
    if yuan in yuan_stu.keys():
        yuan_stu[yuan].append(stu)
    else:
        yuan_stu[yuan]=[stu]
for j in range(m):
    hs,stu=input().split()
    if stu in stu_hs.keys():
        stu_hs[stu].append(hs)
    else:
        stu_hs[stu]=[hs]
for yuan in yuan_stu.keys():
    renshu=0
    for stu in yuan_stu[yuan]:
        t=f(stu_hs[stu])
        if not t:
            renshu+=1
            zongrenshu+=1
    ren=renshu/len(yuan_stu[yuan])
    if ren>=max_ren:
        output=yuan
print(zongrenshu)
print(output)
```

#### 25711:推免工作 http://cs101.openjudge.cn/2024sp_routine/25711/

def一个计算gpa的函数，然后排序计算，注意，def函数的时候要让低于60分的人gpa为0

```python
def gpa(score):
    if score>=60:
        return 4-3*(100-score)**2/1600
    else:
        return 0
nums=list(map(int,input().split()))
a=nums[0]
b=nums[1]
stu_sco=dict()
scos=[]
for j in range(a):
    stus=input().split()
    stu=stus.pop(0)
    s=int((len(stus))/2)
    sum=0
    gpas=0
    for k in range(s):
        gpas+=gpa(int(stus[k*2]))*int(stus[k*2+1])
        sum+=int(stus[k*2+1])
    x=gpas/sum
    stu_sco[x]=stu
    scos.append(x)
scos.sort(reverse=True)
output=[]
for c in range(b):
    output.append(stu_sco[scos[c]])
print(' '.join(output))
```

#### 26977:接雨水 http://cs101.openjudge.cn/2024sp_routine/26977/

构造lf列表和lr列表，来确定每个格的水的左边界和右边界，从而计算该处水的高度，来计算总水量

```python
n=int(input())
ls=list(map(int,input().split()))
m1=0
m2=0
lf=[0]
lr=[0]
for k in range(n-1):
    if ls[k]>m1:
        m1=ls[k]
    lf.append(m1)
for k in range(n-1):
    j=n-1-k
    if ls[j]>m2:
        m2=ls[j]
    lr.insert(0,m2)
sum=0
for k in range(n):
    if min(lf[k],lr[k])>ls[k]:
        sum+=min(lf[k],lr[k])-ls[k]
print(sum)
```

#### 26978:滑动窗口最大值 http://cs101.openjudge.cn/2024sp_routine/26978/

另外构造一个列表帮助实现

```python
n,k=map(int,input().split())
stack=[]
output=[]
nums=list(map(int,input().split()))
for j in range(len(nums)):
    stack.append(nums[j])
    for i in range(min(j,k)):
        t=min(j,k)-1
        if stack[t-i]<= nums[j]:
            stack[t-i]=nums[j]
        else:
            break
    if j >=k-1:
        output.append(str(stack.pop(0)))
print(' '.join(output))
```

#### 27273:简单的数学题 http://cs101.openjudge.cn/2024sp_routine/27273/

math库中log函数应用，在计算二叉树的层数时可以使用

```python
from math import log
time=int(input())
for i in range(time):
    n=int(input())
    m=int(log(n,2))+1
    s=int((1/2)*n*(n+1)-2*((2**m)-1))
    print(s)
```

#### 27274:字符串提炼 http://cs101.openjudge.cn/2024sp_routine/27274/

math库中log函数应用

```python
from math import log
t=input()
n=len(t)
m=int(log(n,2))
s=[]
s_1=''
for i in range(m+1):
    a=t[(2**i)-1]
    s.append(a)
while len(s)>0:
    s_1=s_1+s[0]
    del s[0]
    if len(s)>0:
        s_1=s_1+s[-1]
        del s[-1]
print(s_1)
```

#### 27300:模型整理 http://cs101.openjudge.cn/2024sp_routine/27300/

使用字典的列表存储+排序实现

```python
t=int(input())
d=dict()
for i in range(t):
    ai=input().split('-')
    if ai[0] in d.keys():
        if ai[1][-1]=='B':
            d[ai[0]][1].append(float(ai[1][:-1]))
        else:
            d[ai[0]][0].append(float(ai[1][:-1]))
    else:
        if ai[1][-1]=='B':
            d[ai[0]]=[[],[float(ai[1][:-1])]]
        else:
            d[ai[0]]=[[float(ai[1][:-1])],[]]
st=sorted(d.keys())
for j in st:
    s=''
    d[j][0].sort()
    d[j][1].sort()
    for i in d[j][0]:
        if i==int(i):
            s=s+str(int(i))+'M, '
        else:
            s=s+str(i)+'M, '
    for i in d[j][1]:
        if i==int(i):
            s=s+str(int(i))+'B, '
        else:
            s=s+str(i)+'B, '
    print(j+': '+s[:-2])
```

#### 27301:给植物浇水 http://cs101.openjudge.cn/2024sp_routine/27301/

直接模拟给植物浇水的过程

```python
ab= list(map(int,input().split()))
pla=list(map(int,input().split()))
n=ab[0]//2
m=n+(ab[0]%2)
a=ab[1]
b=ab[2]
if m==n:
    s_1=a
    s_2=b
    t=0
    for i in range(n):
        if s_1< pla[i]:
            s_1=a
            t+=1
        if s_2< pla[ab[0]-1-i]:
            s_2=b
            t+=1
        s_1-=pla[i]
        s_2-=pla[ab[0]-1-i]  
    print(t)
else:
    s_1=a
    s_2=b
    t=0
    for i in range(n):
        if s_1< pla[i]:
            s_1=a
            t+=1
        if s_2< pla[ab[0]-1-i]:
            s_2=b
            t+=1
        s_1-=pla[i]
        s_2-=pla[ab[0]-1-i]
    t_1=(s_1)%a
    t_2=(s_2)%b
    if s_1>=s_2:
        if s_1< pla[n]:
            s_1=a
            t+=1
    elif s_1<s_2:
        if s_2< pla[ab[0]-1-n]:
            s_2=b
            t+=1
    print(t)
```

#### 27310:积木 http://cs101.openjudge.cn/2024sp_routine/27310/

使用hall定理验证：在一个二分图中，存在一一匹配等价于每n个点的邻居的总个数不少于n

（数学竞赛经典图论定理）

```python
i=int(input())
wod=dict()
for j in range(4):
    s=input()
    x=[]
    for t in range(6):
        x.append(s[t])
    wod[j]=x
for k in range(i):
    f=True
    voc=input()
    l=len(voc)
    voc_d=dict()
    for m in range(l):
        voc_d[m]=[]
        for j in range(4):
            if voc[m] in wod[j]:
                voc_d[m].append(j)
    for j in range(2**l):
        n=0
        t=[]
        for m in range(l):
            if j%(2**m) != j%(2**(m+1)):
                n+=1
                for x in voc_d[m]:
                    if x not in t:
                        t.append(x)
        if len(t)< n:
            f=False
    if f:
        print('YES')
    else:
        print('NO')
```

#### 27625:AVL树至少有几个结点 http://cs101.openjudge.cn/2024sp_routine/27625/

需要使用前两行代码来保存之前函数中计算过的数据，不然会超时

```python
from functools import lru_cache
@lru_cache(maxsize=None)
def avl_min_nodes(n):
 if n == 0:
     return 0
 elif n == 1:
     return 1
 else:
     return avl_min_nodes(n-1) + avl_min_nodes(n-2) + 1

n = int(input())
min_nodes = avl_min_nodes(n)
print(min_nodes)
```

## 2 动态规划

使用递归的思想，把n情况的问题变为更小的n-1，n-2或n-k情况下的问题，通常使用函数的算法实现，也需要其他python内置函数辅助实现

第一次月考动态规划仅仅ac1，发现自己在计概时因为课程注重大作业，过于疏于编程的练习，后面加强了练习，熟练度上升了很多，也更加理解了计算机算法中动态规划的思想

#### 02773 采药问题 http://cs101.openjudge.cn/2024sp_routine/02773/

正常直接计算会time limit exceeded，使用动态规划可以实现问题

```python
s=input().split()
t=int(s[0])
i=int(s[1])
her=[]
d=dict()
d[(0,0)]=0
for j in range(i):
    her.append(list(map(int,input().split())))
for j in range(i):
    for k in range(t+1):
        if k ==0:
            d[(k,j)]=0
        elif k < her[j][0]:
            if j>0:
                d[(k,j)]=d[(k,j-1)]
            else:d[(k,j)]=0
        else:
            if j>0:
                if her[j][1]+d[((k-her[j][0]),j-1)]>d[(k,(j-1))]:
                    d[(k,j)]=her[j][1]+d[((k-her[j][0]),j-1)]
                else:d[(k,j)]=d[(k,(j-1))]
            else:d[(k,j)]=her[j][1]
print(d[(t,i-1)])
```

#### 02945 拦截导弹 http://cs101.openjudge.cn/2024sp_routine/02945/

月考题目，定义函数实现动态规划，不使用动态规划会time limit exceeded

```python
k=int(input())
l=list(map(int,input().split()))
def f(lst):
    if len(lst)==0:
        return 0
    elif len(lst) == 1:
        return 1
    else:
        l1=[]
        for i in range(1,len(lst)):
            if lst[i]<=lst[0]:
                l1.append(lst[i])
        return max(f(lst[1:]),f(l1)+1)
print(f(l))           
```

#### 03151 Pots http://cs101.openjudge.cn/2024sp_routine/

先通过壶的最大公约数，最小公倍数判断，后面使用函数进行动态规划，得到倒水步骤

代码又臭又长，属实是缺乏编程素养

```python
a,b,c=map(int,input().split())
def gcd(x,y):
    if x==0:
        return y
    elif y==0:
        return x
    else:
        if x>=y:
            return gcd(x-y,y)
        else:
            return gcd(x,y-x)
net0=[]
net=[]
for k in range(b+1):
    net0.append([-1])
for j in range(a+1):
    net1=net0.copy()
    net.append(net1)
net[0][0]=[0]
net[0][b]=[1,"FILL(2)"]
net[a][0]=[1,"FILL(1)"]
stack1=[[0,0]]
stack2=[[a,0],[0,b]]
role=1
s=2
while s:
    s=0
    role+=1
    stack1=stack2
    stack2=[]
    for _ in stack1:
        if net[a][_[1]]==[-1]:
            net[a][_[1]]=[role]
            net[a][_[1]].extend(net[_[0]][_[1]][1:])
            net[a][_[1]].append('FILL(1)')
            stack2.append([a,_[1]])
            s+=1
        if net[_[0]][b]==[-1]:
            net[_[0]][b]=[role]
            net[_[0]][b].extend(net[_[0]][_[1]][1:])
            net[_[0]][b].append('FILL(2)')
            stack2.append([_[0],b])
            s+=1
        if net[0][_[1]]==[-1]:
            net[0][_[1]]=[role]
            net[0][_[1]].extend(net[_[0]][_[1]][1:])
            net[0][_[1]].append('DROP(1)')
            stack2.append([0,_[1]])
            s+=1
        if net[_[0]][0]==[-1]:
            net[_[0]][0]=[role]
            net[_[0]][0].extend(net[_[0]][_[1]][1:])
            net[_[0]][0].append('DROP(2)')
            stack2.append([_[0],0])
            s+=1
        if _[0]+_[1]<=a:
            if net[_[0]+_[1]][0]==[-1]:
                net[_[0]+_[1]][0]=[role]
                net[_[0]+_[1]][0].extend(net[_[0]][_[1]][1:])
                net[_[0]+_[1]][0].append('POUR(2,1)')
                stack2.append([_[0]+_[1],0])
                s+=1
        if _[0]+_[1]<=b:
            if net[0][_[0]+_[1]]==[-1]:
                net[0][_[0]+_[1]]=[role]
                net[0][_[0]+_[1]].extend(net[_[0]][_[1]][1:])
                net[0][_[0]+_[1]].append('POUR(1,2)')
                stack2.append([0,_[0]+_[1]])
                s+=1
        if _[0]+_[1]>a:
            if net[a][_[0]+_[1]-a]==[-1]:
                net[a][_[0]+_[1]-a]=[role]
                net[a][_[0]+_[1]-a].extend(net[_[0]][_[1]][1:])
                net[a][_[0]+_[1]-a].append('POUR(2,1)')
                stack2.append([a,_[0]+_[1]-a])
                s+=1
        if _[0]+_[1]>b:
            if net[_[0]+_[1]-b][b]==[-1]:
                net[_[0]+_[1]-b][b]=[role]
                net[_[0]+_[1]-b][b].extend(net[_[0]][_[1]][1:])
                net[_[0]+_[1]-b][b].append('POUR(1,2)')
                stack2.append([_[0]+_[1]-b,b])
                s+=1
   
d=gcd(a,b)
if c%d==0:
    out1=[-1]
    out2=[-1]
    out3=[-1]
    out4=[-1]
    s=10000000
    t=[]
    if c<=a:
        out1=net[c][0]
        out2=net[c][b]
    if c<=b:
        out3=net[0][c]
        out4=net[a][c]
    if out1[0]>=0:
        s=out1[0]
        t=out1[1:]
        if out2[0]<s and 0<=out2[0]:
            s=out2[0]
            t=out2[1:]
    if out3[0] >=0:
        if out3[0]<s and 0<=out3[0]:
            s=out3[0]
            t=out3[1:]
        if out4[0]<s and 0<=out4[0]:
            s=out4[0]
            t=out4[1:]
    print(s)
    for k in t:
        print(k)
else:print('impossible')
```

#### 04001 抓住那头牛 http://cs101.openjudge.cn/2024sp_routine/04001/

构造visited集合来储存是否被访问到的位置，构造find函数实现动态规划

```python
visited=[False]*100001
time=[0]*100001
def find(x,y):
    k=0
    role=[x]
    visited[x]=True
    while not visited[y]:
        k+=1
        rolenext=[]
        for s in role:
            if s>0 and (not visited[s-1]):
                visited[s-1]=True
                rolenext.append(s-1)
            if s<100000 and (not visited[s+1]):
                visited[s+1]=True
                rolenext.append(s+1)
            if s<=50000 and (not visited[2*s]):
                visited[2*s]=True
                rolenext.append(2*s)
        role=rolenext
    print(k)
a,b=map(int,input().split())
find(a,b)
```

#### 04077 出栈序列统计 http://cs101.openjudge.cn/2024sp_routine/04077/

使用def函数递归得出结果

```python
def f(x):
    if x==0:
        return 1
    else:
        a=0
        for j in range(x):
            a+=f(j)*f(x-j-1)
        return a
i=int(input())
print(f(i))
```

#### 04117 简单的整数划分问题 http://cs101.openjudge.cn/2024sp_routine/04117/

构造函数f(a,b)来表示划分a，最小数为b的方法数，再求和得到答案

```python
def f(a,b):
    if b==1:return 1
    if a==1:
        if b==1:
            return 1
        else:
            return 0
    else :
        s=f(a-1,b-1)
        if 2*b <=a:
            s+=f(a-b,b)
        return s
def g(x):
    k=0
    for y in range(1,x+1):
        k+=f(x,y)
    return k
while True:
    try:
        n=int(input())
    except EOFError:
        break
    else:
        print(g(n))
```

#### 04147 汉诺塔问题 http://cs101.openjudge.cn/2024sp_routine/04147/

月考题目，经典的计概动态规划题目

```python
pan=input().split()
i=int(pan[0])
move=[(1,1,0)]
pan_1=[pan[3],pan[1],pan[2]]
for k in range(2,i+1):
    s=2**(k-1)-1
    for l in range(s):
        (x,y,z)=(move[0][0],(2-move[0][1])%3,(2-move[0][2])%3)
        move.append((x,y,z))
        move.pop(0)
    move.append((k,1,0))
    for l in range(len(move)-1):
        (x,y,z)=(move[l][0],(move[l][1]+1)%3,(move[l][2]+1)%3)
        move.append((x,y,z))
for l in range(len(move)):
    print(str(move[l][0])+':'+pan_1[move[l][1]]+'->'+pan_1[move[l][2]])
```

#### 08758:2的幂次方表示 http://cs101.openjudge.cn/2024sp_routine/08758/

通过动态规划实现幂次方表示

```python
def f(x):
    if not x:
        return '0'
    elif x==2:
        return '2'
    else:
        t=0
        while 2**(t+1)<=x:
            t+=1
        if t==1:
            if x-2**t!=0:
                return '2+'+f(x-2**t)
            else:
                return '2'
        else:
            if x-2**t!=0:
                return '2('+f(t)+')+'+f(x-2**t)
            else:
                return '2('+f(t)+')'
a=int(input())
print(f(a))
```

#### 20742:泰波拿契數 http://cs101.openjudge.cn/2024sp_routine/20742/

经典计概动态规划问题

```python
def f(x):
    if x==0:
        return 0
    if x==1:
        return 1
    if x==2:
        return 1
    else:
        return f(x-1)+f(x-2)+f(x-3)
n=int(input())
print(f(n))
```

#### 21964:01背包 http://cs101.openjudge.cn/2024sp_routine/21964/

建立l列表储存最大喜爱值，用动态规划的思想实现

```python
n,m=map(int,input().split())
l=[0 for i in range(m+1)]
for i in range(n):
    a,b=map(int,input().split())
    for j in range(m,a-1,-1):
        l[j]=max(l[j],l[j-a]+b)
print(l[-1])
```

#### 22642:括号生成 http://cs101.openjudge.cn/2024sp_routine/22642/

使用动态规划实现的，当时不会栈

```python
def kh(x):
    if not x:
        return []
    elif x==1:
        return [[2,1,0,2]]
    else:
        net=[]
        for j in kh(x-1):
            j.insert(0,1)
            j.insert(0,2)
            l=0
            for k in range(1,len(j)):
                if j[k-l]==2:
                    j.insert(k-l,0)
                    net.append(j.copy())
                    j.pop(k-l)
                    j.pop(k-l)
                    l+=1
        return net
a=int(input())
nt=[')','(','']
nums=[]
nn=dict()
for j in kh(a):
    t=''
    s=''
    for k in range(1,len(j)):
        if j[k]!=2:
            s=s+nt[j[k]]
            t=t+str(j[k])
    nums.append(int(t))
    nn[int(t)]=s
nums.sort(reverse=True)
for j in nums:
    print(nn[j])
```

#### 23997:奇数拆分 http://cs101.openjudge.cn/2024sp_routine/23997/

使用extend函数合并列表，以及math函数的sqrt来辅助实现动态规划

```python
import math
n=int(input())
def f(a,b):
    if a:
        net=[]
        for _ in range(2*int(math.sqrt(a))-1,b+1,2):
            net_0=[]
            if _<=a:
                for j in f(a-_,_-1):
                    j.append(_)
                    net_0.append(j)
                net.extend(net_0)
        return net
    elif b==0:
        if not a:
            return [[]]
        else:
            return []
    else:
        return[[]]
x=sorted(f(n,n))
for k in x:
    s=''
    for t in k:
        s=s+' '+str(t)
    print(s[1:])
print(len(x))
```

#### 26573:康托集的图像表示 http://cs101.openjudge.cn/2024sp_routine/26573/

简单动态规划实现

```python
def ct(x):
    if x==1:
        return '*-*'
    else:
        s=''
        t=ct(x-1)
        for j in range(len(t)):
            s+=t[j]*3
        for j in range(1,len(s)-1):
            if s[j-1]=='*' and s[j+1]=='*':
                s=s[:j]+'-'+s[j+1:]
        return s
i=int(input())
print(ct(i))
```

## 二 时间复杂度

### 1 @lru_cache

使用@lru_cache来储存递归函数中计算过的值，来达到时间复杂度的要求

#### 02192 Zipper http://cs101.openjudge.cn/2024sp_routine/02192/

判断第三个字符串是不是前两个的混合，代码平凡

```python
from functools import lru_cache
@lru_cache
def f(a,b,c):
    if len(c)==0:
        return True
    else:
        if len(a) and c[0]==a[0] and f(a[1:],b,c[1:]):
            return True
        elif len(b) and c[0]==b[0] and f(a,b[1:],c[1:]):
            return True
        else:return False

n=int(input())
for _ in range(n):
    a,b,c=input().split()
    x=len(c)
    if f(a,b,c):
        print('Data set %d: yes' % (_+1))
    else:print('Data set %d: no' % (_+1))
```

#### 02499 Binary Tree http://cs101.openjudge.cn/2024sp_routine/02499/

通过@lru_cache保存辗转相减的结果，也是一种动态规划的思想

```python
from functools import lru_cache
@lru_cache
def f(x,y):
    if x==1 or y==1:
        return [x-1,y-1]
    elif x>y:
        return [f(x%y,y)[0]+(x//y),f(x%y,y)[1]]
    else:return [f(x,y%x)[0],f(x,y%x)[1]+(y//x)]
n=int(input())
for j in range(n):
    s,t=map(int,input().split())
    print('Scenario #%d:' % (j+1))
    print(str(f(s,t)[0])+' '+str(f(s,t)[1]))
    print(' ')
```

### 2 二分查找法

一种经典的算法，在连续的数，从一个数开始满足条件/不满足条件的情况下，可以使用二分查找发更快的锁定分界数

编写过程中一定要注意大指针和小指针的递归结束的情况，一定不要无限循环

#### 02774 木材加工 http://cs101.openjudge.cn/2024sp_routine/02774/

```python
n, k = map(int, input().split())
expenditure = []
for _ in range(n):
    expenditure.append(int(input()))


def check(x):
    num = 0
    for i in range(n):
        num += expenditure[i] // x

    return num >= k

lo = 1
hi = max(expenditure) + 1

if sum(expenditure) < k:
    print(0)
    exit()

ans = 1
while lo < hi:
    mid = (lo + hi) // 2
    if check(mid):
        ans = mid
        lo = mid + 1
    else:
        hi = mid

print(ans)
```

#### 04135 月度开销 http://cs101.openjudge.cn/2024sp_routine/04135/

```python
n,m=map(int,input().split())
moneys=[]
sum=0
for _ in range(n):
    q=int(input())
    moneys.append(q)
    sum+=q
def check(x):
    days=1
    money=0
    for d in range(n):
        if moneys[d]>x:
            return False
        if money+moneys[d]>x:
            money=moneys[d]
            days+=1
        else:money+=moneys[d]
    return days<=m
lo=1
hi=sum
ans=1
s=0
while lo<hi:
    s+=1
    mid=(lo+hi+1)//2
    if check(mid):
        ans=mid
        hi=mid-1
    else:
        lo=mid
print(ans)
```

#### 08210:河中跳房子 http://cs101.openjudge.cn/2024sp_routine/08210/

```python
def f(x):
    num = 0
    now = 0
    for i in range(1, n+2):
        if distances[i] - now < x:
            num += 1
        else:
            now = distances[i]
            
    if num > m:
        return False
    else:
        return True
l,n,m=map(int,input().split())
distances=[0]
for _ in range(n):
    distances.append(int(input()))
distances.append(l)
big=l
small=1
while big - small !=1:
    mid=(big+small)//2
    if f(mid):
        small=mid
    else:
        big=mid
print(small)
```

### 3 MergeSort

排序算法之一，可以以更低的时间复杂度实现排序

mergesort算法的过程有时会便于我们解决一些特殊问题，从而我们可以使用mergesort的具体代码来解决一些相关问题

#### 02299 Ultra-QuickSort http://cs101.openjudge.cn/2024sp_routine/02299/

```python
def mergeSort(arr):
	if len(arr) > 1:
		mid = len(arr)//2

		L = arr[:mid]
		R = arr[mid:]

		a=mergeSort(L)
		b=mergeSort(R)

		i =j = k = s = 0
		while i < len(L) and j < len(R):
			if L[i] <= R[j]:
				arr[k] = L[i]
				i += 1
			else:
				arr[k] = R[j]
				s+=j-k+len(L)
				j += 1
			k += 1
		while i < len(L):
			arr[k] = L[i]
			i += 1
			k += 1

		while j < len(R):
			arr[k] = R[j]
			j += 1
			k += 1
		return a+b+s
	else:return 0
while True:
	i=int(input())
	if i == 0:
		break
	else:
		net=[]
		for j in range(i):
			k=int(input())
			net.append(k)
		x=mergeSort(net)
		print(x)
```

#### 09201:Freda的越野跑http://cs101.openjudge.cn/2024sp_routine/09201/

使用mergesort排序的过程，正好可以适配超越次数的计算方式

```python
def mergeSort(arr):
	if len(arr) > 1:
		mid = len(arr)//2

		L = arr[:mid]
		R = arr[mid:]

		a=mergeSort(L)
		b=mergeSort(R)

		i =j = k = s = 0
		while i < len(L) and j < len(R):
			if L[i] <= R[j]:
				arr[k] = L[i]
				i += 1
			else:
				arr[k] = R[j]
				s+=j-k+len(L)
				j += 1
			k += 1
		while i < len(L):
			arr[k] = L[i]
			i += 1
			k += 1

		while j < len(R):
			arr[k] = R[j]
			j += 1
			k += 1
		return a+b+s
	else:return 0
i=int(input())
net=list(map(int,input().split()))
net_0=[]
for j in range(i):
	net_0.append(net[i-1-j])
print(mergeSort(net_0))
```

#### 20018:蚂蚁王国的越野跑 http://cs101.openjudge.cn/2024sp_routine/20018/

使用mergesort排序的过程，正好可以适配超越次数的计算方式

再用longlong类型输出

```python
import ctypes
def mergeSort(arr):
	if len(arr) > 1:
		mid = len(arr)//2

		L = arr[:mid]
		R = arr[mid:]

		a=mergeSort(L)
		b=mergeSort(R)

		i =j = k = s = 0
		while i < len(L) and j < len(R):
			if L[i] <= R[j]:
				arr[k] = L[i]
				i += 1
			else:
				arr[k] = R[j]
				s+=j-k+len(L)
				j += 1
			k += 1
		while i < len(L):
			arr[k] = L[i]
			i += 1
			k += 1

		while j < len(R):
			arr[k] = R[j]
			j += 1
			k += 1
		return a+b+s
	else:return 0
net=[]
i=int(input())
for j in range(i):
    a=int(input())
    net.insert(0,a)
print(ctypes.c_longlong(mergeSort(net)).value)
```

### 4 T-prime问题

#### 18176:2050年成绩计算 http://cs101.openjudge.cn/2024sp_routine/18176/

需要讲所有t-prime的是否情况打成一个True-False列表，以加速t-prime的判断

True-False列表在后续也有许多的应用

```python
import math
s=list(map(int,input().split()))
p=[True]*10001
for x in range(2,101):
    d=x**2
    if p[x]:
        while d<10001:
            p[d]=False
            d+=x
for i in range(s[0]):
    stu=list(map(int,input().split()))
    n=0
    for j in range(len(stu)):
        t=False
        b=math.sqrt(stu[j])
        if b==int(b):
            b=int(b)
            t=p[b]
        if t:
            n+=stu[j]
    if n ==0:
        print(0)
    else:
        a=f'{(n/len(stu)):.2f}'
        print(a)
```

### 5 堆实现

#### 02442 Sequence http://cs101.openjudge.cn/2024sp_routine/02442/

使用堆实现

```python
import heapq

t = int(input())
for _ in range(t):
    m, n = map(int, input().split())
    seq = list(map(int, input().split()))
    seq.sort()
    for s in range(m - 1):
        seq1 = list(map(int, input().split()))
        seq1.sort()
        ans = [(seq[i] + seq1[0], i, 0) for i in range(n)]
        heapq.heapify(ans)
        temp = []
        for t in range(n):
            now, i, j = heapq.heappop(ans)
            temp.append(now)
            if j + 1 < len(seq1):
                heapq.heappush(ans, (seq[i] + seq1[j + 1], i, j + 1))
        seq = [temp[k] for k in range(n)]
    print(*seq)
```

#### 

## 三 线性数据结构

### 1 栈实现

#### 1.回溯

一种高级的枚举方式，先判断枚举的中间过程是否可行，如果不可行，则往前一步回溯，继续枚举

##### 01321 棋盘问题 http://cs101.openjudge.cn/2024sp_routine/01321/

在棋盘里放棋的可能情况数

定义判断”是否可以继续放“函数与回溯函数，从而实现枚举过程中的回溯

在实现过程中需要栈数据结构辅助实现

```python
def solve_n_queens(m,n,x):
    solutions = []
    queens = [-1] * m
    
    def backtrack(row,t):
        if t == 0:
            solutions.append(0)
        else:
            if row<=m-1:
                for col in x[row]:
                    if is_valid(row, col):  
                        queens[row] = col
                        backtrack(row + 1,t-1)
                        queens[row] = -1
                if row <=m-2:  
                    backtrack(row+1,t)

    def is_valid(row, col):
        for r in range(row):
            if queens[r]==col:
                return False
        return True
    
    backtrack(0,n)

    return len(solutions)
while True:
    a,b=map(int,input().split())
    if (a , b)==(-1,-1):
        break
    else:
        net=[]
        for j in range(a):
            sp=[]
            s=input()
            for k in range(a):
                if s[k]=='#':
                    sp.append(k)
            net.append(sp)
        print(solve_n_queens(a,b,net)
```

##### 02754 八皇后问题 http://cs101.openjudge.cn/2024sp_routine/solution/44929849/

与上一道题类似

```python
def solve_n_queens(n):
    solutions = []
    queens = [-1] * n

    def backtrack(row):
        if row == n:
            solutions.append(queens.copy())
        else:
            for col in range(n):
                if is_valid(row, col):
                    queens[row] = col
                    backtrack(row + 1)
                    queens[row] = -1

    def is_valid(row, col):
        for r in range(row):
            if queens[r] == col or abs(row - r) == abs(col - queens[r]):
                return False
        return True

    backtrack(0)

    return solutions


def get_queen_string(b):
    solutions = solve_n_queens(8)
    if b > len(solutions):
        return None
    queen_string = ''.join(str(col + 1) for col in solutions[b - 1])
    return queen_string


test_cases = int(input()) 
for _ in range(test_cases):
    b = int(input())
    queen_string = get_queen_string(b)
    print(queen_string)
```

#### 2.经典栈实现

##### 02694 波兰表达式 http://cs101.openjudge.cn/2024sp_routine/02694/

前序表达式求值，经典的使用栈实现的案例

```python
v=input().split()
f=['+','-','*','/']
while len(v)>1:
    for i in range(len(v)):
        if i<len(v) and v[i]=='+':
            if v[i+1] not in f and v[i+2] not in f:
                a=float(v[i+1])+float(v[i+2])
                v[i]=a
                v.pop(i+1)
                v.pop(i+1)
    for i in range(len(v)):
        if i<len(v) and v[i]=='-':
            if v[i+1] not in f and v[i+2] not in f:
                a=float(v[i+1])-float(v[i+2])
                v[i]=a
                v.pop(i+1)
                v.pop(i+1)
    for i in range(len(v)):
        if i<len(v) and v[i]=='*':
            if v[i+1] not in f and v[i+2] not in f:
                a=float(v[i+1])*float(v[i+2])
                v[i]=a
                v.pop(i+1)
                v.pop(i+1)
    for i in range(len(v)):
        if i<len(v) and v[i]=='/':
            if v[i+1] not in f and v[i+2] not in f:
                a=float(v[i+1])/float(v[i+2])
                v[i]=a
                v.pop(i+1)
                v.pop(i+1)
print(f'{v[0]:.6f}')
```

##### 03704 括号匹配问题 http://cs101.openjudge.cn/2024sp_routine/03704/

左括号入栈，右括号使左括号出栈，是括号匹配问题的经典思想

```python
while True:
    try:
        t = input()
    except EOFError:
        break
    else:
        a=[]
        b=[]
        a_0=[]
        b_0=[]
        for i in range(len(t)):
            if t[i]=='(':
                a.append(i)
            elif t[i]==')':
                b.append(i)
        while len(a)>0:
            if len(b)>0:
                f=True
                for j in range(len(b)):
                    if a[-1]<b[j]:
                        if j==0 or a[-1]>b[j-1]:
                            a.pop()
                            b.pop(j)
                            f=False
                            break
                if f:
                    a_0.append(a[-1])
                    a.pop()
            else:
                a_0.append(a[-1])
                a.pop()
        if len(b)>0:
            for i in range(len(b)):
                b_0.append(b[i])
        s=''
        for i in range(len(t)):
            if i in a_0:
                s=s+'$'
            elif i in b_0:
                s=s+'?'
            else:
                s=s+' '
        print(t)
        print(s)
```

##### 04099 队列和栈 http://cs101.openjudge.cn/2024sp_routine/04099/

熟悉栈的入栈和出栈过程

```python
t=int(input())
for i in range(t):
    s=int(input())
    m=''
    n=''
    x=0
    a=''
    b=''
    for j in range(s):
        p=input().split()
        if p[0]== 'push':
            a=a+p[1]
            b=b+p[1]
        else:
            if len(a)==0:
                x=1
            else:
                a=a[1:]
                b=b[:-1]
    if x==1:
        print('error')
        print('error')
    else:
        for d in range(len(a)) :
            m=m+' '+a[d]
        for e in range(len(b)):
            n=n+' '+b[e]
        m=m[1:]
        n=n[1:]
        print(m)
        print(n)
```

##### 06263 布尔表达式 http://cs101.openjudge.cn/2024sp_routine/06263/

一直runtime error，使用内置eval函数过掉了，不知道问题出在哪里

```python
while True:
    try:
        string=input()
        finalstring=""
        for v in string:
            if v:
                if v=="V":
                    finalstring+="True"
                elif v=="F":
                    finalstring+="False"
                elif v=="!":
                    finalstring+=" not "
                elif v=="&":
                    finalstring+=" and "
                elif v=="|":
                    finalstring+=" or "
                else:
                    finalstring+=v
        if eval(finalstring):
            print("V")
        else:
            print("F")
    except EOFError:
        break
```

##### 20140:今日化学论文 http://cs101.openjudge.cn/2024sp_routine/20140/

使用栈来处理论文，和经典的括号匹配问题类似，只不过在括号之间添加了内容，只需在括号匹配时把括号内内容处理

```python
pap=input()
stack=[]
nums=['0','1','2','3','4','5','6','7','8','9']
for s in pap:
    if s==']':
        t=''
        while True:
            t0=stack.pop()
            if t0=='[':
                break
            else:
                t=t0+t
        num=''
        for l in t:
            if l in nums:
                num=num+l
                t=t[1:]
            else:
                break
        if num:
            d=t*int(num)
        else:d=t
        stack.append(d)
    else:
        stack.append(s)
print(''.join(stack))
```

##### 20743:整人的提词本 http://cs101.openjudge.cn/2024sp_routine/20743/

同上一题类似

```python
s=input()
stack=[]
out=[]
for k in s:
    if k ==')':
        while True:
            a=stack.pop()
            if a =='(':
                stack.extend(out)
                out=[]
                break
            else:out.append(a)
    else:
        stack.append(k)
print(''.join(stack))
```

##### 22067:快速堆猪 http://cs101.openjudge.cn/2024sp_routine/22067/

构造最轻猪栈，用两个栈来解决栈数据结构中的最值问题

```python
stack=[]
minstack=[]
while True:
    try:
        s=input()
    except EOFError:
        break
    else:
        if s[:2]=='po':
            if len(stack)!=0:
                stack.pop()
                minstack.pop()
        elif s[:2]=='pu':
            stack.append(int(s[5:]))
            if len(minstack)>0:
                if minstack[-1]>int(s[5:]):
                    minstack.append(int(s[5:]))
                else:
                    minstack.append(minstack[-1])
            else:minstack.append(int(s[5:]))
        elif s=='':
            break
        else:
            if len(stack)!=0:
                print(minstack[-1])
```

##### 24588:后序表达式求值 http://cs101.openjudge.cn/2024sp_routine/solution/44076307/

使用栈结构实现，将后续表达式中的内容依次放入栈中，进行运算

```python
m=int(input())
for j in range(m):
    v=input().split()
    f=['+','-','*','/']
    while len(v)>1:
        for i in range(len(v)):
            if i>1 and i<len(v) and v[i]=='*':
                if v[i-1] not in f and v[i-2] not in f:
                    a=float(v[i-1])*float(v[i-2])
                    v[i]=a
                    v.pop(i-2)
                    v.pop(i-2)
        for i in range(len(v)):
            if i>1 and i<len(v) and v[i]=='/':
                if v[i-1] not in f and v[i-2] not in f:
                    a=float(v[i-2])/float(v[i-1])
                    v[i]=a
                    v.pop(i-2)
                    v.pop(i-2)
        for i in range(len(v)):
            if i>1 and i<len(v) and v[i]=='+':
                if v[i-1] not in f and v[i-2] not in f:
                    a=float(v[i-1])+float(v[i-2])
                    v[i]=a
                    v.pop(i-2)
                    v.pop(i-2)
        for i in range(len(v)):
            if i>1 and i<len(v) and v[i]=='-':
                if v[i-1] not in f and v[i-2] not in f:
                    a=float(v[i-2])-float(v[i-1])
                    v[i]=a
                    v.pop(i-2)
                    v.pop(i-2)
    print(f'{float(v[0]):.2f}')
```

##### 24591:中序表达式转后序表达式 http://cs101.openjudge.cn/2024sp_routine/solution/44077911/

构造括号栈和算式栈，一个十分经典而困难的问题

```python
def infix_to_postfix(expression):
    precedence = {'+':1, '-':1, '*':2, '/':2}
    stack = []
    postfix = []
    number = ''

    for char in expression:
        if char.isnumeric() or char == '.':
            number += char
        else:
            if number:
                num = float(number)
                postfix.append(int(num) if num.is_integer() else num)
                number = ''
            if char in '+-*/':
                while stack and stack[-1] in '+-*/' and precedence[char] <= precedence[stack[-1]]:
                    postfix.append(stack.pop())
                stack.append(char)
            elif char == '(':
                stack.append(char)
            elif char == ')':
                while stack and stack[-1] != '(':
                    postfix.append(stack.pop())
                stack.pop()

    if number:
        num = float(number)
        postfix.append(int(num) if num.is_integer() else num)

    while stack:
        postfix.append(stack.pop())

    return ' '.join(str(x) for x in postfix)

n = int(input())
for _ in range(n):
    expression = input()
    print(infix_to_postfix(expression))
```

### 2 双端队列

python中有双端队列内置函数deque

#### 05902 双端队列 http://cs101.openjudge.cn/2024sp_routine/05902/

熟悉双端队列的进队出队方式

```python
i=int(input())
for j in range(i):
    t=int(input())
    l=[]
    for s in range(t):
        d=list(map(int,input().split()))
        if d[0]==1:
            l.append(d[1])
        else:
            if d[1]==0:
                l.pop(0)
            else:l.pop(-1)
    if len(l)==0:
        print('NULL')
    else:
        s=''
        for k in l:
            s=s+' '+str(k)
        print(s[1:])
```

#### 27925:小组队列 http://cs101.openjudge.cn/2024sp_routine/27925/

熟悉deque函数的代码

```python
from collections import deque
t = int(input())
groups = {}
member_to_group = {}
for _ in range(t):
    members = list(map(int, input().split()))
    group_id = members[0]
    groups[group_id] = deque()
    for member in members:
        member_to_group[member] = group_id

queue = deque()
queue_set = set()


while True:
    command = input().split()
    if command[0] == 'STOP':
        break
    elif command[0] == 'ENQUEUE':
        x = int(command[1])
        group = member_to_group.get(x, None)
        if group is None:
            group = x
            groups[group] = deque([x])
            member_to_group[x] = group
        else:
            groups[group].append(x)
        if group not in queue_set:
            queue.append(group)
            queue_set.add(group)
    elif command[0] == 'DEQUEUE':
        if queue:
            group = queue[0]
            x = groups[group].popleft()
            print(x)
            if not groups[group]:
                queue.popleft()
                queue_set.remove(group)
```

## 三 树

树本质上是由一些拥有单向二元关系的单位元组成的数据结构，即树的本质是单向图，只不过树中不会出现环

类对象可以添加一些属性，我们经常利用其描述树

在对树进行成功的刻画之后，我们便可以使用其去解决一些问题

### 1 树的基本问题

在算法中，一般用类对象来描述树的数据结构，使用value，neighbor等指标来刻画节点，从而构建出树的数据结构，从而解决以树为数据结构的问题

#### 01760:Disk Tree http://cs101.openjudge.cn/2024sp_routine/01760/

建立文件树，利用经典的类对象来表示节点，建立表达树函数与建造树函数，是经典的算法

使用了defaultdict函数辅助实现，该函数的作用是可以给字典一个默认值

通过value，neighbor，neighborlabor三个值去描绘节点

```python
from collections import defaultdict
class TreeNode:
    def __init__(self,value):
        self.value=value
        self.neighbor=[]
        self.neighborlabor=[]
def print_tree(roots):
    roots.sort(key=lambda x: x.value)
    out=[]
    for i in roots:
        out.append(i.value)
        if i.neighbor:
            i.neighbor.sort(key=lambda x: x.value)
        for j in i.neighbor:
            for k in print_tree([j]):
                out.append(' '+k)
    return out

def build_tree(tree):
    rootslabor=[]
    roots=[]
    for line in tree:
        if line[0] not in rootslabor:
            rootslabor.append(line[0])
            roots.append(TreeNode(line[0]))
            node=roots[-1]
        else:
            t0=rootslabor.index(line[0])
            node=roots[t0]
        for t in range(1,len(line)):
            if line[t] not in node.neighborlabor:
                node.neighborlabor.append(line[t])
                node1=TreeNode(line[t])
                node.neighbor.append(node1)
                node=node1
            else:
                t0=node.neighborlabor.index(line[t])
                node1=node.neighbor[t0]
                node=node1
    for k in print_tree(roots):
        print(k)

n=int(input())
tree=[]
for _ in range(n):
    l=input()
    l2=l.strip()
    line=l2.split("\\")
    tree.append(line)
build_tree(tree)
```

#### 02775 文件结构“图” http://cs101.openjudge.cn/2024sp_routine/02775/

抄写的讲义代码，非常简洁的表达，值得学习

```python
from sys import exit

class dir:
    def __init__(self, dname):
        self.name = dname
        self.dirs = []
        self.files = []
    
    def getGraph(self):
        g = [self.name]
        for d in self.dirs:
            subg = d.getGraph()
            g.extend(["|     " + s for s in subg])
        for f in sorted(self.files):
            g.append(f)
        return g

n = 0
while True:
    n += 1
    stack = [dir("ROOT")]
    while (s := input()) != "*":
        if s == "#": exit(0)
        if s[0] == 'f':
            stack[-1].files.append(s)
        elif s[0] == 'd':
            stack.append(dir(s))
            stack[-2].dirs.append(stack[-1])
        else:
            stack.pop()
    print(f"DATA SET {n}:")
    print(*stack[0].getGraph(), sep='\n')
    print()
```

#### 04081 树的转换 http://cs101.openjudge.cn/2024sp_routine/04081/

虽然是树的问题，但是使用递归实现了

这个转换过程在笔试中会出现，建议还是使用树的算法编译一边

```python
def f(x):
    t=0
    a=0
    for j in range(len(x)):
        if x[j]=='d':
            a+=1
        else:a-=1
        if t<a:
            t=a
    return t
def g(x):
    a=0
    if len(x)>0:
        for j in range(len(x)):
            if x[j]=='d':
                a+=1
            else:a-=1
            if a==0:
                t=j
                break
        if t==len(x)-1:
            return g(x[1:t])+1
        else:
            return max(1+g(x[t+1:]),g(x[:t+1]))
    else:return 0
l=input()
print(str(f(l))+' => '+str(g(l)))
```

#### 04082 树的镜面映射 http://cs101.openjudge.cn/2024sp_routine/04082/

经典的树算法，除了宽度优先遍历序列在输出的时候需要bfs的思想，即需要deque双端队列函数辅助实现

```python
from collections import deque

class TreeNode:
    def __init__(self, x):
        self.x = x
        self.children = []

def create_node():
    return TreeNode('')

def build_tree(tempList, index):
    node = create_node()
    node.x = tempList[index][0]
    if tempList[index][1] == '0':
        index += 1
        child, index = build_tree(tempList, index)
        node.children.append(child)
        index += 1
        child, index = build_tree(tempList, index)
        node.children.append(child)
    return node, index

def print_tree(p):
    Q = deque()
    s = deque()

    while p is not None:
        if p.x != '$':
            s.append(p)
        p = p.children[1] if len(p.children) > 1 else None

    while s:
        Q.append(s.pop())

    while Q:
        p = Q.popleft()
        print(p.x, end=' ')

        if p.children:
            p = p.children[0]
            while p is not None:
                if p.x != '$':
                    s.append(p)
                p = p.children[1] if len(p.children) > 1 else None

            while s:
                Q.append(s.pop())


n = int(input())
tempList = input().split()

root, _ = build_tree(tempList, 0)

print_tree(root)
```

#### 04089 电话号码 http://cs101.openjudge.cn/2024sp_routine/04089/

抄写的答案代码，search函数用来判断这个新的输入是不是某个前面已经输入的数的延长版，insert就是把新的数加入这些里面

一种比较新奇的树的形式

```python
class TrieNode:
    def __init__(self):
        self.child={}


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, nums):
        curnode = self.root
        for x in nums:
            if x not in curnode.child:
                curnode.child[x] = TrieNode()
            curnode=curnode.child[x]

    def search(self, num):
        curnode = self.root
        for x in num:
            if x not in curnode.child:
                return 0
            curnode = curnode.child[x]
        return 1


t = int(input())
p = []
for _ in range(t):
    n = int(input())
    nums = []
    for _ in range(n):
        nums.append(str(input()))
    nums.sort(reverse=True)
    s = 0
    trie = Trie()
    for num in nums:
        s += trie.search(num)
        trie.insert(num)
    if s > 0:
        print('NO')
    else:
        print('YES')
```

#### 07161 森林的带度数层次序列存储 http://cs101.openjudge.cn/2024sp_routine/07161/

层次遍历可以通过递归实现，其余都是经典算法

```python
class TreeNode:
    def __init__(self,value):
        self.value=value
        self.children=[]
def buildTree(s):
    stack1=[]
    stack2=[]
    for _ in range(len(s)//2):  
        if _==0:
           root=TreeNode(s[0])
           x=int(s[1])
           stack1.append(root)
           stack2.append(x)
        else:
            node=TreeNode(s[_*2])
            stack1[0].children.append(node)
            if int(s[_*2+1])!=0:
                stack2.append(int(s[_*2+1]))
                stack1.append(node)
            stack2[0]-=1
            if stack2[0]==0:
                stack2.pop(0)
                stack1.pop(0)
    return root
def back(s):
    l=str()
    for k in s.children:
        l+=back(k)
    return l+str(s.value)
n=int(input())
out=str()
for _ in range(n):
    s=input().split()
    out+=back(buildTree(s))
print(' '.join(str(d) for d in out))
```

#### 24728:括号嵌套树 http://cs101.openjudge.cn/2024sp_routine/24728/

虽然是树的题目，但是没有用树的经典算法去写，使用了递归的思想

```python
a=input()
net=['(',')',',']
def fro(x):
    if x:
        y=''
        for j in range(len(x)):
            if x[j] not in net:
                y=y+x[j]
        return y
    else:
        return ''
def bac(x):
    if len(x)>1:
        trees=[]
        stack=[]
        tree=''
        for j in range(2,len(x)-1):
            if x[j] in '()':
                stack.append(x[j])
                tree=tree+x[j]
            while len(stack)>0 and stack[-1]==')':
                stack.pop()
                stack.pop()
            if x[j] not in net:
                tree=tree+x[j]
            if len(stack)==0 and j==len(x)-2 :
                trees.append(tree)
                tree=''
            if len(stack)==0 and x[j+1]==',' :
                trees.append(tree)
                tree=''
            if len(stack)!=0 and x[j]==',':
                tree=tree+x[j]
        s=''
        for tree in trees:
            s=s+bac(tree)
        return s+x[0]
    elif len(x)==1:
        return x
    else:
        return ''
print(fro(a))
print(bac(a))
```

#### 25140:根据后序表达式建立表达式树http://cs101.openjudge.cn/2024sp_routine/25140/

经典的树的算法，使用value，children刻画节点，用树的数据结构实现

```python
class TreeNode:
    def __init__(self,value):
        self.value=value
        self.children = []
def build_tree(s):
    stack=[]
    node=None
    for char in s:
        if char.isupper():
            node=TreeNode(char)
            node.children.append(stack.pop())
            node.children.append(stack.pop())
            stack.append(node)
        else:
            node=TreeNode(char)
            stack.append(node)
    return node
def listorder(node):
    output=[node.value]
    storey=[node]
    while len(storey)>0:
        new_storey=[]
        for nod in storey:
            for child in nod.children:
                new_storey.append(child)
        storey=new_storey  
        for l in range(len(storey)):
            output.insert(0,storey[len(storey)-1-l].value)
    return ''.join(output)
i=int(input())
for j in range(i):
    s=input()
    gen=build_tree(s)
    print(listorder(gen))
```

#### 27928:遍历树 http://cs101.openjudge.cn/2024sp_routine/27928/

甚至没有用类实现，又是一次用的defaultdict字典实现的树的遍历

```python
from collections import defaultdict
n = int(input())
dic = {}
parent = defaultdict(int)
for i in range(n):
    temp = list(map(int, input().split()))
    dic[temp[0]] = temp
    if len(temp) > 1:
        for x in temp[1:]:
            parent[x] = temp[0]

def travel(root):
    if len(dic[root]) == 1:
        return dic[root]
    else:
        temp = []
        temp1 = dic[root]
        temp1.sort()
        for x in temp1:
            if x == root:
                temp += [root]
            else:
                temp += travel(x)
        return temp

nodes = list(dic.keys())
for x in nodes:
    if parent[x] == 0:
        root = x
        break
ans = travel(root)
for x in ans:
    print(x)
```

### 2 二叉树

二叉树的节点更多地使用value，left（左节点），right（右节点）刻画，其他的和树的一般问题保持一致

二叉树也有许多特殊的性质，二叉树的数据结构便于实现后续的一些进阶问题

#### 01145 Tree Summing http://cs101.openjudge.cn/2024sp_routine/01145/

这个题本质上其实就是普通的二叉树问题，只不过最头疼的是它的输入数据很难处理，不仅需要考虑换行和空格，还需要通过判断括号是否匹配判断一组数据是否输入完毕，这个代码是我亲自写的，又臭又长，但是最后还是实现了

```python
class TreeNode:
    def __init__(self, value): 
        self.value = value
        self.weight= int(value)
        self.ismove=False

def parse_tree(s):
    stack=[]
    node=None
    name=''
    rot=''
    p=1
    if s[1]!='-':
        for char in s[1:]:
            if char.isdigit():
                rot+=char
            else:
                break
    else:
        for char in s[2:]:
            if char.isdigit():
                rot+=char
            else:
                break
        rot=int(rot)
        rot*=-1
    leaves=[int(rot)]
    for char in s:
        m=[]
        for k in stack:
            m.append(k.value)
        if char=='(':
            if name:
                node=TreeNode(int(name)*p)
                if stack:
                    node.weight=stack[-1].weight+node.value
                    if not stack[-1].ismove:
                        leaves.remove(stack[-1].weight)
                        stack[-1].ismove=True
                    leaves.append(node.weight)
                stack.append(node)
                node=None
                name=''
                p=1
        elif char==')':
            if name:
                node=TreeNode(int(name)*p)
                if stack:
                    node.weight=stack[-1].weight+node.value
                    if not stack[-1].ismove:
                        leaves.remove(stack[-1].weight)
                        stack[-1].ismove=True
                    leaves.append(node.weight)
                node=None
                name=''
                p=1
            if stack:
                node=stack.pop()
        elif char.isdigit():
            name+=char        
        elif char==',':
            if name:
                node=TreeNode(int(name)*p)
                if stack:
                    node.weight=stack[-1].weight+node.value
                    if not stack[-1].ismove:
                        leaves.remove(stack[-1].weight)
                        stack[-1].ismove=True
                    leaves.append(node.weight)
                node=None
                name=''
                p=1
        else:
            p=-1
    return leaves
s=[]
l=''
while True:
    try:n=input().replace(' ','')
    except EOFError:
        break
    else:
        for _ in n:
            if _ =='(':
                s.append('(')
            if _ ==')':s.pop()
        if s:
            l+=n
        else:
            l+=n
            a=l.index('(')
            t=int(l[:a])
            l_0=l[a:]
            l_0=l_0.replace('()','')
            l_0=l_0.replace(')(',',')
            if l_0:
                h=parse_tree(l_0)
                if t in h:
                    print('yes')
                else:
                    print('no')
            else:
                if t!=0:
                    print('no')
                else:
                    print('yes')
            l=''
```

#### 01577 Falling Leaves http://cs101.openjudge.cn/2024sp_routine/01577/

经典的树的遍历问题，代码些许丑陋

```python
alp=dict()
alp['A']=1
alp['B']=2
alp['C']=3
alp['D']=4
alp['E']=5
alp['F']=6
alp['G']=7
alp['H']=8
alp['I']=9
alp['J']=10
alp['K']=11
alp['L']=12
alp['M']=13
alp['N']=14
alp['O']=15
alp['P']=16
alp['Q']=17
alp['R']=18
alp['S']=19
alp['T']=20
alp['U']=21
alp['V']=22
alp['W']=23
alp['X']=24
alp['Y']=25
alp['Z']=26
class TreeNode:
    def __init__(self,value):
        self.value=value
        self.left=None
        self.right=None
def preorder(x):
    if x:
        if x.left:
            x1=x.left
        if x.right:
            x2=x.right
        if x.left and x.right:
            return x.value+preorder(x1)+preorder(x2)
        elif x.left:
            return x.value+preorder(x1)
        elif x.right:
            return x.value+preorder(x2)
        else:return x.value

def out(tree):
    root=TreeNode(tree[-1])
    tree.pop()
    while tree:
        leaves=tree.pop()
        for k in leaves:
            k0=alp[k]
            node=root
            while True:
                if alp[node.value]>k0:
                    if node.left:
                        node=node.left
                    else:
                        node.left=TreeNode(k)
                        break
                if alp[node.value]<k0:
                    if node.right:
                        node=node.right
                    else:
                        node.right=TreeNode(k)
                        break
    return preorder(root)


tree=[]
while True:
    leaves=input()
    if leaves=='*':
        print(out(tree))
        tree=[]
    elif leaves=='$':
        print(out(tree))
        break
    else:
        tree.append(leaves)
```

#### 02255 重建二叉树 http://cs101.openjudge.cn/2024sp_routine/02255/

这里没有使用树的数据结构去实现，而是使用了递归

```python
def f_m(x,y):
    if x:
        pos=y.find(x[0])
        return f_m(x[1:pos+1],y[:pos])+f_m(x[pos+1:],y[pos+1:])+x[0]
    else:
        return ''
while True:
    try:
        s=input()
    except EOFError:
        break
    else:
        a,b=s.split()
        print(f_m(a,b))
```

#### 03720 文本二叉树 http://cs101.openjudge.cn/2024sp_routine/03720/

经典的树算法

```python
class TreeNode():
    def __init__(self,value):
        self.value=value
        self.left=None
        self.right=None
def build_tree(x):
    stack=[0]*101
    stack_pos=[0]*101
    for j in range(len(x)):
        node = TreeNode(x[j])
        stack[n[j]]=node
        stack_pos[n[j]]=0
        if stack[n[j]-1]!=0:
            if stack_pos[n[j]-1]==0:
                stack[n[j]-1].left=node
                stack_pos[n[j]-1]+=1
            else:
                stack[n[j]-1].right=node
                stack_pos[n[j]-1]=0
                if n[j] !=2:
                    stack[n[j]-1]=0
    return stack[1]
def preorder(x):
    if x == None or x.value =='*':
        return ''
    else:
        return x.value+preorder(x.left)+preorder(x.right)
def midorder(x):
    if x == None or x.value=='*':
        return ''
    else:
        return midorder(x.left)+x.value+midorder(x.right)
def bacorder(x):
    if x == None or x.value=='*':
        return ''
    else:
        return bacorder(x.left)+bacorder(x.right)+x.value
m=''
n=[]
i=int(input())
for z in range(i):
    while True:
        s=input()
        if s=='0':
            break
        else:
            m=m+s[-1]
            n.append(len(s))
    root=build_tree(m)
    print(preorder(root))
    print(bacorder(root))
    print(midorder(root))
    print('')
```



#### 06646 二叉树的深度 http://cs101.openjudge.cn/2024sp_routine/06646/

还不完全掌握树的经典算法的时候编的，使用字典来实现

```python
i=int(input())
tree=dict()
s=0
for j in range(i):
    d=list(map(int,input().split()))
    tree[j]=d
leaves=[1]
l_l=dict()
l_l[1]=1
ml=1
while s<i:
    for leaf in leaves:
        s+=1
        if tree[leaf-1][0]!= -1:
            l_l[tree[leaf-1][0]]=l_l[leaf]+1
            leaves.append(tree[leaf-1][0])
            if l_l[tree[leaf-1][0]]>ml:
                ml=l_l[tree[leaf-1][0]]
        if tree[leaf-1][1]!= -1:
            l_l[tree[leaf-1][1]]=l_l[leaf]+1
            leaves.append(tree[leaf-1][1])
            if l_l[tree[leaf-1][1]]>ml:
                ml=l_l[tree[leaf-1][1]]
        leaves.remove(leaf)
print(ml)
```

#### 22158:根据二叉树前中序序列建树 http://cs101.openjudge.cn/2024sp_routine/22158/

又一次没有使用类实现，使用了递归，通过前序表达式可以找出根，从而推出左树与右树的前序和中序表达式，进而可以进行递归

```python
def f_m(x,y):
    if x:
        pos=y.find(x[0])
        return f_m(x[1:pos+1],y[:pos])+f_m(x[pos+1:],y[pos+1:])+x[0]
    else:
        return ''
while True:
    try:
        a=input()
        b=input()
    except EOFError:
        break
    else:
        print(f_m(a,b))
```

#### 22275 二叉搜索树的遍历 http://cs101.openjudge.cn/2024sp_routine/22275/

利用二叉搜索树的性质：小于父节点的键都在左子树中，大于父节点的键则都在右子树中

结合前序遍历序列建树，然后输出后序遍历序列

```python
class Node():
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None


def buildTree(preorder):
    if len(preorder) == 0:
        return None

    node = Node(preorder[0])

    idx = len(preorder)
    for i in range(1, len(preorder)):
        if preorder[i] > preorder[0]:
            idx = i
            break
    node.left = buildTree(preorder[1:idx])
    node.right = buildTree(preorder[idx:])

    return node


def postorder(node):
    if node is None:
        return []
    output = []
    output.extend(postorder(node.left))
    output.extend(postorder(node.right))
    output.append(str(node.val))

    return output


n = int(input())
preorder = list(map(int, input().split()))
print(' '.join(postorder(buildTree(preorder))))
```

#### 25145:猜二叉树 http://cs101.openjudge.cn/2024sp_routine/25145/

通过后序表达式可以找出根，然后就可以找出左儿子和右儿子，如此建树然后层次遍历输出

```python
class TreeNode:
    def __init__(self,value):
        self.value=value
        self.left= None
        self.right= None
def build_tree(x,y):
    if not x:
        return None
    else:
        root_val=y[-1]
        root=TreeNode(root_val)
        pos=x.index(root_val)
        root.left=build_tree(x[:pos],y[:pos])
        root.right=build_tree(x[pos+1:],y[pos:len(y)-1])
        return root
def listorder(x):
    output=[]
    storey=[x]
    while len(storey)>0:
        net=[]
        new_storey=[]
        for j in storey:
            if j.left !=None:
                new_storey.append(j.left)
            if j.right !=None:
                new_storey.append(j.right)
            net.append(j.value)
        storey=new_storey 
        output+=net
    return ''.join(output)
i=int(input())
for j in range(i):
    s=input()
    t=input()
    rot=build_tree(s,t)
    print(listorder(rot))
```

#### 27638:求二叉树的高度和叶子数目 http://cs101.openjudge.cn/2024sp_routine/27638/

还不完全掌握树的经典算法的时候编的，使用字典来实现

```python
i=int(input())
tree=dict()
for j in range(i):
    d=list(map(int,input().split()))
    tree[j]=d
ml=0
ll=0
for j in range(i):
    leaves=[j]
    l_l=dict()
    l_l[j]=0
    while len(leaves)>0:
        for leaf in leaves:
            if tree[leaf][0]!= -1:
                l_l[tree[leaf][0]]=l_l[leaf]+1
                leaves.append(tree[leaf][0])
                if l_l[tree[leaf][0]]>ml:
                    ml=l_l[tree[leaf][0]]
            if tree[leaf][1]!= -1:
                l_l[tree[leaf][1]]=l_l[leaf]+1
                leaves.append(tree[leaf][1])
                if l_l[tree[leaf][1]]>ml:
                    ml=l_l[tree[leaf][1]]
            leaves.remove(leaf)
for j in range(i):
    if tree[j]==[-1,-1]:
        ll+=1
print(str(ml)+' '+str(ll))
```

### 3 并查集

并查集是一种常用的树结构，它把一些孩子的父亲并为一起，然后进行一系列查找

我们使用union函数将孩子的父亲合并，用find函数实现查找时的路径缩减

#### 01182 食物链 http://cs101.openjudge.cn/2024sp_routine/01182/

使用扩展版的孩子集去表达并查集，这样可以判断不该在同一集里的孩子有没有被分到同一集

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            if self.rank[rootX] > self.rank[rootY]:
                self.parent[rootY] = rootX
            elif self.rank[rootX] < self.rank[rootY]:
                self.parent[rootX] = rootY
            else:
                self.parent[rootY] = rootX
                self.rank[rootX] += 1

def solve():
    x=0
    n, m = map(int, input().split())
    uf = UnionFind(3 * n)
    for _ in range(m):
        operation, a, b = map(int,input().split())
        a, b = a - 1, b - 1
        if a>n-1 or b>n-1:
            x+=1
        else:
            if operation == 1:
                if uf.find(a)==uf.find(b+n) or uf.find(a)==uf.find(b+2*n):
                    x+=1
                else:
                    if a!=b:
                        uf.union(a, b)
                        uf.union(a + n, b + n)
                        uf.union(a + 2*n, b + 2*n) 
            else:
                if uf.find(a)==uf.find(b) or uf.find(a)==uf.find(b+2*n)or a==b:
                    x+=1
                else:
                    if uf.find(a)!=uf.find(b+n):
                        uf.union(a, b + n)
                        uf.union(a + n, b + 2*n)
                        uf.union(a + 2*n, b)
    print(x)
solve()
```

#### 01703 发现它，抓住它 http://cs101.openjudge.cn/2024sp_routine/01703/

和食物链使用的类似的算法

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            if self.rank[rootX] > self.rank[rootY]:
                self.parent[rootY] = rootX
            elif self.rank[rootX] < self.rank[rootY]:
                self.parent[rootX] = rootY
            else:
                self.parent[rootY] = rootX
                self.rank[rootX] += 1

def solve():
    n, m = map(int, input().split())
    uf = UnionFind(2 * n)
    for _ in range(m):
        operation, a, b = input().split()
        a, b = int(a) - 1, int(b) - 1
        if operation == "D":
            uf.union(a, b + n) 
            uf.union(a + n, b)  
        else: 
            if uf.find(a) == uf.find(b):
                print("In the same gang.")
            elif uf.find(a) == uf.find(b + n):
                print("In different gangs.")
            else:
                print("Not sure yet.")

T = int(input())
for _ in range(T):
    solve()
```

#### 02386 Lake Counting http://cs101.openjudge.cn/2024sp_routine/02386/

经典并查集问题，使用union函数合并两个孩子，再使用find函数缩减找家长的路径

```python
m,n=map(int,input().split())
def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]
def union(a,b):
    roota=find(a)
    rootb=find(b)
    if roota!=rootb:
        parent[roota]=rootb
parent=[x for x in range(m*n)]
nodes=[False for _ in range(n*m)]
for x in range(m):
    s=input()
    for y in range(n):
        if s[y]=='W':
            nodes[x*n+y]=True
for x in range(m):
    for y in range(n):
        if nodes[x*n+y]:
            if x>=1:
                if nodes[(x-1)*n+y]:
                    union(x*n+y,(x-1)*n+y)
                if y>=1:
                    if nodes[(x-1)*n+y-1]:
                        union(x*n+y,(x-1)*n+y-1)
                if y<n-1:
                    if nodes[(x-1)*n+y+1]:
                        union(x*n+y,(x-1)*n+y+1)
            if y>=1:
                if nodes[x*n+y-1]:
                    union(x*n+y,x*n+y-1)
visited=[False]*(m*n)
count=0
for i in range(n*m):
    if nodes[i] and find(i)==i:
        count+=1
print(count)
```

#### 07734:虫子的生活 http://cs101.openjudge.cn/2024sp_routine/07734/

一种自创的方法，把并查集的parent变成二元列表来实现

但是因为迭代次数过多，尽管进行了路径缩减，但还是一直re，最后使用了前两行代码增加迭代次数终于通过

```python
import sys
sys.setrecursionlimit(100000)
n=int(input())
for _ in range(n):
    def find(x):
        if parent[x][0] != x: 
            if parent[x][1]==1:
                parent[x][0],parent[x][1]= find(parent[x][0])[0],1-find(parent[x][0])[1]
            else:parent[x]=find(parent[x][0])
        return parent[x]

    def union(y,z):
        if find(y)==find(z):
            print('Suspicious bugs found!')
            return False
        elif find(y)[1]==0:
            a=find(y)[0]
            b=find(z)
            parent[a][1],parent[a][0]=1-b[1],b[0]
            return True
        else:
            a=find(y)[0]
            b=find(z)
            parent[a]=b
            return True
    p=True
    print('Scenario #%d:' % (_+1))
    s,t=map(int,input().split())
    parent=list()
    for h in range(s+1):
        parent.append([h,0])
    for k in range(t):
        y,z=map(int,input().split())
        if p:
            p=union(y,z)
    if p:
        print('No suspicious bugs found!')
    print()
```

#### 18250:冰阔落 I http://cs101.openjudge.cn/2024sp_routine/18250/

经典并查集问题，使用union函数合并两个孩子，再使用find函数缩减找家长的路径

```python
def find(x):
    if parent[x]!=x:
        parent[x]=find(parent[x])
        return parent[x]
    else:
        return x
def union(x,y):
    parent[find(y)]=x
while True:
    try:
        a,b=map(int,input().split())
    except EOFError:break
    else:
        parent=list(range(a+1))
        for _ in range(b):
            u,v=map(int,input().split())
            if find(u)==find(v):
                print('Yes')
            else:
                print('No')
                union(u,v)
        net=[]
        for t in range(1,a+1):
            if parent[t]==t:
                net.append(str(t))
        print(len(net))
        print(' '.join(net))
```

#### 27880:繁忙的厦门 http://cs101.openjudge.cn/2024sp_routine/27880/

使用并查集判断是否“连通”，第一个值必然是n-1（图论知识分析），第二个值是依次把权值小的边练起来直到“连通”

```python
n,m=map(int,input().split())
edges=[]
parent=[ x for x in range(n+1)]
def find(x):
    if parent[x] !=x:
        parent[x]=find(parent[x])
    return parent[x]
def union(x,y):
    parent[find(x)]=find(y)

for _ in range(m):
    u,v,w=map(int,input().split())
    edges.append([u,v,w])
edges.sort(key=lambda x: x[2])
for x,y,z in edges:
    union(x,y)
    f=True
    net=[]
    for l in range(1,n):
        net.append(find(l))
        if find(l)!=find(l+1):
            f=False
    net.append(find(n))
    if f:
        s=z
        break
print(n-1,s)
```

### 4 huffman树

huffman树是一种特殊的树，它通过依次将列表中最小的两个数加起来再重新放入列表，并不断操作至结束的过程看做一个树结构

在具体代码实现的时候，可以调用bisect函数

#### 04080 huffman编码树 http://cs101.openjudge.cn/2024sp_routine/04080/

根据字符使用频率(权值)生成一棵唯一的哈夫曼编码树。生成树时需要遵循以下规则以确保唯一性：

选取最小的两个节点合并时，节点比大小的规则是:

1. 权值小的节点算小。权值相同的两个节点，字符集里最小字符小的，算小。

例如 （{'c','k'},12) 和 ({'b','z'},12)，后者小。

1. 合并两个节点时，小的节点必须作为左子节点
2. 连接左子节点的边代表0,连接右子节点的边代表1

然后对输入的串进行编码或解码

**输入**

第一行是整数n，表示字符集有n个字符。 接下来n行，每行是一个字符及其使用频率（权重）。字符都是英文字母。 再接下来是若干行，有的是字母串，有的是01编码串。

**输出**

对输入中的字母串，输出该字符串的编码 对输入中的01串,将其解码，输出原始字符串

样例输入

```
3
g 4
d 8
c 10
dc
110
```

样例输出

```
110
dc
```

建树：主要利用最小堆，每次取出weight最小的两个节点，weight相加后创建节点，连接左右孩子，再入堆，直至堆中只剩一个节点.

编码：跟踪每一步走的是左还是右，用0和1表示，直至遇到有char值的节点，说明到了叶子节点，将01字串添加进字典.

解码：根据01字串决定走左还是右，直至遇到有char值的节点，将char值取出.

```python
import heapq

class Node:
    def __init__(self, weight, char=None):
        self.weight = weight
        self.char = char
        self.left = None
        self.right = None

    def __lt__(self, other):
        if self.weight == other.weight:
            return self.char < other.char
        return self.weight < other.weight

def build_huffman_tree(characters):
    heap = []
    for char, weight in characters.items():
        heapq.heappush(heap, Node(weight, char))

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        #merged = Node(left.weight + right.weight) #note: 合并后，char 字段默认值是空
        merged = Node(left.weight + right.weight, min(left.char, right.char))
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)

    return heap[0]

def encode_huffman_tree(root):
    codes = {}

    def traverse(node, code):
        #if node.char:
        if node.left is None and node.right is None:
            codes[node.char] = code
        else:
            traverse(node.left, code + '0')
            traverse(node.right, code + '1')

    traverse(root, '')
    return codes

def huffman_encoding(codes, string):
    encoded = ''
    for char in string:
        encoded += codes[char]
    return encoded

def huffman_decoding(root, encoded_string):
    decoded = ''
    node = root
    for bit in encoded_string:
        if bit == '0':
            node = node.left
        else:
            node = node.right

        #if node.char:
        if node.left is None and node.right is None:
            decoded += node.char
            node = root
    return decoded

# 读取输入
n = int(input())
characters = {}
for _ in range(n):
    char, weight = input().split()
    characters[char] = int(weight)

#string = input().strip()
#encoded_string = input().strip()

# 构建哈夫曼编码树
huffman_tree = build_huffman_tree(characters)

# 编码和解码
codes = encode_huffman_tree(huffman_tree)

strings = []
while True:
    try:
        line = input()
        strings.append(line)

    except EOFError:
        break

results = []
#print(strings)
for string in strings:
    if string[0] in ('0','1'):
        results.append(huffman_decoding(huffman_tree, string))
    else:
        results.append(huffman_encoding(codes, string))

for result in results:
    print(result)
```

#### 05333 Fence Repair http://cs101.openjudge.cn/2024sp_routine/05333/

现在需要n个木板，且给定这n个木板的长度。现有一块长度为这n个木板长度之和的长木板，需要把这个长木板分割需要的n块（一空需要切n-1刀）。每次切一刀时，切之前木板的长度是本次切割的成本。（例如，将长度为21的木板切成长度分别为8、5、8的三块。切第一刀时的成本为21，将其切成长度分别为13和8的两块。第二刀成本为13，并且将木板切成长度为8和5的两块，这样工作完成，总成本为21+13=34。另外，假如第一刀将木板切成长度为16和5的两块，则总开销为21+16=37，比上一个方案开销更大）。请你设计一种切割的方式，使得最后切完后总成本最小。

输入：
第1行：一个整数n，为需要的木板数量
第2行----第n+1行：每块木板的长度

输出：
一个整数，最小的总成本

```python
import bisect
N=int(input())
ribbons=[]
for _ in range(N):
    c=int(input())
    ribbons.append(-c)
ribbons=sorted(ribbons)
mini=0
for i in range(N-1):
    A=ribbons.pop()
    B=ribbons.pop()
    mini-=A+B
    bisect.insort(ribbons,A+B)
print(mini)
```

使用bisect内置函数的调用

#### 18164:剪绳子 http://cs101.openjudge.cn/2024sp_routine/18164/

huffman编码树的数据结构，用bisect内置函数进行实现

```python
import bisect
N=int(input())
ribbons=sorted(list(map(lambda x:-int(x),input().split())))
mini=0
for i in [0]*(N-1):
    A=ribbons.pop()
    B=ribbons.pop()
    mini-=A+B
    bisect.insort(ribbons,A+B)
print(mini)
```



## 四 图

图本质上是由一些拥有双向（无向）二元关系的单位元组成的数据结构，即图的本质是无向图

我们经常利用字典描述图

在对图进行成功的刻画之后，我们便可以基于一些算法解决一些问题

### 1 图的基本问题

#### 20741:两座孤岛最短距离 http://cs101.openjudge.cn/2024sp_routine/20741/

利用图的节点的关系分化出两个岛屿，然后依次计算岛屿中点的距离来实现

```python
n=int(input())
land=[]
island_1=[]
island_2=[]
for k in range(n):
    s=input()
    for l in range(n):
        if s[l]=='1':
            land.append([k,l])
x=land.pop()
island_1.append(x)
b=[x]
t=1
while t:
    t=0
    c=[]
    for y in b:
        if y[0]>0:
            if [y[0]-1,y[1]] in land:
                island_1.append([y[0]-1,y[1]])
                c.append([y[0]-1,y[1]])
                land.remove([y[0]-1,y[1]])
                t+=1
        if y[0]<n-1:
            if [y[0]+1,y[1]] in land:
                island_1.append([y[0]+1,y[1]])
                c.append([y[0]+1,y[1]])
                land.remove([y[0]+1,y[1]])
                t+=1
        if y[1]>0:
            if [y[0],y[1]-1] in land:
                island_1.append([y[0],y[1]-1])
                c.append([y[0],y[1]-1])
                land.remove([y[0],y[1]-1])
                t+=1
        if y[1]<n-1:
            if [y[0],y[1]+1] in land:
                island_1.append([y[0],y[1]+1])
                c.append([y[0],y[1]+1])
                land.remove([y[0],y[1]+1])
                t+=1
        b=c
island_2=land
d=2*n
for x1 in island_1:
    for x2 in island_2:
        d_0=abs(x1[0]-x2[0])+abs(x1[1]-x2[1])
        if d_0<d:
            d=d_0
print(d-1)
```

### 2 图的bfs算法

广度优先搜索算法，利用visited列表和访问node列表来辅助实现，更多地解决最值问题

#### 01611 The Suspects http://cs101.openjudge.cn/2024sp_routine/01611/

其实可以用并查集实现，但是这里使用了bfs算法

```python
from collections import deque
def bfs(n,graph):
    visited=[False]*n
    visited[0]=True
    nodes=[0]
    while nodes:
        k=nodes.pop(0)
        for neighbor in graph[k]:
            if not visited[neighbor]:
                nodes.append(neighbor)
                visited[neighbor]=True
    out=0
    for _ in visited:
        if _:
            out+=1
    return out
outt=[]
while True:
    n,m=map(int,input().split())
    if n==0 and m==0:break
    else:
        stus=[]
        graph=[[] for _ in range(n)]
        for _ in range(m):
            group=list(map(int,input().split()))
            a=group.pop(0)
            for k in range(1,a):
                graph[group[0]].append(group[k])
                graph[group[k]].append(group[0])
        print(bfs(n,graph))
```

#### 04116 拯救行动 http://cs101.openjudge.cn/2024sp_routine/04116/

抄写的答案代码，直接利用bfs实现十分困难，需要借助堆来实现

```python
from heapq import heappush, heappop

dx = [-1, 1, 0, 0]
dy = [0, 0, -1, 1]


def bfs(matrix, start):
    n, m = len(matrix), len(matrix[0])
    visited = [[False for _ in range(m)] for _ in range(n)]
    q = []
    heappush(q, (0, start[0], start[1]))
    visited[start[0]][start[1]] = True
    while len(q) != 0:
        time, x, y = heappop(q)
        for i in range(4):
            nx, ny = x + dx[i], y + dy[i]
            if 0 <= nx < n and 0 <= ny < m and not visited[nx][ny]:
                if matrix[nx][ny] == "a":
                    return time + 1
                elif matrix[nx][ny] == "@":
                    heappush(q, (time + 1, nx, ny))
                    visited[nx][ny] = True
                elif matrix[nx][ny] == "x":
                    heappush(q, (time + 2, nx, ny))
                    visited[nx][ny] = True

    return "Impossible"


S = int(input())
for _ in range(S):
    N, M = map(int, input().split())
    matrix = [list(input()) for _ in range(N)]
    start = None
    ans = []
    for i in range(N):
        for j in range(M):
            if matrix[i][j] == "r":
                start = (i, j)
                break
    print(bfs(matrix, start))
```



#### 07206:我是最快的马 http://cs101.openjudge.cn/2024sp_routine/07206/

一种朴素的本质上是bfs的算法，还不完全掌握时自己研究探索出来的

```python
from collections import defaultdict
x1,y1=map(int,input().split())
start=[x1,y1]
x2,y2=map(int,input().split())
goal=[x2,y2]
m=int(input())
horses=[]
for _ in range(m):
    horse=list(map(int,input().split()))
    horses.append(horse)
steps=0
places=[[(x1,y1)]]
placesnow=[(x1,y1)]

before=defaultdict(list)
while True and steps<10:
    steps+=1
    placesnext=[]
    for place in placesnow:
        if [place[0]+1,place[1]] not in horses:
            placesnext.append((place[0]+2,place[1]+1))
            placesnext.append((place[0]+2,place[1]-1))
            before[(place[0]+2,place[1]+1)].append(place)
            before[(place[0]+2,place[1]-1)].append(place)
        if [place[0]-1,place[1]] not in horses:
            placesnext.append((place[0]-2,place[1]+1))
            placesnext.append((place[0]-2,place[1]-1))
            before[(place[0]-2,place[1]+1)].append(place)
            before[(place[0]-2,place[1]-1)].append(place)
        if [place[0],place[1]+1] not in horses:
            placesnext.append((place[0]+1,place[1]+2))
            placesnext.append((place[0]-1,place[1]+2))
            before[(place[0]+1,place[1]+2)].append(place)
            before[(place[0]-1,place[1]+2)].append(place)
        if [place[0],place[1]-1] not in horses:
            placesnext.append((place[0]+1,place[1]-2))
            placesnext.append((place[0]-1,place[1]-2))
            before[(place[0]+1,place[1]-2)].append(place)
            before[(place[0]-1,place[1]-2)].append(place)
    if (x2,y2) in placesnext:
        root=[(x2,y2)]
        place=(x2,y2)
        f=True
        st=steps
        while True:
            if st:
                if len(before[place])==1:
                    st-=1
                    root.insert(0,before[place][0])
                    place=before[place][0]
                else:
                    f=False
                    print(steps)
                    break
            else:break
        if f:
            outroot=[]
            for k in root:
                outroot.append('('+str(k[0])+','+str(k[1])+')')
            print('-'.join(outroot))
        break
    else:
        places.append(placesnext)
        placesnow=placesnext
```

#### 07209:升级的迷宫寻宝 http://cs101.openjudge.cn/2024sp_routine/07209/

经典的bfs算法

```python
from collections import deque
x,y=map(int,input().split())
places=[0]*(x*y)
graph=[[] for _ in range(x*y)]
def search(a,b,x,y,graph):
    visited=[False]*(x*y)
    d=deque()
    before=[[] for _ in range(x*y)]
    d.append(a)
    while d:
        k=d.popleft()
        if k==b:
            road=[b]
            while k!=a:
                road.append(before[k][0])
                k=before[k][0]
            return road
        else:
            if not visited[k]:
                visited[k]=True
                for neighbor in graph[k]:
                    if not visited[neighbor]:
                        before[neighbor].append(k)
                        d.append(neighbor)
for x0 in range(x):
    line=input()
    for y0 in range(y):
        places[x0*y+y0]=line[y0]
can=['R','Y','C','0']
for x0 in range(x):
    for y0 in range(y):
        if places[x0*y+y0]=='R':
            rukou=x0*y+y0
        if places[x0*y+y0]=='Y':
            yaoshi=x0*y+y0
        if places[x0*y+y0]=='C':
            chukou=x0*y+y0
        if places[x0*y+y0] in can:
            if x0>0:
                if places[x0*y+y0-y] in can:
                    graph[x0*y+y0-y].append(x0*y+y0)
                    graph[x0*y+y0].append(x0*y+y0-y)
            if x0<x-1:
                if places[x0*y+y0+y] in can:
                    graph[x0*y+y0+y].append(x0*y+y0)
                    graph[x0*y+y0].append(x0*y+y0+y)
            if y0>0:
                if places[x0*y+y0-1] in can:
                    graph[x0*y+y0-1].append(x0*y+y0)
                    graph[x0*y+y0].append(x0*y+y0-1)
            if y0<y-1:
                if places[x0*y+y0+1] in can:
                    graph[x0*y+y0+1].append(x0*y+y0)
                    graph[x0*y+y0].append(x0*y+y0+1)
l1=search(rukou,yaoshi,x,y,graph)
l2=search(yaoshi,chukou,x,y,graph)
l2.pop()
for z in range(len(l1)):
    p=l1[len(l1)-1-z]
    print(str((p//y)+1)+' '+str((p%y)+1))
for z in range(len(l2)):
    p=l2[len(l2)-1-z]
    print(str((p//y)+1)+' '+str((p%y)+1))
```

#### 22485:升空的焰火，从侧面看 http://cs101.openjudge.cn/2024sp_routine/22485/

用bfs进行“层数遍历“

```python
from collections import deque
def bfs():
    queue = deque()
    queue.append(1)
    height[1]=1
    while queue:
        currentNode = queue.popleft()
        if nodes[currentNode-1][0]!=-1:
            queue.append(nodes[currentNode-1][0])
            out[height[currentNode]+1]=nodes[currentNode-1][0]
            height[nodes[currentNode-1][0]]=height[currentNode]+1
        if nodes[currentNode-1][1]!=-1:
            queue.append(nodes[currentNode-1][1])
            out[height[currentNode]+1]=nodes[currentNode-1][1]
            height[nodes[currentNode-1][1]]=height[currentNode]+1

nodes=[]
n=int(input())
out=[0]*(n+1)
height=[0]*(n+1)
for _ in range(n):
    node=list(map(int,input().split()))
    nodes.append(node)
bfs()
s='1'
for _ in out:
    if _:
        s+=' '
        s+=str(_)
print(s)
```

#### 28046:词梯 http://cs101.openjudge.cn/2024sp_routine/28046/

最原始与最经典的bfs算法，其他的bfs算法基本都是这一道题基础上演化而来

- 总时间限制: 

  1000ms

- 内存限制: 

  65536kB

- 描述

  词梯问题是由“爱丽丝漫游奇境”的作者 Lewis Carroll 在1878年所发明的单词游戏。从一个单词演变到另一个单词，其中的过程可以经过多个中间单词。要求是相邻两个单词之间差异只能是1个字母，如fool -> pool -> poll -> pole -> pale -> sale -> sage。与“最小编辑距离”问题的区别是，中间状态必须是单词。目标是找到最短的单词变换序列。假设有一个大的单词集合（或者全是大写单词，或者全是小写单词），集合中每个元素都是四个字母的单词。采用图来解决这个问题，如果两个单词的区别仅在于有一个不同的字母，就用一条边将它们相连。如果能创建这样一个图，那么其中的任意一条连接两个单词的路径就是词梯问题的一个解，我们要找最短路径的解。下图展示了一个小型图，可用于解决从 fool 到sage的词梯问题。注意，它是无向图，并且边没有权重。![img](http://media.openjudge.cn/images/upload/2596/1712744630.jpg)

- 输入

  输入第一行是个正整数 n，表示接下来有n个四字母的单词，每个单词一行。2 <= n <= 4000。 随后是 1 行，描述了一组要找词梯的起始单词和结束单词，空格隔开。

- 输出

  输出词梯对应的单词路径，空格隔开。如果不存在输出 NO。 如果有路径，保证有唯一解。

- 样例输入

  `25 bane bank bunk cane dale dunk foil fool kale lane male mane pale pole poll pool quip quit rain sage sale same tank vain wane fool sage`

- 样例输出

  `fool pool poll pole pale sale sage`

```python
import sys
from collections import deque

class Graph:
    def __init__(self):
        self.vertices = {}
        self.num_vertices = 0

    def add_vertex(self, key):
        self.num_vertices = self.num_vertices + 1
        new_vertex = Vertex(key)
        self.vertices[key] = new_vertex
        return new_vertex

    def get_vertex(self, n):
        if n in self.vertices:
            return self.vertices[n]
        else:
            return None

    def __len__(self):
        return self.num_vertices

    def __contains__(self, n):
        return n in self.vertices

    def add_edge(self, f, t, cost=0):
        if f not in self.vertices:
            nv = self.add_vertex(f)
        if t not in self.vertices:
            nv = self.add_vertex(t)
        self.vertices[f].add_neighbor(self.vertices[t], cost)

    def get_vertices(self):
        return list(self.vertices.keys())

    def __iter__(self):
        return iter(self.vertices.values())


class Vertex:
    def __init__(self, num):
        self.key = num
        self.connectedTo = {}
        self.color = 'white'
        self.distance = sys.maxsize
        self.previous = None
        self.disc = 0
        self.fin = 0

    def add_neighbor(self, nbr, weight=0):
        self.connectedTo[nbr] = weight

    # def __lt__(self,o):
    #     return self.id < o.id

    # def setDiscovery(self, dtime):
    #     self.disc = dtime
    #
    # def setFinish(self, ftime):
    #     self.fin = ftime
    #
    # def getFinish(self):
    #     return self.fin
    #
    # def getDiscovery(self):
    #     return self.disc

    def get_neighbors(self):
        return self.connectedTo.keys()

    # def getWeight(self, nbr):
    #     return self.connectedTo[nbr]

    # def __str__(self):
    #     return str(self.key) + ":color " + self.color + ":disc " + str(self.disc) + ":fin " + str(
    #         self.fin) + ":dist " + str(self.distance) + ":pred \n\t[" + str(self.previous) + "]\n"




def build_graph(filename):
    buckets = {}
    the_graph = Graph()
    with open(filename, "r", encoding="utf8") as file_in:
        all_words = file_in.readlines()
    # all_words = ["bane", "bank", "bunk", "cane", "dale", "dunk", "foil", "fool", "kale",
    #              "lane", "male", "mane", "pale", "pole", "poll", "pool", "quip",
    #              "quit", "rain", "sage", "sale", "same", "tank", "vain", "wane"
    #              ]

    # create buckets of words that differ by 1 letter
    for line in all_words:
        word = line.strip()
        for i, _ in enumerate(word):
            bucket = f"{word[:i]}_{word[i + 1:]}"
            buckets.setdefault(bucket, set()).add(word)

    # connect different words in the same bucket
    for similar_words in buckets.values():
        for word1 in similar_words:
            for word2 in similar_words - {word1}:
                the_graph.add_edge(word1, word2)

    return the_graph


#g = build_graph("words_small")
g = build_graph("vocabulary.txt")
print(len(g))


def bfs(start):
    start.distnce = 0
    start.previous = None
    vert_queue = deque()
    vert_queue.append(start)
    while len(vert_queue) > 0:
        current = vert_queue.popleft()  # 取队首作为当前顶点
        for neighbor in current.get_neighbors():   # 遍历当前顶点的邻接顶点
            if neighbor.color == "white":
                neighbor.color = "gray"
                neighbor.distance = current.distance + 1
                neighbor.previous = current
                vert_queue.append(neighbor)
        current.color = "black" # 当前顶点已经处理完毕，设黑色

"""
BFS 算法主体是两个循环的嵌套: while-for
    while 循环对图中每个顶点访问一次，所以是 O(|V|)；
    嵌套在 while 中的 for，由于每条边只有在其起始顶点u出队的时候才会被检查一次，
    而每个顶点最多出队1次，所以边最多被检查次，一共是 O(|E|)；
    综合起来 BFS 的时间复杂度为 0(V+|E|)

词梯问题还包括两个部分算法
    建立 BFS 树之后，回溯顶点到起始顶点的过程，最多为 O(|V|)
    创建单词关系图也需要时间，时间是 O(|V|+|E|) 的，因为每个顶点和边都只被处理一次
"""



#bfs(g.getVertex("fool"))

# 以FOOL为起点，进行广度优先搜索, 从FOOL到SAGE的最短路径,
# 并为每个顶点着色、赋距离和前驱。
bfs(g.get_vertex("FOOL"))


# 回溯路径
def traverse(starting_vertex):
    ans = []
    current = starting_vertex
    while (current.previous):
        ans.append(current.key)
        current = current.previous
    ans.append(current.key)

    return ans


# ans = traverse(g.get_vertex("sage"))
ans = traverse(g.get_vertex("SAGE")) # 从SAGE开始回溯，逆向打印路径，直到FOOL
print(*ans[::-1])
"""
3867
FOOL TOOL TOLL TALL SALL SALE SAGE
"""
```

### 3 图的dfs算法

深度优先搜索算法，更多地解决存在性问题

#### 28170:算鹰 http://cs101.openjudge.cn/2024sp_routine/28170/

经典的深度优先遍历算法

```python
def dfs(matrix, i, j, visited):
    rows = len(matrix)
    cols = len(matrix[0])
    if i < 0 or i >= rows or j < 0 or j >= cols or matrix[i][j] == 0 or visited[i][j]:
        return 0

    visited[i][j] = True
    dfs(matrix, i + 1, j, visited)
    dfs(matrix, i - 1, j, visited)
    dfs(matrix, i, j + 1, visited)
    dfs(matrix, i, j - 1, visited)
    return 1


def f(matrix):
    if not matrix or not matrix[0]:
        return 0

    rows = len(matrix)
    cols = len(matrix[0])
    visited = [[False] * cols for _ in range(rows)]
    num = 0

    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] == 1 and not visited[i][j]:
                num += dfs(matrix, i, j, visited)

    return num

l = []
for i in range(10):
    x = input()
    l1 = []
    for j in range(10):
        if x[j] == '-':
            l1.append(0)
        else:
            l1.append(1)
    l.append(l1)
print(f(l))
```



### 4 拓扑排序

对有向无环图进行排序，其中序列中只有靠前的点指向靠后的点的边

#### 04084 拓扑排序 http://cs101.openjudge.cn/2024sp_routine/04084/

描述

给出一个图的结构，输出其拓扑排序序列，要求在同等条件下，编号小的顶点在前。

输入

若干行整数，第一行有2个数，分别为顶点数v和弧数a，接下来有a行，每一行有2个数，分别是该条弧所关联的两个顶点编号。
v<=100, a<=500

输出

若干个空格隔开的顶点构成的序列(用小写字母)。

样例输入

```
6 8
1 2
1 3
1 4
3 2
3 5
4 5
6 4
6 5
```

样例输出

```
v1 v3 v2 v6 v4 v5
```

```python
from collections import  defaultdict
import heapq

v,a=map(int,input().split())
graph={}
for k in range(v):
    graph[k+1]=[]
for _ in range(a):
    s,t=map(int,input().split())
    if t not in graph[s]:
        graph[s].append(t)


def topological_sort(graph):
    indegree = defaultdict(int)
    result = []
    queue = []

    for u in graph:
        for v in graph[u]:
            indegree[v] += 1
    for u in graph:
        if indegree[u] == 0:
            heapq.heappush(queue,u)

    while queue:
        u = heapq.heappop(queue)
        result.append(u)

        for v in graph[u]:
            indegree[v] -= 1
            if indegree[v] == 0:
                heapq.heappush(queue,v)

    if len(result) == len(graph):
        return result
    else:
        return None
net=topological_sort(graph)
out=''
if net:
    for i in range(len(net)):
        if i<len(net)-1:
            out+='v'
            out+=str(net[i])
            out+=' '
        else:
            out+='v'
            out+=str(net[i])
    print(out)
else:print("No topological order exists due to a cycle in the graph.")
```

#### 09202:舰队、海域出击！http://cs101.openjudge.cn/2024sp_routine/09202/

利用拓扑排序判断是否有环，不能使用dfs：遍历顺序的原因会影响对环的判断

```python
from collections import deque, defaultdict


def topological_sort(graph):
    indegree = defaultdict(int)
    result = []
    queue = deque()

    for u in graph:
        for v in graph[u]:
            indegree[v] += 1

    for u in graph:
        if indegree[u] == 0:
            queue.append(u)


    while queue:
        u = queue.popleft()
        result.append(u)

        for v in graph[u]:
            indegree[v] -= 1
            if indegree[v] == 0:
                queue.append(v)

    if len(result) == len(graph):
        print('No')
    else:
        print('Yes')
T=int(input())
for _ in range(T):
    N,M=map(int,input().split())
    graph=defaultdict(list)
    for _ in range(M):
        x,y=map(int,input().split())
        graph[x].append(y)
    topological_sort(graph
```

### 5 dijkstra算法

解决有权图最短路径问题，采用贪心算法的策略，每次遍历到始点距离最近且未访问过的顶点的邻接节点，直到扩展到终点为止。

借助堆实现

#### 01724 ROADS http://cs101.openjudge.cn/2024sp_routine/01724/

```python
import heapq

def dijkstra(n, edges, s, d,t):
    graph = [[] for _ in range(n+1)]
    for u, v, w, x in edges:
        graph[u].append((v, w, x))

    pq = [(0, 0, s, 0)] 
    
    while pq:
        dist, mon, node, step= heapq.heappop(pq)
        if node == d and mon<=t:
            return dist
        for neighbor, distance, money  in graph[node]:
            new_dist = dist + distance
            new_money= mon + money

            if new_money<=t and step+1 <=n:
                heapq.heappush(pq, (new_dist, new_money, neighbor,step+1))
    return -1

t=int(input())
n=int(input())
m=int(input())
edges = [list(map(int, input().split())) for _ in range(m)]

result = dijkstra(n, edges, 1, n, t)
print(result)
```

## 总结

对于机考，其考察的更多是数据结构与算法的实践部分。算法是建立在数据结构基础之上，用于处理有一定结构的数据的机制，这一个学期的数据结构与算法的学习过程中，我们学习了如下几类：

1.基础线性数据结构：列表

基于此的算法：

1）排序算法

2）二分查找法

3）动态规划

2.栈

基于栈的入栈和出栈的特殊机制，即栈数据结构自带的算法属性，我们可以解决

1）回溯问题

2）括号匹配问题

3）前序、中序、后序表达式问题

3.队列

双端队列一般作为后续图算法的数据结构基础

4.堆：一种有序且快速提取最大值与最小值的数据结构，基于堆排序算法

5.树&二叉树

树的本质是有向无环图，因此其节点优先级关系可以通过类对象进行刻画，并有上下次序地访问

基于此的算法有

1）树的遍历

不仅如此，还有其他数据结构

1）并查集：基于此的算法为路径缩并

2）huffman树、二叉搜索树、avl树：一种更特殊的树结构

6.图

无向图的节点之间不存在次序关系，所以可以使用字典进行表示

基于此的算法有

1）bfs算法：广度优先搜索

2）dfs算法：深度优先搜索

3）拓扑排序：找生成树的过程

4）dijkstra算法：找最小路径的算法



算法的目的是解决问题，数据结构的意义是作为算法的基础的同时便于刻画我们所处的场景。



这份机考总结，记录了一个普通的学员学习的过程，同时也是她整个对编程的学习的总结。

虽然代码处处尽显青涩与笨拙，但每一道题都是认真钻研后的结果，尽显从思路模糊到略知一二的过程。