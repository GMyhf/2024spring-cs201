# cheat sheet

杨博文 数学科学学院

## 一、几个模版

### 最小生成树

```python
#最小生成树prim,输入用mat存编号为0~n-1的邻接表，表内元素为(x,d)x为下一个点d为权值，输出最小总权值
import heapq
def solve(mat):
    h=[(0,0)]
    n=len(mat)
    vis=[1 for i in range(n)]
    ans=0
    while h:
        (d,x)=heapq.heappop(h)
        if vis[x]:
            vis[x]=0
            ans+=d
            for (y,t) in mat[x]:
                if vis[y]:
	                heapq.heappush(h,(t,y))
    return ans
```

```python
#最小生成树kruskal,输入同上
import heapq
p=[i for i in range(n)]
def find(x):
    if p[x]==x:
        return x
    p[x]=find(p[x])
    return p[x]
def union(x,y):
    u,v=find(x),find(y)
    p[u]=v
def solve(mat):
    h=[]
    ans=0
    for i in range(n):
        for (j,d) in mat[i]:
            heapq.heappush(h,(d,i,j))
    while h:
        (d,i,j)=heapq.heappop(h)
        if find(i)!=find(j):
            union(i,j)
            ans+=d
	return ans
```



### 最短路径

```python
#dijkstra，输入同上,求出from和to之间的最短路径，需要保证输入无负权值，输出-1表示无法到达to
#若输入to则处理点对点问题，否则返回所有点的最短距离
import heapq
def solve(mat,f,to=-1):
    h=[(0,f)]
    n=len(mat)
    vis=[-1 for i in range(n)]
    while h:
        (d,x)=heapq.heappop(h)
        if x==to:
            return d
        if vis[x]==-1:
            vis[x]=d
	        for (y,s) in mat[x]:
    	        if vis[y]==-1:
                    heapq.heappush(h,(d+s,y))
    return vis
```



### 拓扑排序

```python
#输入为mat存邻接表，mat[i]为i指向的点的列表，输出-1表示有环，顺便做了输出字典序最小排序方式
import heapq
def solve(mat):
    n=len(mat)
    h=[]
    ru=[len(mat[i]) for i in range(n)]
    ans=[]
    for i in range(n):
        if ru[i]==0:
            heapq.heappush(h,i)
    while h:
        x=heapq.heappop(h)
        ans.append(x)
        for y in mat[x]:
            ru[y]-=1
            if ru[y]==0:
                heapq.heappush(h,y)
   	if len(ans)<n:
        return -1
    else:
        return ans
   	
```



### 判断无向图是否连通、有无回路

```python
#输入同上，基于并查集和图论
p=[i for i in range(n)]
e=[0 for i in range(n)]
v=[1 for i in range(n)]
def find(x):
    if p[x]==x:
        return x
    p[x]=find(p[x])
    return p[x]
def union(x,y):
    s,t=find(x),find(y)
    if s==t:
        e[t]+=1
        return
    p[s]=t
    v[t]+=v[s]
    e[t]+=e[s]+1
def connect(mat):
    for i in range(n):
        for j in mat[i]:
            union(i,j)
    root=find(0)
    if v[root]==n:
        return True
    return False
def loop(mat):
    for i in range(n):
        for j in mat[i]:
            union(i,j)
    for i in range(n):
        r=find(i)
        if e[r]>=v[r]:
            return True
    return False
```



### 单调栈

```python
#给定一个列表，输出每个元素之前小于它的最后一个元素的下标
def solve(lis):
    n=len(lis)
    stack=[]
    ans=[]
    for i in range(n):
        x=lis[i]
        while stack:
            (y,j)=stack[-1]
            if y>=x:
                stack.pop()
                continue
            break
        if not stack:
            stack.append((x,i))
            ans.append(-1)
        else:
            ans.append(stack[-1][1])
            stack.append((x,i))
    return ans
```

## 强连通图（Kosaraju/2 DFS）

```python
def dfs1(graph, node, visited, stack):
    visited[node] = True
    for neighbor in graph[node]:
        if not visited[neighbor]:
            dfs1(graph, neighbor, visited, stack)
    stack.append(node)

def dfs2(graph, node, visited, component):
    visited[node] = True
    component.append(node)
    for neighbor in graph[node]:
        if not visited[neighbor]:
            dfs2(graph, neighbor, visited, component)

def kosaraju(graph):
   # Step 1: Perform first DFS to get finishing times
   stack = []
   visited = [False] * len(graph)
   for node in range(len(graph)):
       if not visited[node]:
           dfs1(graph, node, visited, stack)
   
    # Step 2: Transpose the graph
    transposed_graph = [[] for _ in range(len(graph))]
    for node in range(len(graph)):
        for neighbor in graph[node]:
            transposed_graph[neighbor].append(node)
    
    # Step 3: Perform second DFS on the transposed graph to find SCCs
    visited = [False] * len(graph)
    sccs = []
    while stack:
        node = stack.pop()
        if not visited[node]:
            scc = []
            dfs2(transposed_graph, node, visited, scc)
            sccs.append(scc)
    return sccs
```

# 

## 二、语法

### math库

向上取整：math.ceil()

向下取整：math.floor()

阶乘：math.factoria()

数学常数：math.pi（圆周率），math.e（自然对数的底）

math.sqrt(x), math.pow(x,y), math.exp(x), math.log(真数，底数)（默认为自然对数）

math.sin(),math.cos(),math.tan()

math.asin(),math.acos(),math.atan()

### heapq库：实现堆

heapq.heapify(list)

heapq.heappush(堆名，被插元素)  heapq.heappop(堆名)

插入元素的同时弹出顶部元素：heapq.heappushpop(堆名，被插元素)

  （或heapq.heapreplace(堆名，被插元素)）

·以上操作在最大堆中应换为“_X_max”（X是它们中的任意一个）

### itertools库：

整数集：itertools.count(x,y)（从x开始往大数的整数，间隔为y）

循环地复制一组变量：itertools.cycle(list)

所有排列：itertools.permutations(集合，选取个数)

所有组合：itertools.combinations

已排序列表去重：[i for i,_ in itertools.groupby(list)]（每种元素只能保留一个）

​      或者list(group)[:n]（group被定义为分组，保留每组的n个元素）

### collections库：

双端队列：

​	创建：a=deque(list)

​	从末尾添加元素：a.append(x)

​	从开头添加元素：a.appendleft(x)

​	从末尾删除元素：b=a.pop()

​	从开头删除元素：b=a.popleft()

有序字典：Ordereddict()

默认值字典：a=defaultdict(默认值)，如果键不在字典中，会自动添加值为默认值的键值对，而不报KeyError。

计数器：Counter(str)，返回以字符种类为键，出现个数为值的字典

### sys库：

sys.exit()用于及时退出程序

sys.setrecursionlimit()用于调整递归限制（递归层数过多会引起MLE）

### statistics库： 

1.mean(data)：计算数据的平均值（均值）。

2.harmonic_mean(data)：计算数据的调和平均数。

3.median(data)：计算数据的中位数。

4.median_low(data)：计算数据的低中位数。

5.median_high(data)：计算数据的高中位数。

6.median_grouped(data, interval=1)：计算分组数据的估计中位数。

7.mode(data)：计算数据的众数。

8.pstdev(data)：计算数据的总体标准差。

9.pvariance(data)：计算数据的总体方差。

10.stdev(data)：计算数据的样本标准差。

11.variance(data)：计算数据的样本方差。

### 数据处理：

二进制：bin()，八进制：oct()，十六进制：hex()

保留n位小数：round(原数字，保留位数)；’%.nf’%原数字；’{:.nf}’.format(原数字)；n位有效数字：’%.ng’%原数字；’{:.ng}’.format(原数字)

**ASCII转字符：chr();字符转ASCII：ord()**

**判断数据类型：isinstance(object,class)**

**try-except 某error**

1.`split()`使用时注意可能存在连续多个空格的情况

2.`RuntimeError`考虑使用`sys.setrecrusionlimit(layor)`避免爆栈

3.==不要默认最大值初始是0，有可能所有数据都小于零，最后输出0导致WA==

4.注意浮点数精度问题

5.候选人追踪$k=314159$等特殊情况==（边界情况）==

6.矩阵行数列数区分好

7.==多组数据不能使用exit()==

8.==不能对空列表进行某些操作，如`min`，`max`等==

9.考虑统一输出节省时间

10.多组测试数据中途`break`考虑忽略后续输入是否会引起下一组测试数据输入起点不对齐

11.注意输入数据可能的重复性

12.二分查找上下界设定要小心

13.第一行加`# pylint: skip-fle`可以忽略检查

14.标准答案一般不多于50行

### 快速读写

```python
import sys
input = lambda: sys.stdin.readline().strip()
write = lambda x: sys.stdout.write(str(x))
```

### 位运算

![image-20240604203335872](https://raw.githubusercontent.com/xjsongphy/repository_for_typora/main/img/202406042033958.png)

#### 与运算的用途

##### 清零

如果想将一个单元清零，即使其全部二进制位为0，只要与一个各位都为零的数值相与，结果为零。

##### 取一个数的指定位

比如取数 `X=1010 1110` 的低4位，只需要另找一个数Y，令Y的低4位为1，其余位为0，即`Y=0000 1111`，然后将X与Y进行按位与运算`X&Y=0000 1110`即可得到X的指定位。

##### 判断奇偶

只要根据最未位是0还是1来决定，为0就是偶数，为1就是奇数。因此可以用`if ((a & 1) == 0)`来判断a是不是偶数。

### 小技巧

1.进制转换：2进制`bin()`输出0b...，八进制`oct()`输出0o...，十六进制`hex()`输出0x...，`int(str,base)`可以把字符串按照指定进制转为十进制默认base=10

2.format：

```python
print('{:.2f}'.format(num))
```

3.字符串匹配等

```python
iterable.count(value)
str.find(sub)		#未找到抛出-1
list.index(x)		#未找到抛出ValueError
```

4.使用`try`+`except`判断错误类型，辅助处理RE问题

5.math库

```python
math.pow(x, y) == x**y
math.factorial(n) == n!
```

6.`ord()`把字符变为ASCII，`chr()`把ASCII变为字符

7.`calendar.isleap(year)`返回T/F判断闰年

8.`for a1, a2, ..., am in zip(b1, b2, ..., bn)`旋转矩阵

9.排列组合

```python
from itertools import permutations,combinations
permutations(list)			#生成list的全排列（每个以元组形式存在）
combinations(list,k)		#生成list的k元组合（无序）（每个以元组形式存在）
```

10.`replace()`替换字符串中的指定字符，`eval()`函数计算表达式的值

11.`print(*r)`快速输出空格连接的列表/元组等

12.取整函数：`ceil`向上取整（math库）,`floor`向下取整（math库）,`round(num, n)`四舍五入, 小数点后最终有n位

13.`enumerate`快速获取索引和值：`for index, value in enumerate(list, start)`

14.补齐位数：`str.ljust(width, string)`右侧补充`string`至`str`长度为`width`，`str.rjust(width, string)`左侧补充`string`至`str`长度为`width`

## 三、令人印象深刻的题目

### 02299: Ultra-QuickSort

http://cs101.openjudge.cn/practice/02299/



思路：

用归并排序计算逆序数

代码

```python
# 
def solve(lis):
    if len(lis)==1:
        #print(lis)
        return (lis,0)
    k=len(lis)//2
    l=len(lis)-k
    le=lis[:k]
    ri=lis[k:]
    u=solve(le)
    v=solve(ri)
    p=u[0];q=v[0]
    i=0;j=0
    ans=0
    num=[]
    while i<k or j<l:
        if j<l:
            if i<k and p[i]<=q[j]:
                num.append(p[i])
                ans+=j
                i+=1
            else:
                num.append(q[j])
                j+=1
        else:
            ans+=j
            num.append(p[i])
            i+=1
    #print(num)
    return (num,u[1]+v[1]+ans)
while True:
    n=int(input())
    if n==0:
        break
    lis=[]
    d=0
    for _ in range(n):
        temp=int(input())
        lis.append(temp)
    print(solve(lis)[1])
```

### 24591: 中序表达式转后序表达式

http://cs101.openjudge.cn/practice/24591/



思路：

先处理输入，然后依次遍历输入列表，如果是数，直接输出；如果是左括号，先入栈；如果是运算符，就把此前能确定先运算的运算符出栈输出，即输出所有比目前元素高级的运算符直到上一个括号或栈底；如果是右括号，就输出至上一个左括号

代码

```python
# 
n=int(input())
for _ in range(n):
    s=input()
    lis=[];temp=''
    for x in s:
        if x in '+-*/()':
            if temp!='':
                lis.append(temp)
                temp=''
            lis.append(x)
        else:
            temp+=x
    if temp!='':
        lis.append(temp)
    os=[]
    for x in lis:
        if x=='(':
            os.append(x)
        if x not in '+-*/()':
            print(x,end=' ')
        if x in '+-*/':
            if os:
                y=os[-1]
                if x in '+-':
                    while os and os[-1] not in '()':
                        y=os.pop()
                        print(y,end=' ')
                else:
                    while os and os[-1] not in '+-()':
                        y=os.pop()
                        print(y,end=' ')
            os.append(x)
        if x==')':
            y=os.pop()
            while y!='(':
                print(y,end=' ')
                y=os.pop()
    while os:
        print(os.pop(),end=' ')
    print()
```

### 晴问9.5: 平衡二叉树的建立

https://sunnywhy.com/sfbj/9/5/359



思路：

照猫画虎，好难

代码

```python
# 
class node:
    def __init__(self,key):
        self.key=key
        self.p=None
        self.l=None
        self.r=None
        self.h=0
def hi(x):
    if x==None:
        return -1
    else:
        return x.h
def lr(rt):
    global root
    x=rt.r;y=x.l;z=rt.p
    x.p=z
    if z:
        if rt==z.l:
            z.l=x
        else:
            z.r=x
    rt.p=x
    x.l=rt
    rt.r=y
    if y:
        y.p=rt
    rt.h=1+max(hi(rt.l),hi(rt.r))
    x.h=1+max(hi(x.l),hi(x.r))
    if z:
        z.h=1+max(hi(z.l),hi(z.r))
    y=rt
    while y.p:
        y=y.p
    root=y
def rr(rt):
    global root
    x=rt.l;y=x.r;z=rt.p
    x.p=z
    if z:
        if rt==z.l:
            z.l=x
        else:
            z.r=x
    rt.p=x
    x.r=rt
    rt.l=y
    if y:
        y.p=rt
    rt.h=1+max(hi(rt.l),hi(rt.r))
    x.h=1+max(hi(x.l),hi(x.r))
    if z:
        z.h=1+max(hi(z.l),hi(z.r))
    y=rt
    while y.p:
        y=y.p
    root=y
n=int(input())
lis=list(map(int,input().split()))
root=node(lis[0])
for i in lis[1:]:
    x=node(i)
    y=root;z=root
    while y:
        z=y
        if y.key<=i:
            y=y.r
        else:
            y=y.l
    if z.key<=i:
        z.r=x
    else:
        z.l=x
    x.p=z
    t=x
    while t:
        t.h=1+max(hi(t.l),hi(t.r))
        t=t.p
    t=x
    while True:
        if not t:
            break
        t.h=1+max(hi(t.l),hi(t.r))
        if abs(hi(t.l)-hi(t.r))<=1:
            t=t.p
        else:
            if hi(t.l)>hi(t.r):
                x=t.l
                y=x.l;z=x.r
                if hi(y)<hi(z):
                    lr(x)
                rr(t)
            else:
                x=t.r
                y=x.r;z=x.l
                if hi(y)<hi(z):
                    rr(x)
                lr(t)
            t=t.p
ans=[]
def solve(rt):
    if not rt:
        return
    ans.append(rt.key)
    solve(rt.l)
    solve(rt.r)
solve(root)
print(' '.join(map(str,ans)))
```

### 04089: 电话号码

trie, http://cs101.openjudge.cn/practice/04089/

Trie 数据结构可能需要自学下。



思路：

用字典嵌套表示trie在其中进行插入和查询即可

代码

```python
# 
for ____ in range(int(input())):
    trie={}
    n=int(input())
    lis=[]
    for _ in range(n):
        lis.append(input())
    lis.sort(key=len,reverse=True)
    flag=1
    for w in lis:
        temp=trie;l=len(w);f=1
        for x in w:
            if x in temp:
                temp=temp[x]
            else:
                f=0
                temp[x]={}
                temp=temp[x]
        if f:
            flag=0
    if flag:
        print("YES")
    else:
        print("NO")
```

### 03441: 4 Values whose Sum is 0

data structure/binary search, http://cs101.openjudge.cn/practice/03441



思路：

用一个字典存储AB相加后每个和的个数，再对CD遍历计算

代码

```python
# 
A=[];B=[];C=[];D=[]
n=int(input())
for _ in range(n):
    a,b,c,d=map(int,input().split())
    A.append(a);B.append(b);C.append(c);D.append(d)
dic={}
for i in range(n):
    for j in range(n):
        x=A[i]+B[j]
        if x in dic:
            dic[x]+=1
        else:
            dic[x]=1
ans=0
for i in range(n):
    for j in range(n):
        x=-(C[i]+D[j])
        if x in dic:
            ans+=dic[x]
print(ans)
```



### 22067: 快速堆猪

http://cs101.openjudge.cn/practice/22067/



思路：

用两个栈分别存猪和这个猪进去之后的最小值

代码

```python
# 
pigstack=[]
minstack=[]
mi=-1
while True:
    try:
        s=input().split()
        if s[0]=='push':
            n=int(s[1])
            pigstack.append(n)
            if mi==-1:
                minstack.append(n)
                mi=n
            elif mi>=n:
                minstack.append(n)
                mi=n
        if s[0]=='pop':
            if not pigstack:
                continue
            temp=pigstack.pop()
            if temp==mi:
                minstack.pop()
                if minstack:
                    mi=minstack[-1]
                else:
                    mi=-1
        if s[0]=='min':
            if mi>=0:
                print(mi)
    except EOFError:
        break
```



### 27947: 动态中位数

http://cs101.openjudge.cn/practice/27947/



思路：

用两个栈维护左半边与右半边

代码

```python
# 
import heapq

for _____ in range(int(input())):
    lis=list(map(int,input().split()))
    n=len(lis)
    if True:
        print((n+1)//2)
        h1=[-lis[0]];h2=[lis[0]]
        print(h2[0],end=' ')
        for i in range(1,1+(n-1)//2):
            x,y=lis[2*i-1],lis[2*i]
            if x>y:
                x,y=y,x
            mid=h2[0]
            if x<=mid and y>=mid:
                heapq.heappush(h1,-x)
                heapq.heappush(h2,y)
            if x>mid:
                heapq.heappop(h2)
                heapq.heappush(h2,x)
                heapq.heappush(h2,y)
                heapq.heappush(h1,-h2[0])
                mid=h2[0]
            if y<mid:
                heapq.heappop(h1)
                heapq.heappush(h1,-x)
                heapq.heappush(h1,-y)
                heapq.heappush(h2,-h1[0])
                mid=h2[0]
            print(mid,end=' ')
        print()
```



### 04135: 月度开销

http://cs101.openjudge.cn/practice/04135/



思路：

注意到题目数据数据总和上界不大，采用对目标值二分查找

代码

```python
# 
def check(lis,k,m):
    c=0;i=0;temp=0
    for x in lis:
        if x>k:
            return False
        if temp+x<=k:
            temp+=x
        else:
            c+=1
            temp=x
        if c>=m:
            return False
    return True
n,m=map(int,input().split())
lis=[int(input()) for i in range(n)]
l=sum(lis)//m+1;r=sum(lis)
while l<r:
    k=(l+r)//2
    if check(lis,k,m):
        r=k
    else:
        l=k+1
print(r)
```



### 01182: 食物链

http://cs101.openjudge.cn/practice/01182/



思路：

把一个点、该点所吃的、被该点吃的共3n个点做成并查集

代码

```python
# 

def find(x):
    if p[x]==x:
        return x
    p[x]=find(p[x])
    return p[x]
def union(x,y):
    u,v=find(x),find(y)
    p[u]=v
n,k=map(int,input().split())
p=[i for i in range(3*n+1)]
ans=0
for _ in range(k):
    d,x,y=map(int,input().split())
    if x>n or y>n:
        ans+=1
        continue
    if d==1:
        if find(x)==find(y+n) or find(x+n)==find(y):
            ans+=1
            continue
        union(x,y)
        union(x+n,y+n)
        union(x+2*n,y+2*n)
    if d==2:
        if find(x)==find(y) or find(x)==find(y+n):
            ans+=1
            continue
        union(x+n,y)
        union(x,y+2*n)
        union(x+2*n,y+n)
print(ans)
```



### 28046: 词梯

bfs, http://cs101.openjudge.cn/practice/28046/



思路：

先利用字典存每种可能的词桶，便于建图，bfs

代码

```python
# 
class word:
    def __init__(self,k):
        self.k=k
        self.n=[]
n=int(input())
dic1={};dic2={};dic={}
for ___ in range(n):
    w=input()
    for i in range(0,4):
        ww=w[:i]+'_'+w[i+1:]
        if w not in dic1:
            dic1[w]=[ww]
        else:
            dic1[w].append(ww)
        if ww not in dic2:
            dic2[ww]=[w]
        else:
            dic2[ww].append(w)
for w in dic1:
    dic[w]=word(w)
for w in dic1:
    x=dic[w]
    for ww in dic1[w]:
        for v in dic2[ww]:
            if v!=w:
                x.n.append(dic[v])
    #print(list(map(lambda x:x.k,dic[w].n)))
from collections import deque
f,t=map(lambda x:dic[x],input().split())
q=deque([f]);p={f:None}
while q:
    x=q.popleft()
    for y in x.n:
        if y not in p:
            q.append(y)
            p[y]=x
if t not in p:
    print("NO")
else:
    ans=[]
    while t:
        ans.append(t.k)
        t=p[t]
    print(" ".join(reversed(ans)))
```

### 28050: 骑士周游

dfs, http://cs101.openjudge.cn/practice/28050/



思路：

神奇的优化：dfs的时候走下一步可行路径最少的，这样可以避免一头扎进节点海里出不来

代码

```python
# 
n=int(input())
x,y=map(int,input().split())
vis=[[0 for i in range(n)] for j in range(n)]
vis[x][y]=1
def check(x,y,n):
    if x<0 or x>n-1 or y<0 or y>n-1:
        return False
    return True
def nei(i,j):
    ans=0
    for k in range(8):
        ii=i+dx[k];jj=j+dy[k]
        if check(ii,jj,n) and vis[ii][jj]==0:
            ans+=1
    return ans
dx=[-2,-1,1,2,2,1,-1,-2]
dy=[1,2,2,1,-1,-2,-2,-1]
def dfs(x,y,n,c):
    if c==n*n:
        return 1
    vis[x][y]=1
    lis=[(x+dx[i],y+dy[i]) for i in range(8)]
    lis.sort(key=lambda x:nei(x[0],x[1]))
    for (xx,yy) in lis:
        if check(xx,yy,n) and vis[xx][yy]==0:
            if dfs(xx,yy,n,c+1):
                return True
    vis[x][y]=0
    return False
if dfs(x,y,n,1):
    print("success")
else:
    print("fail")
```
