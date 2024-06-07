# 2024春季数算B5班机考

Updated 2117 GMT+8 Jun 7, 2024

2024 spring, Complied by Hongfei Yan



数算xzm班

http://xzmdsa.openjudge.cn/2024final/



| 题目                         | tags                 |
| ---------------------------- | -------------------- |
| 坏掉的键盘                   | string               |
| 餐厅订单页面设计             | sortings             |
| 抓住那头牛                   | bfs                  |
| 正方形                       | math, implementation |
| Taki的乐队梦想               | 拓扑排序变形         |
| 斗地主大师                   | 二分查找dfs          |
| 小明的刷题计划               | 二分查找             |
| 银河贸易问题                 | bfs                  |
| Taki的乐队梦想（数据强化版） | 拓扑排序变形         |





## 27378: 坏掉的键盘

http://cs101.openjudge.cn/practice/27378/

小张在机房找到一个坏掉的键盘，26个字母只有一个键坏掉了，当遇到这个字母时，小张会用'.'（英文句号）替代。请你将小张的文字还原。

**输入**

首先输入一个字符c，表示坏掉的键。

第二行有一串只包括剩余25个字母，空格以及英文句点的字符串，请你帮忙还原

**输出**

还原后的字符串

样例输入

```
i
look . f.nd a d.rty keyboard.
```

样例输出

```
look i find a dirty keyboardi
```





```python
t = input()
print(input().replace('.', t))
```







## 28404: 餐厅订单页面设计

http://xzmdsa.openjudge.cn/2024final/2/

你是餐厅订单系统的设计师，你需要设计一个页面展示客户的点菜订单。

给你一个数组orders，表示客户在餐厅中完成的订单（订单编号记为i），orders[i]=[customerName[i], tableNumber[i], foodItem[i]]，其中customerName[i]是客户的姓名，tableNumber[i]是客户所在餐桌的桌号，而foodItem[i]是客户点的餐品名称。

请你设计该餐厅的订单页面，用一张表表示。在这张表中，第一行为标题，其第一列为餐桌桌号"Table",后面每一列都是按字母顺序排列的餐品名称。接下来的每一行中的项则表示每张餐桌订购的相应餐品数量，第一列应当填对应的桌号，后面依次填写下单的餐品数量。

注意：客户姓名不是点菜展示表的一部分。此外，表中的数据行应按照餐桌桌号升序排列。

**输入**

第1行是orders的个数N
第1至N+1行是多行orders，每行orders由三个元素组成：customerName[i], tableNumber[i], foodItem[i]，其中customerName[i]是客户的姓名，tableNumber[i]是客户所在餐桌的桌号，而foodItem[i]是客户点的餐品名称。每个元素由','分隔。

**输出**

一张订单展示表。在这张表中，第一行为标题，其第一列为餐桌桌号"Table",后面每一列都是按字母顺序排列的餐品名称。
接下来的每一行中的项则表示每张餐桌订购的相应餐品数量，第一列应当填对应的桌号，后面依次填写下单的餐品数量。
注意:为了订单展示表的美观，每行数据有空行分隔，每行数据中的每个元素由制表符分割('\t')

样例输入

```
6
David,3,Ceviche
Corina,10,Beef-Burrito
David,3,Fried-Chicken
Carla,5,Water
Carla,5,Ceviche
Rous,3,Ceviche
```

样例输出

```
Table   Beef-Burrito    Ceviche Fried-Chicken   Water

3       0       2       1       0

5       0       1       0       1

10      1       0       0       0
```

提示

1.1<=orders.length<=5*10^4;
2.orders[i].length==3;
3.1<=customerName[i].length,foodItem[i].length<=20;
4.customerName[i]和foodItem[i]有大小写英文字母，连字符'-'及空格' '组成；
5.tableNumber[i]是1到500范围内的整数。





```python
# 23物院 宋昕杰
tables = {}
dishes = {}
for _ in range(int(input())):
    name, table, dish = input().split(',')
    table = int(table)
    if table not in tables:
        tables[table] = True
    if dish not in dishes:
        dishes[dish] = {}
    if table not in dishes[dish]:
        dishes[dish][table] = 0
    dishes[dish][table] += 1

keys = sorted(list(dishes.keys()))
tables = sorted(list(tables.keys()))
print('\t'.join(['Table'] + keys))
print()
for table in tables:
    ls = [str(table)]
    for dish in keys:
        if table not in dishes[dish]:
            ls.append('0')
        else:
            ls.append(str(dishes[dish][table]))
    print('\t'.join(ls))
    print()
```



## 04001: 抓住那头牛

http://cs101.openjudge.cn/practice/04001/

农夫知道一头牛的位置，想要抓住它。农夫和牛都位于数轴上，农夫起始位于点N(0<=N<=100000)，牛位于点K(0<=K<=100000)。农夫有两种移动方式：

1、从X移动到X-1或X+1，每次移动花费一分钟

2、从X移动到2*X，每次移动花费一分钟

假设牛没有意识到农夫的行动，站在原地不动。农夫最少要花多少时间才能抓住牛？



**输入**

两个整数，N和K

**输出**

一个整数，农夫抓到牛所要花费的最小分钟数

样例输入

```
5 17
```

样例输出

```
4
```



bfs+剪枝

```python
# 23物院 宋昕杰
from queue import Queue

n, k = map(int, input().split())
q = Queue()
vis = {}

q.put((n, 0))
while True:
    x, t = q.get()
    if x == k:
        print(t)
        exit()

    if x in vis:
        continue
    vis[x] = True

    t += 1
    if x < k:
        q.put((2*x, t))
    q.put((x + 1, t))
    if x > 0:
        q.put((x - 1, t))
```



## 02002: 正方形

http://cs101.openjudge.cn/practice/02002/

给定直角坐标系中的若干整点，请寻找可以由这些点组成的正方形，并统计它们的个数。

**输入**

包括多组数据，每组数据的第一行是整点的个数n(1<=n<=1000)，其后n行每行由两个整数组成，表示一个点的x、y坐标。输入保证一组数据中不会出现相同的点，且坐标的绝对值小于等于20000。输入以一组n=0的数据结尾。

**输出**

对于每组输入数据，输出一个数，表示这组数据中的点可以组成的正方形的数量。

样例输入

```
4
1 0
0 1
1 1
0 0
9
0 0
1 0
2 0
0 2
1 2
2 2
0 1
1 1
2 1
4
-2 5
3 7
0 0
5 2
0
```

样例输出

```
1
6
1
```



For each pair of points, it checks the two possible orientations of squares (clockwise and counterclockwise) by calculating the required third and fourth points.

```python
def count_squares(points):
    point_set = set(points)
    count = 0

    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            p1 = points[i]
            p2 = points[j]
            
            if p1 == p2:
                continue
            
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            
            p3 = (p1[0] + dy, p1[1] - dx)
            p4 = (p2[0] + dy, p2[1] - dx)
            
            if p3 in point_set and p4 in point_set:
                count += 1
            
            p3 = (p1[0] - dy, p1[1] + dx)
            p4 = (p2[0] - dy, p2[1] + dx)
            
            if p3 in point_set and p4 in point_set:
                count += 1

    return count // 4

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    while True:
        n = int(data[index])
        index += 1
        if n == 0:
            break
        
        points = []
        for _ in range(n):
            x = int(data[index])
            y = int(data[index + 1])
            points.append((x, y))
            index += 2
        
        print(count_squares(points))

if __name__ == "__main__":
    main()
```

### Explanation
1. **Function `count_squares(points)`**:
   - It takes a list of points and returns the number of squares that can be formed by these points.
   - It uses a set to quickly check if a point exists.
   - For each pair of points, it checks the two possible orientations of squares (clockwise and counterclockwise) by calculating the required third and fourth points.
   - It counts these squares and divides by 4 at the end because each square is counted four times (once for each pair of its points).
2. **Function `main()`**:
   - Reads input from standard input (usually used in competitive programming).
   - Processes multiple sets of data until a set with `n = 0` is encountered.
   - For each set, it collects the points and uses `count_squares` to find the number of squares.
   - Prints the result for each set.





```python
# 23物院 宋昕杰
import sys

input = lambda : sys.stdin.readline().strip()

while n := int(input()):

    ls = []
    dic = {}
    vis = {}

    for _ in range(n):
        x, y = map(int, input().split())
        ls.append((x, y))
        dic[(x, y)] = True

    ls.sort()
    cnt = 0
    for i in range(n - 1):
        x1, y1 = ls[i]
        for j in range(i + 1, n):
            x2, y2 = ls[j]
            dx, dy = x2 - x1, y2 - y1

            x3, y3 = x1 - dy, y1 + dx
            x4, y4 = x2 - dy, y2 + dx

            key = tuple(sorted([(x1, y1), (x2, y2), (x3, y3), (x4, y4)]))
            if (x3, y3) in dic and (x4, y4) in dic and key not in vis:
                cnt += 1
                vis[key] = True
    print(cnt)
```



## 28413: Taki的乐队梦想

http://xzmdsa.openjudge.cn/2024final/5/

Taki有一个组乐队的梦想。她和她的伙伴们一起组建了一支乐队，但是这个过程并不是很顺利……

好不容易办完了第一次演出，乐队有了雏形，但是很快打击接踵而至，成员们又各自分离。

为了挽救乐队，Taki试图有所行动。她开始逐个联系乐队成员，希望她们能够回来组乐队。

她的队友们具有一些怪异的个性：每个乐队成员都有一个单方面对之“相爱相杀”的队友清单。也有的队员性格平和，那么她的清单就是空的。
如果“相爱相杀”的队友都已归队，Taki一联系这名队员，她就会马上归队；但遗憾的是，她归队的同时又会把“相爱相杀”的队友都气走。

幸好，在不愿透露姓名的工作人员S的帮助下，Taki对乐队成员进行了一次排序编号。保证每个队员对之“相爱相杀”的对象编号都严格小于队员自己。在这种情况下，无疑是可以让所有乐队成员都归队的。

请告诉Taki，为了让所有乐队成员归队，她最少需要联系多少次乐队成员，并且给出字典序最小的一种联系方案。

（这里的字典序是指乐队成员排序的顺序，而非名字顺序或者别的什么顺序）

**输入**

本题有多组数据。第一行是一个正整数T，表示数据组数。
每组数据的第一行是一个正整数n，表示乐队成员数量。
接下来n行，每行表示一名乐队成员的信息：开头是一个字符串，表示乐队成员的姓名（由字母和数字组成，没有空格），后面跟着零个或者若干个数字，表示她的相关人员的序号（序号从1开始计数）

数据保证一定存在让所有人都归队的联系方式。

**输出**

对于每组数据，第一行输出一个数字m，表示最小联系次数。
第二行输出m个姓名，中间用空格分割，表示字典序最小的联系方案下，每一次联系的人的姓名。

样例输入

```
3
3
A
B 1
C 2 1
4
A
B 1
C 2
D 3
5
Takamatsu
Kaname
Shiina 2
Chihaya 1
Nagasaki 1 4
```

样例输出

```
7
A B A C A B A
10
A B A C B A D C B A
10
Takamatsu Kaname Shiina Kaname Chihaya Takamatsu Nagasaki Takamatsu Chihaya Takamatsu
```



拓扑排序变形（9也可以过）

```python
# 23物院 宋昕杰
from heapq import *
import sys

input = lambda: sys.stdin.readline().strip()

for _ in range(int(input())):
    n = int(input())

    names = []
    edges = {i: {} for i in range(n)}
    hated = {i: {} for i in range(n)}
    in_degree = [0]*n
    cnt = 0

    for i in range(n):
        ls = input().split()
        names.append(ls[0])
        for t in ls[1:]:
            t = int(t) - 1
            edges[t][i] = hated[i][t] = True
            in_degree[i] += 1

    heap = []
    for i in range(n):
        if in_degree[i] == 0:
            heap.append(i)
    heapify(heap)
    ls = []
    inside = [False]*n
    while cnt < n:
        idx = heappop(heap)
        ls.append(names[idx])
        cnt += 1
        for i in hated[idx]:
            cnt -= 1
            for j in edges[i]:
                in_degree[j] += 1
            inside[i] = False
        for i in edges[idx]:
            in_degree[i] -= 1

        for i in hated[idx]:
            if in_degree[i] == 0 and not inside[i]:
                inside[i] = True
                heappush(heap, i)
        for i in edges[idx]:
            if in_degree[i] == 0 and not inside[i]:
                inside[i] = True
                heappush(heap, i)
    print(len(ls))
    print(*ls)
```





## 24837: 斗地主大师

http://cs101.openjudge.cn/practice/24837/

斗地主大师今天有P个欢乐豆，他夜观天象，算出了一个幸运数字Q，如果他能有恰好Q个欢乐豆，就可以轻松完成程设大作业了。

斗地主大师显然是斗地主大师，可以在斗地主的时候轻松操控游戏的输赢。

1.他可以轻松赢一把，让自己的欢乐豆变成原来的Y倍

2.他也可以故意输一把，损失X个欢乐豆(注意欢乐豆显然不能变成负数，所以如果手里没有X个豆就不能用这个能力)

而斗地主大师还有一种怪癖，扑克除去大小王只有52张，所以他一天之内最多只会打52把斗地主。

斗地主大师希望你能告诉他，为了把P个欢乐豆变成Q个，他至少要打多少把斗地主？

**输入**

第一行4个正整数 P,Q,X,Y
0< P,X,Q <= 2^31, 1< Y <= 225

**输出**

输出一个数表示斗地主大师至少要用多少次能力
如果打了52次斗地主也不能把P个欢乐豆变成Q个，请输出一行 "Failed"

样例输入

```
输入样例1：
2 2333 666 8
输入样例2：
1264574 285855522 26746122 3
```

样例输出

```
输出样例1：
Failed
输出样例2：
33
```

提示

可以考虑深搜
要用long long



二分查找dfs

```python
# 23物院 宋昕杰
p, q, x, y = map(int, input().split())
dp = {q: 0}
mid = 0


def dfs(p, cnt):
    if cnt > mid:
        return False
    if p > q and (p - q)//x + cnt > mid:
        return False
    elif p > q and (p - q)//x + cnt <= mid and (p - q) % x == 0:
        return True

    if p == q:
        return True

    cnt += 1
    if p > x and dfs(p - x, cnt):
        return True
    if dfs(p*y, cnt):
        return True
    return False


mid = 52
if not dfs(p, 0):
    print('Failed')
    exit()

l, r = 0, 52
while l < r:
    mid = (l + r)//2

    if dfs(p, 0):
        r = mid
    else:
        l = max(mid, l + 1)
print(r)
```



## 28405: 小明的刷题计划

http://xzmdsa.openjudge.cn/2024final/7/

为了提高自己的代码能力，小明制定了OpenJudge的刷题计划。他选中OpenJudge中的n道题，编号0~n-1，并计划在m天内按照题目编号顺序刷完所有题目。提醒：小明不能用多天完成同一题。

在小明刷题计划中，需要用time[i]的时间完成编号i的题目。同时，小明可以使用场外求助功能，通过询问编程高手小红可以省去该题的做题时间。为了防止小明过度依赖场外求助，小明每天只能使用一次求助机会。

我们定义m天中做题时间最多的一天耗时为T（小红完成的题目不计入做题总时间）。

请你帮助小明指定做题计划，求出最小的T是多少。

**输入**

第一行小明需要完成的题目个数N；
第二至N+1行小明完成每个题目的时间time；
最后一行小明计划完成刷题任务的时间m。

**输出**

做题最多一天的耗时T。

样例输入

```
4
1
2
3
3
2
```

样例输出

```
3
```

提示

1.1<=time.length<=10^5;
2.1<=time[i]<10000;
3.1<=m<=1000



二分查找

```python
# 23物院 宋昕杰
n = int(input())
times = [int(input()) for _ in range(n)]
m = int(input())

l, r = 0, sum(times) + 1


def judge(t):
    i = 0
    cnt = 0
    while i < n:
        total = 0
        max_today = 0
        while i < n and total <= t:
            total += times[i]
            max_today = max(max_today, times[i])
            i += 1

        total -= max_today

        while i < n:
            if total + times[i] <= t:
                total += times[i]
                i += 1
            else:
                break
        cnt += 1

    return cnt


while l < r:
    mid = (l + r) // 2
    cnt = judge(mid)

    if cnt <= m:
        r = mid
    else:
        l = max(mid, l + 1)

print(r)
```



## 03447: 银河贸易问题

http://cs101.openjudge.cn/practice/03447/

随着一种称为“￥”的超时空宇宙飞船的发明，一种叫做“￥￥”的地球与遥远的银河之间的商品进出口活动应运而生。￥￥希望从PluralZ星团中的一些银河进口商品，这些银河中的行星盛产昂贵的商品和原材料。初步的报告显示：
（1） 每个银河都包含至少一个和最多26个行星，在一个银河的每个行星用A~Z中的一个字母给以唯一的标识。
（2） 每个行星都专门生产和出口一种商品，在同一银河的不同行星出口不同的商品。
（3） 一些行星之间有超时空货运航线连接。如果行星A和B相连，则它们可以自由贸易；如果行星C与B相连而不与A相连，则A和C之间仍可通过B进行贸易，不过B要扣留5%的货物作为通行费。一般来说，只要两个行星之间可以通过一组货运航线连接，他们就可以进行贸易，不过每个中转站都要扣留5%的货物作为通行费。
（4） 在每个银河至少有一个行星开放一条通往地球的￥航线。对商业来说，￥航线和其他星际航线一样。
￥￥已经对每个行星的主要出口商品定价（不超过10的正实数），数值越高，商品的价值越高。在本地市场，越值钱的商品获利也越高。问题是要确定如果要考虑通行费是，哪个行星的商品价值最高。

输入

输入包含若干银河的描述。每个银河的描述开始的第1行是一个整数n，表示银河的行星数。接下来的n行每行包括一个行星的描述，即：
（1） 一行用以代表行星的字母；
（2） 一个空格；
（3） 以d.dd的形式给出该行星的出口商品的价值；
（4） 一个空格；
（5） 一个包含字母和（或）字符“*”的字符串；字母表示一条通往该行星的货运航线；“*”表示该行星向地球开放￥货运航线。

输出

对每个银河的描述，输出一个字母P表示在考虑通行费的前提下，行星P具有最高价值的出口商品。如果用有最高价值的出口商品的行星多于一个，只需输出字母序最小的那个行星。

样例输入

```
5
E 0.01 *A
D 0.01 A*
C 0.01 *A
A 1.00 EDCB
B 0.01 A*
```

样例输出

```
A
```





和这道爆了。”输入包含若干银河的描述“以为是多组输入，价值理解反了，被硬控30分钟。

”输入包含若干银河的描述“应该改成”输入包含银河的若干描述“

```python
# 23物院 宋昕杰
from queue import Queue

n = int(input())

values = []
names = []
ways = []
name_to_idx = {}
for i in range(n):
    t = input().split()
    if len(t) == 2:
        t.append('')
    name, value, way = t

    names.append(name)
    values.append(int(value[0] + value[2:]))
    ways.append(way)
    name_to_idx[name] = i

real_values = []
for i in range(n):
    q = Queue()
    q.put((i, 0))
    found = False
    vis = {}
    while not found and not q.empty():
        idx, distance = q.get()
        if idx in vis:
            continue
        vis[idx] = True

        for name in ways[idx]:
            if name == '*':
                found = True
                real_values.append((values[i]*0.95**distance, names[i]))
                break
            q.put((name_to_idx[name], distance + 1))

    if not found:
        real_values.append((0, names[i]))

real_values.sort(key=lambda t: (-t[0], t[1]))
print(real_values[0][1])
```



```python
# 肖添天
from collections import defaultdict, deque

n = int(input())
graph = defaultdict(set)
to_earth = set()
price = {}
for i in range(n):
    a, b, c = input().split()
    b = float(b)
    price[a] = b if a not in price else max(price[a], b)
    for x in c:
        if x == "*":
            to_earth.add(a)
        else:
            graph[a].add(x)
            graph[x].add(a)

def bfs(start):
    Q = deque([start])
    visited = set()
    visited.add(start)
    cnt = 0
    while Q:
        l = len(Q)
        for _ in range(l):
            f = Q.popleft()
            if f in to_earth:
                return price[start] * (0.95 ** cnt)
            for x in graph[f]:
                if x not in visited:
                    Q.append(x)
                    visited.add(x)
        cnt += 1
    return 0


ans = []
for planet in price.keys():
    ans.append((bfs(planet), planet))

ans.sort(key=lambda x: [-x[0], x[1]])
print(ans[0][1])
```



## 28416: Taki的乐队梦想（数据强化版）

http://xzmdsa.openjudge.cn/2024final/9/

题目描述全部同上，唯一的区别是增强了数据的强度。

现在T=20，n<=3000，保证输出总长度不超过500kB。

输入

同上

输出

同上

样例输入

```
同上
```

样例输出

```
同上
```

提示

如果涉及到大量的列表合并需求的话，使用list1.extend(list2)会比list1+=list2快很多。基本可以认为extend操作是O(1)的。
你的代码应该具有O(输入规模+输出规模)的时间复杂度。
（当然，如果你能靠优化常数通过这道题也是一种能力的体现）



同 28413
