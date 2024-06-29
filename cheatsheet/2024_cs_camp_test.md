# 2024北京大学计算机学院优秀大学生暑期夏令营机试

http://bailian.openjudge.cn/xly2024062702/

2024-06-27 18:00~20:00



| 题目               | tags             |
| ------------------ | ---------------- |
| 金币               | implementation   |
| 话题焦点人物       | Implementation   |
| 算24               | bruteforce       |
| 怪盗基德的滑翔翼   | dp               |
| 合并编码           | Stack            |
| Sorting It All Out | topological sort |
| A Bug's Life       | disjoint set     |
| 正方形破坏者       | IDA*             |







## 02000: 金币

http://cs101.openjudge.cn/practice/02000/

国王将金币作为工资，发放给忠诚的骑士。第一天，骑士收到一枚金币；之后两天（第二天和第三天）里，每天收到两枚金币；之后三天（第四、五、六天）里，每天收到三枚金币；之后四天（第七、八、九、十天）里，每天收到四枚金币……这种工资发放模式会一直这样延续下去：当连续N天每天收到N枚金币后，骑士会在之后的连续N+1天里，每天收到N+1枚金币（N为任意正整数）。

你需要编写一个程序，确定从第一天开始的给定天数内，骑士一共获得了多少金币。

输入

输入包含至少一行，但不多于21行。除最后一行外，输入的每行是一组输入数据，包含一个整数（范围1到10000），表示天数。输入的最后一行为0，表示输入结束。

输出

对每个数据输出一行，包含该数据对应天数和总金币数，用单个空格隔开。

样例输入

```
10
6
7
11
15
16
100
10000
1000
21
22
0
```

样例输出

```
10 30
6 14
7 18
11 35
15 55
16 61
100 945
10000 942820
1000 29820
21 91
22 98
```

来源

Rocky Mountain 2004



```python
def calculate_gold_days(days):
    total_gold = 0
    days_count = 1
    while days > 0:
        for _ in range(days_count):
            if days == 0:
                break
            total_gold += days_count
            days -= 1
        days_count += 1
    return total_gold

while True:
    days = int(input())
    if days == 0:
        break
    total_gold = calculate_gold_days(days)
    print(f"{days} {total_gold}")
```





## 06901: 话题焦点人物

http://cs101.openjudge.cn/practice/06901/

微博提供了一种便捷的交流平台。一条微博中，可以提及其它用户。例如Lee发出一条微博为：“期末考试顺利 @Kim @Neo”，则Lee提及了Kim和Neo两位用户。

我们收集了N(1 < N < 10000)条微博，并已将其中的用户名提取出来，用小于等于100的正整数表示。

通过分析这些数据，我们希望发现大家的话题焦点人物，即被提及最多的人（题目保证这样的人有且只有一个），并找出那些提及它的人。

**输入**

输入共两部分：
第一部分是微博数量N，1 < N < 10000。
第二部分是N条微博，每条微博占一行，表示为：
发送者序号a，提及人数k(0 < = k < = 20)，然后是k个被提及者序号b1,b2...bk；
其中a和b1,b2...bk均为大于0小于等于100的整数。相邻两个整数之间用单个空格分隔。

**输出**

输出分两行：
第一行是被提及最多的人的序号；
第二行是提及它的人的序号，从小到大输出，相邻两个数之间用单个空格分隔。同一个序号只输出一次。

样例输入

```
5
1 2 3 4
1 0
90 3 1 2 4
4 2 3 2
2 1 3
```

样例输出

```
3
1 2 4
```

来源

医学部计算概论2011年期末考试（谢佳亮）



```python
def find_topic_center_and_mentioners():
    n = int(input())
    mention_count = {}  # 记录每个人被提及的次数
    mention_relations = {}  # 记录提及关系，key为提及的人，value为提及的人的集合
    
    for _ in range(n):
        tweet = input().split()
        sender, k = int(tweet[0]), int(tweet[1])
        if k > 0:
            mentioned = list(map(int, tweet[2:]))
            for person in mentioned:
                if person not in mention_count:
                    mention_count[person] = 1
                    mention_relations[person] = set([sender])
                else:
                    mention_count[person] += 1
                    mention_relations[person].add(sender)
    
    # 找到被提及最多的人
    topic_center = max(mention_count, key=mention_count.get)
    
    # 输出结果
    print(topic_center)
    print(' '.join(map(str, sorted(mention_relations[topic_center]))))

# 调用函数处理输入数据
find_topic_center_and_mentioners()
```



## 02787: 算24

http://cs101.openjudge.cn/practice/02787/

给出4个小于10个正整数，你可以使用加减乘除4种运算以及括号把这4个数连接起来得到一个表达式。现在的问题是，是否存在一种方式使得得到的表达式的结果等于24。

这里加减乘除以及括号的运算结果和运算的优先级跟我们平常的定义一致（这里的除法定义是实数除法）。

比如，对于5，5，5，1，我们知道5 * (5 – 1 / 5) = 24，因此可以得到24。又比如，对于1，1，4，2，我们怎么都不能得到24。

输入

输入数据包括多行，每行给出一组测试数据，包括4个小于10个正整数。最后一组测试数据中包括4个0，表示输入的结束，这组数据不用处理。

输出

对于每一组测试数据，输出一行，如果可以得到24，输出“YES”；否则，输出“NO”。

样例输入

```
5 5 5 1
1 1 4 2
0 0 0 0
```

样例输出

```
YES
NO
```





```python
'''
在这个优化的代码中，我们使用了递归和剪枝策略。首先按照题目的要求，输入的4个数字保持不变，
不进行排序。在每一次运算中，我们首先尝试加法和乘法，因为它们的运算结果更少受到数字大小的影响。
然后，我们根据数字的大小关系尝试减法和除法，只进行必要的组合运算，避免重复运算。

值得注意的是，这种优化策略可以减少冗余计算，但对于某些输入情况仍需要遍历所有可能的组合。
因此，在最坏情况下仍然可能需要较长的计算时间。
'''

from functools import lru_cache 

@lru_cache(maxsize = None)
def find(nums):
    if len(nums) == 1:
        return abs(nums[0] - 24) <= 0.000001

    for i in range(len(nums)):
        for j in range(i+1, len(nums)):
            a = nums[i]
            b = nums[j]
            remaining_nums = []

            for k in range(len(nums)):
                if k != i and k != j:
                    remaining_nums.append(nums[k])

            # 尝试加法和乘法运算
            if find(tuple(remaining_nums + [a + b])) or find(tuple(remaining_nums + [a * b])):
                return True

            # 尝试减法运算
            if a > b and find(tuple(remaining_nums + [a - b])):
                return True
            if b > a and find(tuple(remaining_nums + [b - a])):
                return True

            # 尝试除法运算
            if b != 0 and find(tuple(remaining_nums + [a / b])):
                return True
            if a != 0 and find(tuple(remaining_nums + [b / a])):
                return True

    return False

while True:
    card = [int(x) for x in input().split()]
    if sum(card) == 0:
        break

    print("YES" if find(tuple(card)) else "NO")
```





## 04977: 怪盗基德的滑翔翼

http://cs101.openjudge.cn/practice/04977/

怪盗基德是一个充满传奇色彩的怪盗，专门以珠宝为目标的超级盗窃犯。而他最为突出的地方，就是他每次都能逃脱中村警部的重重围堵，而这也很大程度上是多亏了他随身携带的便于操作的滑翔翼。

有一天，怪盗基德像往常一样偷走了一颗珍贵的钻石，不料却被柯南小朋友识破了伪装，而他的滑翔翼的动力装置也被柯南踢出的足球破坏了。不得已，怪盗基德只能操作受损的滑翔翼逃脱。

![img](http://media.openjudge.cn/images/upload/1340073200.jpg)

假设城市中一共有N幢建筑排成一条线，每幢建筑的高度各不相同。初始时，怪盗基德可以在任何一幢建筑的顶端。他可以选择一个方向逃跑，但是不能中途改变方向（因为中森警部会在后面追击）。因为滑翔翼动力装置受损，他只能往下滑行（即：只能从较高的建筑滑翔到较低的建筑）。他希望尽可能多地经过不同建筑的顶部，这样可以减缓下降时的冲击力，减少受伤的可能性。请问，他最多可以经过多少幢不同建筑的顶部（包含初始时的建筑）？



输入

输入数据第一行是一个整数K（K < 100），代表有K组测试数据。
每组测试数据包含两行：第一行是一个整数N(N < 100)，代表有N幢建筑。第二行包含N个不同的整数，每一个对应一幢建筑的高度h（0 < h < 10000），按照建筑的排列顺序给出。

输出

对于每一组测试数据，输出一行，包含一个整数，代表怪盗基德最多可以经过的建筑数量。

样例输入

```
3
8
300 207 155 299 298 170 158 65
8
65 158 170 298 299 155 207 300
10
2 1 3 4 5 6 7 8 9 10
```

样例输出

```
6
6
9
```





```python
def max_increasing_subsequence(a):
    n = len(a)
    dpu = [1] * n
    for i in range(1, n):
        for j in range(i):
            if a[i] > a[j]:
                dpu[i] = max(dpu[i], dpu[j] + 1)
    return max(dpu)

def max_decreasing_subsequence(a):
    n = len(a)
    dpd = [1] * n
    for i in range(1, n):
        for j in range(i):
            if a[i] < a[j]:
                dpd[i] = max(dpd[i], dpd[j] + 1)
    return max(dpd)

def main():
    k = int(input())
    while k:
        k -= 1
        n = int(input())
        a = list(map(int, input().split()))
        mxu = max_increasing_subsequence(a)
        mxd = max_decreasing_subsequence(a)
        print(max(mxu, mxd))

if __name__ == "__main__":
    main()
```





## 28496: 合并编码

给定一个合并编码后的字符串，返回原始字符串。

编码规则为: k[string]，表示其中方括号内部的string重复 k 次，k为正整数且0 < k < 20。
可以认为输入数据中所有的数字只表示重复的次数k，且方括号总是满足要求的，例如不会出现像 3a 、 2[b 或 2[4] 的输入。

输入保证符合格式，且不包含空格。

输入

一行，一个长度为n(1 ≤ n ≤ 500)的字符串，代表合并编码后的字符串。

输出

一行，代表原始字符串。长度m满足0 ≤ m ≤ 1500。

样例输入

```
样例1：
3[abc]1[o]2[n]

样例2：
3[a2[c]]
```

样例输出

```
样例1：
abcabcabconn

样例2：
accaccacc
```



```python
def decode_string(s):
    stack = []
    current_num = 0
    current_str = ''
    result = ''

    for char in s:
        if char.isdigit():
            current_num = current_num * 10 + int(char)
        elif char == '[':
            stack.append(current_str)
            stack.append(current_num)
            current_str = ''
            current_num = 0
        elif char == ']':
            num = stack.pop()
            prev_str = stack.pop()
            current_str = prev_str + num * current_str
        else:
            current_str += char

    return current_str

# 示例
input_str1 = "3[abc]1[o]2[n]"
input_str2 = "3[a2[c]]"

print(decode_string(input_str1))  # 应输出: abcabcabconn
print(decode_string(input_str2))  # 应输出: accaccacc
```





## 01094: Sorting It All Out

http://cs101.openjudge.cn/practice/01094/

An ascending sorted sequence of distinct values is one in which some form of a less-than operator is used to order the elements from smallest to largest. For example, the sorted sequence A, B, C, D implies that A < B, B < C and C < D. in this problem, we will give you a set of relations of the form A < B and ask you to determine whether a sorted order has been specified or not. 

输入

Input consists of multiple problem instances. Each instance starts with a line containing two positive integers n and m. the first value indicated the number of objects to sort, where 2 <= n <= 26. The objects to be sorted will be the first n characters of the uppercase alphabet. The second value m indicates the number of relations of the form A < B which will be given in this problem instance. Next will be m lines, each containing one such relation consisting of three characters: an uppercase letter, the character "<" and a second uppercase letter. No letter will be outside the range of the first n letters of the alphabet. Values of n = m = 0 indicate end of input.

输出

For each problem instance, output consists of one line. This line should be one of the following three:

Sorted sequence determined after xxx relations: yyy...y.
Sorted sequence cannot be determined.
Inconsistency found after xxx relations.

where xxx is the number of relations processed at the time either a sorted sequence is determined or an inconsistency is found, whichever comes first, and yyy...y is the sorted, ascending sequence.

样例输入

```
4 6
A<B
A<C
B<C
C<D
B<D
A<B
3 2
A<B
B<A
26 1
A<Z
0 0
```

样例输出

```
Sorted sequence determined after 4 relations: ABCD.
Inconsistency found after 2 relations.
Sorted sequence cannot be determined.
```

来源

East Central North America 2001





```python
#23n2310307206胡景博
from collections import deque
def topo_sort(graph):
    in_degree = {u:0 for u in graph}
    for u in graph:
        for v in graph[u]:
            in_degree[v] += 1
    q = deque([u for u in in_degree if in_degree[u] == 0])
    topo_order = [];flag = True
    while q:
        if len(q) > 1:
            flag = False#topo_sort不唯一确定
        u = q.popleft()
        topo_order.append(u)
        for v in graph[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                q.append(v)
    if len(topo_order) != len(graph): return 0
    return topo_order if flag else None
while True:
    n,m = map(int,input().split())
    if n == 0: break
    graph = {chr(x+65):[] for x in range(n)}
    edges = [tuple(input().split('<')) for _ in range(m)]
    for i in range(m):
        a,b = edges[i]
        graph[a].append(b)
        t = topo_sort(graph)
        if t:
            s = ''.join(t)
            print("Sorted sequence determined after {} relations: {}.".format(i+1,s))
            break
        elif t == 0:
            print("Inconsistency found after {} relations.".format(i+1))
            break
    else:
        print("Sorted sequence cannot be determined.")
```





## 02492: A Bug's Life

http://cs101.openjudge.cn/practice/02492/

**Background**
Professor Hopper is researching the sexual behavior of a rare species of bugs. He assumes that they feature two different genders and that they only interact with bugs of the opposite gender. In his experiment, individual bugs and their interactions were easy to identify, because numbers were printed on their backs.
**Problem**
Given a list of bug interactions, decide whether the experiment supports his assumption of two genders with no homosexual bugs or if it contains some bug interactions that falsify it.

输入

The first line of the input contains the number of scenarios. Each scenario starts with one line giving the number of bugs (at least one, and up to 2000) and the number of interactions (up to 1000000) separated by a single space. In the following lines, each interaction is given in the form of two distinct bug numbers separated by a single space. Bugs are numbered consecutively starting from one.

输出

The output for every scenario is a line containing "Scenario #i:", where i is the number of the scenario starting at 1, followed by one line saying either "No suspicious bugs found!" if the experiment is consistent with his assumption about the bugs' sexual behavior, or "Suspicious bugs found!" if Professor Hopper's assumption is definitely wrong.

样例输入

```
2
3 3
1 2
2 3
1 3
4 2
1 2
3 4
```

样例输出

```
Scenario #1:
Suspicious bugs found!

Scenario #2:
No suspicious bugs found!
```

提示

Huge input,scanf is recommended.

来源

TUD Programming Contest 2005, Darmstadt, Germany





```python
class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, x):
        if x != self.parent[x]:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            self.parent[rootY] = rootX

    def is_connected(self, x, y):
        return self.find(x) == self.find(y)


def solve_bug_life(scenarios):
    for i in range(1, scenarios + 1):
        n, m = map(int, input().split())
        uf = UnionFind(2 * n + 1)  # 为每个虫子创建两个节点表示其可能的两种性别
        suspicious = False
        for _ in range(m):
            u, v = map(int, input().split())
            if suspicious:
                continue

            if uf.is_connected(u, v):
                suspicious = True
            uf.union(u, v + n)  # 将u的一种性别与v的另一种性别关联
            uf.union(u + n, v)  # 同理


        print(f'Scenario #{i}:')
        print('Suspicious bugs found!' if suspicious else 'No suspicious bugs found!')
        print()


# 读取场景数量并解决问题
scenarios = int(input())
solve_bug_life(scenarios)
```





## 01084: 正方形破坏者

http://cs101.openjudge.cn/practice/01084/

- 左图展示了一个由 24 根火柴棍组成的 3 * 3 的网格，所有火柴棍的长度都是 1。在这张网格图中有很多的正方形：边长为 1 的有 9 个，边长为 2 的有 4 个，边长为 3 的有 1 个。

  每一根火柴棍都被编上了一个号码，编码的方式是从上到下：横着的第一行，竖着的第一行，横着的第二行一直到横着的最后一行。在同一行内部，编码的方式是从左到右。 其中对 3 * 3 的火柴网格编码的结果已经标在左图上了。

  右图展示了一个不完整的 3 * 3 的网格，它被删去了编号为 12,17,23 的火柴棍。删去这些火柴棍后被摧毁了 5 个大小为 1 的正方形，3 个大小为 2 的正方形和 1 个大小为 3 的正方形。（一个正方形被摧毁当且仅当它的边界上有至少一个火柴棍被移走了）

  

  ![img](http://media.openjudge.cn/images/g86/1084.gif)

  

  可以把上述概念推广到 n * n 的火柴棍网格。在完整的 n * n 的网格中，使用了 2n(n+1) 根火柴棍，其中边长为 i(i ∈ [1,n]) 的正方形有 (n-i+1)2个。

  现在给出一个 n * n 的火柴棍网格，最开始它被移走了 k 根火柴棍。问最少再移走多少根火柴棍，可以让所有的正方形都被摧毁。

  输入

  输入包含多组数据，第一行一个整数 T 表示数据组数。

  对于每组数据，第一行输入一个整数 n 表示网格的大小( n <= 5)。第二行输入若干个空格隔开的整数，第一个整数 k 表示被移走的火柴棍个数，接下来 k 个整数表示被移走的火柴棍编号。

  输出

  对于每组数据，输出一行一个整数表示最少删除多少根火柴棍才能摧毁所有的正方形。

  样例输入

  ```
  2
  2
  0
  3
  3 12 17 23
  ```

  样例输出

  ```
  3
  3
  ```



Deng_Leo

https://blog.csdn.net/2301_79402523/article/details/137194237

```python
import copy
import sys
sys.setrecursionlimit(1 << 30)
found = False
 
def check1(x, tmp):
    for y in graph[x]:
        if tmp[y]:
            return False
    return True
 
def check2(x):
    for y in graph[x]:
        if judge[y]:
            return False
    return True
 
def estimate():
    cnt = 0
    tmp = copy.deepcopy(judge)
    for x in range(1, total+1):
        if check1(x, tmp):
            cnt += 1
            for u in graph[x]:
                tmp[u] = True
    return cnt
 
def dfs(t):
    global found
    if t + estimate() > limit:
        return
    for x in range(1, total+1):
        if check2(x):
            for y in graph[x]:
                judge[y] = True
                dfs(t+1)
                judge[y] = False
                if found:
                    return
            return
    found = True
 
for _ in range(int(input())):
    n = int(input())
    lst = list(map(int, input().split()))
    d, m, nums, total = 2*n+1, lst[0], lst[1:], 0
    graph = {}
    for i in range(n):
        for j in range(n):
            for k in range(1, n+1):
                if i+k <= n and j+k <= n:
                    total += 1
                    graph[total] = []
                    for p in range(1, k+1):
                        graph[total] += [d*i+j+p, d*(i+p)+j-n, d*(i+p)+j-n+k, d*(i+k)+j+p]
    judge = [False for _ in range(2*n*(n+1)+1)]
    for num in nums:
        judge[num] = True
    limit = estimate()
    found = False
    while True:
        dfs(0)
        if found:
            print(limit)
            break
        limit += 1
```













```python

```

