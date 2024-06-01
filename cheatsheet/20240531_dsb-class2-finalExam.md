# 2024春-数据结构与算法B-2班



2024-05-31 17:50 ~ 20:20

http://sydsb.openjudge.cn/2024exam/

http://sydsb.openjudge.cn/2024jkkf/



8个题目，考了7个计概的。只有22460是数据结构题目。



## 28332: 收集金币达成成就

http://sydsb.openjudge.cn/2024jkkf/1/

小明和他的伙伴们正在玩一个游戏。游戏中有26种不同的金币和成就，金币用小写字母'a'到'z'表示，成就用大写字母'A'到'Z'表示。每种成就需要收集指定数量的金币才能达成：成就'A'需要26个'a'金币，成就'B'需要25个'b'金币，依此类推，成就'Z'需要1个'z'金币。玩家每分钟可以收集一枚金币。每个玩家的金币收集和成就独立于其他玩家。

游戏结束后，你拿到了每个玩家的金币收集记录——由**小写字母和空格**组成的字符串，表示依次收集的金币，空格表示该分钟内没有收集到金币。如果玩家收集的某种金币总数达到了达成对应成就所需的数量，他就获得该成就。获得成就后，玩家可能还会继续收集这种金币。

现在给出小明和伙伴们的金币收集记录，计算每个玩家在游戏期间达成了多少种成就，并输出依次达成的成就。

输入

若干行字符串表示若干个玩家，每一行包含一个长度为n的字符串 (1≤n≤1000)，字符串仅由小写字母和空格组成，表示一个玩家的金币收集记录。

输出

对于每个玩家输出一行结果，首先包含一个整数，表示该玩家达成的成就数量；如果成就数量不为0，再空一格，输出一个仅由大写字母组成的字符串，表示玩家依次达成的成就，否则仅输出成就数量0。

样例输入

```
xyy xxz
zz z
swsw sweet ttuu sswwwtt ttt
a bccba
```

样例输出

```
3 YXZ
1 Z
2 WT
0
```



```python
def achievements(records):
    results = []
    for record in records:
        coins = [0] * 26
        achieved = []
        for coin in record.replace(' ', ''):
            coins[ord(coin) - ord('a')] += 1
            if coins[ord(coin) - ord('a')] == 26 - (ord(coin) - ord('a')):
                achieved.append(chr(ord('A') + (ord(coin) - ord('a'))))
        if achieved:
            results.append(f"{len(achieved)} {''.join(achieved)}")
        else:
            results.append("0")
    return results

# 读取输入
records = []
while True:
    try:
        records.append(input().strip())
    except EOFError:
        break

# 计算并输出结果
results = achievements(records)
for result in results:
    print(result)
```





## 28231: 卡牌游戏

http://sydsb.openjudge.cn/2024jkkf/2//

小K同学最近迷上了一款名为《束狻十三：惧与罚》的卡牌游戏，在该游戏中角色受到攻击时可以选取背包中的**连续一列卡牌**进行防守。防守规则如下：选取的每张卡牌都有其对应的正整数能力值，选取一列卡牌后生成护盾至多能抵挡的伤害值为其所有卡牌能力值之积。现在给出其背包中N（N<=1000000）个卡牌分别的能力值，请计算共有多少种选取方式使得生成的护盾能够抵挡伤害为K（K在int表示范围内）的攻击。（这个数字可能很大，需要定义long long 类型，请输出其关于233333的余数，即ans%233333）

输入

输入共两行，第一行为两个正整数N,K.
第二行共N个正整数，表示背包中每个位置卡牌对应的能力值。

输出

一个数字，表示符合要求的卡牌选取方式数ans关于233333的余数。

样例输入

```
5 8
1 2 5 3 4
```

样例输出

```
9
```

提示

题目中K在int表示范围内，N<=1000000.(1后6个0）.如果定义数组元素>1000000的数组，建议定义为全局变量，比如int card[1000005];

样例中，共有如下9个连续子列符合要求：
(1,2,5) (1,2,5,3) (1,2,5,3,4)
(2,5) (2,5,3) (2,5,3,4)
(5,3) (5,3,4)
(3,4)

中间运算结果可能比较大，需要定义long long 类型，一个long long 变量占8个字节,比如 long long ans;
long long 类型的输出方式为 printf("%lld",ans);



Memory Limit Exceeded

```python
MOD = 233333


def count_valid_subarrays(N, K, prefix):

    for i in range(N):
        prefix[i+1] *= prefix[i]

    ans = 0
    for i in range(N):
        for j in range(i + 1, N+1):
            product = prefix[j] // prefix[i]
            if product >= K:
                ans = (ans + N - j + 1) % MOD
                break

    return ans

N, K = map(int, input().split())
cards = [1] + list(map(int, input().split()))

result = count_valid_subarrays(N, K, cards)
print(result)
```





## 28332: 小明的加密算法

http://sydsb.openjudge.cn/2024jkkf/3/

小明设计了一个加密算法，用于加密一个字母字符串 s。

假设 s 中只有小写字母，且a-z的值分别对应正整数1-26。

小明想利用栈这个结构来加密字符串 s：

1. 从左到右遍历字符串 s，对于每个字符 c，先将其压入栈中。
2. 如果此时栈顶元素的值为偶数，则将所有元素出栈。如果此时栈顶元素的值为奇数，则继续遍历下一个字符。
3. 如果所有字符都遍历完毕，但栈中仍有元素，将0压入栈中。此时，栈顶元素为偶数0，将所有元素出栈。

最终，栈中的元素按出栈顺序组成一个新的字符串 t。小明将字符串 t 作为加密后的字符串。

现在你需要完成该算法的加解密算法。加密算法和解密算法的定义如下：

加密算法：给定一个字符串 s，返回按照上述算法加密后的字符串 t。

解密算法：给定一个字符串 t，返回解密后的字符串 s，即，s可以通过上述算法加密得到 t。

输入

每个测试包含多个测试用例。第一行包含测试用例的数量 t（1 ≤ t ≤ 100）。每个测试用例的描述如下。

每个测试用例的第一行为"encrypt"或"decrypt"，表示加密或解密操作。

第二行包含一个待加密或解密的字符串 s 或 t，长度不超过 100。

输出

对于每个测试用例，输出一个字符串作为答案，每个答案占一行。

样例输入

```
2
encrypt
abcde
decrypt
badc
```

样例输出

```
badc0e
abcd
```



```python
def encrypt(s):
    stack = []
    result = []

    for c in s:
        stack.append(c)
        if ord(c) % 2 == 0:
            while stack:
                result.append(stack.pop())

    if stack:
        stack.append('0')
        while stack:
            result.append(stack.pop())

    return ''.join(result)

def decrypt(t):
    result = []
    temp = []

    for c in t[::-1]:
        if c == '0':
            result += temp
            temp.clear()
        elif ord(c)  % 2 == 0:
            result = temp + [c] + result
            temp.clear()
        else:
            temp.append(c)

    return ''.join(result)


n = int(input())

results = []

for _ in range(n):
    operation = input()
    string = input()

    if operation == "encrypt":
        results.append(encrypt(string))
    elif operation == "decrypt":
        results.append(decrypt(string))

for result in results:
    print(result)

```



## 28307: 老鼠和奶酪

http://cs101.openjudge.cn/practice/28307/

有两只老鼠和 n 块不同类型的奶酪，每块奶酪都只能被其中一只老鼠吃掉。
下标为 i 处的奶酪被吃掉的得分为：
如果被第一只老鼠吃掉，则得分为reward1[i] 。 
如果被第二只老鼠吃掉，则得分为reward2[i]。 
给你一个正整数数组 reward1 ，一个正整数数组 reward2，和一个非负整数 k 。
请你返回全部奶酪被吃完，但第一只老鼠恰好吃掉 k 块奶酪的情况下，总的最大得分为多少。

输入

第一行是一个整数n，代表奶酪个数。1<=n<=10000。
第二行是n个整数，代表上述reward1，分别对应n个奶酪被第一只老鼠吃掉的得分。行内使用空格分割。
第三行是n个整数，代表上述reward2，分别对应n个奶酪被第二只老鼠吃掉的得分。行内使用空格分割。
第一只或第二只老鼠吃到任意奶酪的得分1<=reward1[i],reward2[i]<=1000
第四行是一个整数k，代表第一只老鼠吃掉的奶酪块数。 0<=k<=n。

输出

一个整数，代表最大得分。

样例输入

```
4
1 1 3 4
4 4 1 1
2
```

样例输出

```
15
```

提示

贪心法



```python
def max_cheese_score(n, reward1, reward2, k):
    # 计算每块奶酪的得分差值
    differences = [(reward1[i] - reward2[i], i) for i in range(n)]
    
    # 按得分差值从大到小排序
    differences.sort(reverse=True, key=lambda x: x[0])
    
    # 初始化总得分
    total_score = 0
    
    # 选择前k块奶酪给第一只老鼠
    for i in range(k):
        index = differences[i][1]
        total_score += reward1[index]
    
    # 选择剩余的奶酪给第二只老鼠
    for i in range(k, n):
        index = differences[i][1]
        total_score += reward2[index]
    
    return total_score

# 输入处理
n = int(input())
reward1 = list(map(int, input().split()))
reward2 = list(map(int, input().split()))
k = int(input())

# 计算并输出结果
print(max_cheese_score(n, reward1, reward2, k))
```





## 28321: 电影排片

http://cs101.openjudge.cn/practice/28321/

一个电影排片系统现在要安排n部电影的上映，第i部电影的评分要求不小于bi。

现在已经有一个初步的上映电影的列表，长度为n，第i部电影的评分为ai。

最初，数组 {ai} 和 {bi} 均按升序排列。

由于某些电影的评分可能低于要求，即存在 ai < bi，因此我们需要添加评分更高的电影（假设我们有无数多的任何评分的电影供选择）。

当我们添加一个评分为 0 <= w <= 100 的新电影进入排片系统时，系统将删除评分最低的电影，并按升序排列电影的评分。

换句话说，在每次操作中，你选择一个整数 w，将其插入数组 {ai} 中，然后将数组 {ai} 按升序排列，并移除第一个元素。

求需要添加的最少新电影数量，使得对于所有的 i，满足 ai >= bi。

输入

每个测试包含多个测试用例。第一行包含测试用例的数量 t（1 ≤ t ≤ 10000）。每个测试用例的描述如下。

每个测试用例的第一行包含一个正整数 n（1 ≤ n ≤ 100），表示电影数量。

第二行包含一个长度为 n 的数组 a（0 ≤ a1 ≤ a2 ≤ ... ≤ an ≤ 100）。

第三行包含一个长度为 n 的数组 b（0 ≤ b1 ≤ b2 ≤ ... ≤ bn ≤ 100）。

输出

对于每个测试用例，输出一个整数作为答案，每个答案占一行。

样例输入

```
2
6
10 20 30 40 50 60
8 21 35 35 55 65

6
9 9 9 30 40 50
10 20 30 40 50 60
```

样例输出

```
1
3
```

提示

在第一个测试用例中：
\- 添加一个评分为 w = 70 的电影，数组 a 变为 [20, 30, 40, 50, 60, 70]。

在第二个测试用例中：
\- 添加一个评分为 w = 60 的电影，数组 a 变为 [9, 9, 30, 40, 50, 60]。
\- 添加一个评分为 w = 20 的电影，数组 a 变为 [9, 20, 30, 40, 50, 60]。
\- 添加一个评分为 w = 10 的电影，数组 a 变为 [10, 20, 30, 40, 50, 60]。





```python
from collections import deque

t = int(input())
for _ in range(t):
    n = int(input())
    a = deque(map(int, input().split()))
    b = deque(map(int, input().split()))

    cnt = 0
    i = 0
    while i < n:
        if a[i] < b[i]:
            cnt += 1
            a.popleft()
            a.append(b[-1] + 1)
            i = 0
            continue

        i += 1

    print(cnt)

```





## 28274: 细胞计数

http://cs101.openjudge.cn/practice/28274/

小北是一名勤奋的医学生，某天，他们实验室中新来了一台显微镜，这台显微镜拥有先进的图像分析功能，能够通过特殊的算法识别和分析细胞结构。显微镜能识别每个图像点的物质密度，并用1到9加以标识，对于空白区域则用数字0标识。同时我们规定，所有上、下、左、右联通，且数字不为0的图像点都属于同一细胞，请你帮助小北求出给定图像内的细胞个数。

输入

第一行两个整数代表矩阵大小n和m。
接下来n行，每行一个长度为m的只含字符 0 到 9 的字符串，代表这个nxm的矩阵。

输出

输出细胞个数

样例输入

```
4 10
0234500067
1034560500
2045600671
0000000089
```

样例输出

```
4
```



```python
def count_cells(n, m, matrix):
    visited = [[False]*m for _ in range(n)]
    dx = [-1, 0, 1, 0]
    dy = [0, 1, 0, -1]

    def dfs(x, y):
        visited[x][y] = True
        for i in range(4):
            nx, ny = x + dx[i], y + dy[i]
            if 0 <= nx < n and 0 <= ny < m and not visited[nx][ny] and matrix[nx][ny] != '0':
                dfs(nx, ny)

    count = 0
    for i in range(n):
        for j in range(m):
            if not visited[i][j] and matrix[i][j] != '0':
                dfs(i, j)
                count += 1

    return count

n, m = map(int, input().split())
matrix = [list(input().strip()) for _ in range(n)]
print(count_cells(n, m, matrix))
```



## 22460: 火星车勘探

http://cs101.openjudge.cn/practice/22460/

火星这颗自古以来寄托了中国人无限遐思的红色星球，如今第一次留下了中国人的印迹。2021年5月15日，“天问一号”探测器成功降落在火星预选着陆区。这标志着中国首次火星探测着陆任务取得成功，同时也使中国成为继美国之后第二个实现探测器着陆火星的国家。

假设火星车需要对形如二叉树的地形进行遍历勘察。火星车初始处于二叉树地形的根节点，对二叉树进行前序遍历。当火星车遇到非空节点时，则采样一定量的泥土样本，记录下样本数量；当火星车遇到空节点，使用一个标记值`#`进行记录。

![img](http://media.openjudge.cn/images/upload/3861/1621603311.png)

对上面的二叉树地形可以前序遍历得到 `9 3 4 # # 1 # # 2 # 6 # #`，其中 `#` 代表一个空节点，整数表示在该节点采样的泥土样本数量。

我们的任务是，给定一串以空格分隔的序列，验证它是否是火星车对于二叉树地形正确的前序遍历结果。



输入

每组输入包含多个测试数据，每个测试数据由两行构成。
每个测试数据的第一行：1个正整数N，表示遍历结果中的元素个数。
每个测试数据的第二行：N个以空格分开的元素，每个元素可以是#，也可以是小于100的正整数。(1<=N<=200000)
输入的最后一行为0，表示输入结束。

输出

对于每个测试数据，输出一行判断结果。
输入的序列如果是对某个二叉树的正确的前序遍历结果，则输出“T”，否则输出“F”。

样例输入

```
13
9 3 4 # # 1 # # 2 # 6 # #
4
9 # # 1
2
# 99
0
```

样例输出

```
T
F
F
```



验证给定的前序遍历结果是否能构成一个有效的二叉树。需要通过维护节点的出入度来检查遍历是否有效。

在前序遍历中，每个非空节点会贡献2个出度（左右子节点），而每个节点（包括空节点）会消耗1个入度。因此，我们可以通过维护一个计数器来记录当前剩余的可用入度。

```python
def is_valid_preorder(n, sequence):
    out_degree = 1  # Initial out_degree for the root

    for value in sequence:
        out_degree -= 1  # Every node uses one out_degree

        if out_degree < 0:
            return "F"

        if value != '#':
            out_degree += 2  # Non-null nodes provide 2 out_degrees

    return "T" if out_degree == 0 else "F"

# 读取输入
import sys
input = sys.stdin.read

data = input().strip().split('\n')
index = 0

while index < len(data):
    n = int(data[index])
    if n == 0:
        break
    index += 1
    sequence = data[index].split()
    index += 1
    print(is_valid_preorder(n, sequence))

```







## 28389: 跳高

http://cs101.openjudge.cn/practice/28389/

体育老师组织学生进行跳高训练，查看其相对于上一次训练中跳高的成绩是否有所进步。为此，他组织同学们按学号排成一列进行测试。本次测验使用的老式测试仪，只能判断同学跳高成绩是否高于某一预设值，且由于测试仪器构造的问题，其横杠只能向上移动。由于老师只关心同学是否取得进步，因此老师只将跳高的横杠放在该同学上次跳高成绩的位置，查看同学是否顺利跃过即可。为了方便进行上次成绩的读取，同学们需按照顺序进行测验，因此对于某个同学，当现有的跳高测试仪高度均高于上次该同学成绩时，体育老师需搬出一个新的测试仪进行测验。已知同学们上次测验的成绩，请问体育老师至少需要使用多少台测试仪进行测验？

由于采用的仪器精确度很高，因此测试数据以毫米为单位，同学们的成绩为正整数，最终测试数据可能很大，但不超过10000，且可能存在某同学上次成绩为0。

输入

输入共两行，第一行为一个数字N，N<=100000，表示同学的数量。第二行为N个数字，表示同学上次测验的成绩（从1号到N号排列）。

输出

一个正整数，表示体育老师最少需要的测试仪数量。

样例输入

```
5
1 7 3 5 2
```

样例输出

```
3
```

提示

1.50%的数据中，N<=5000.100%的数据中，N<=100000.
2.可通过观察规律将题目转化为学过的问题进行求解。
3.对于10w的数据，朴素算法可能会超时，可以采用二分法进行搜索上的优化。



这个题目挺好的，维护多个非递减队列。

```python
from bisect import bisect_right


def min_instruments_needed(scores):
    instruments = []  # 用于存储当前的各个递增序列的最后一个元素

    for score in scores:
        if instruments:
            # 找到第一个大于或等于当前成绩的位置
            pos = bisect_right(instruments, score)
            if pos == 0:
                instruments.insert(0, score)
            else:
                instruments[pos - 1] = score

        else:
            instruments.append(score)

    return len(instruments)


N = int(input())
scores = list(map(int, input().split()))

result = min_instruments_needed(scores)
print(result)

"""
5
1 2 1 3 2

2
"""
```



```python
"""
Dilworth定理:
Dilworth定理表明，任何一个有限偏序集的最长反链(即最长下降子序列)的长度，
等于将该偏序集划分为尽量少的链(即上升子序列)的最小数量。
因此，计算序列的最长下降子序列长度，即可得出最少需要多少台测试仪。
"""

from bisect import bisect_left

def min_testers_needed(scores):
    scores.reverse()  # 反转序列以找到最长下降子序列的长度
    lis = []  # 用于存储最长上升子序列

    for score in scores:
        pos = bisect_left(lis, score)
        if pos < len(lis):
            lis[pos] = score
        else:
            lis.append(score)

    return len(lis)


N = int(input())
scores = list(map(int, input().split()))

result = min_testers_needed(scores)
print(result)
```

