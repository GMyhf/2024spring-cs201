# 2024北京大学智能学院优秀大学生暑期夏令营机试

http://bailian.openjudge.cn/xly2024062701/

2024-06-27 14:00~16:00



| 题目                           | tags           |
| ------------------------------ | -------------- |
| 画矩形                         | implementation |
| 花生采摘                       | Implementation |
| 棋盘问题                       | dfs            |
| 最大上升子序列和               | dp             |
| 文件结构“图”                   | tree           |
| Stockbroker                    | floyd_warshall |
| 由中根序列和后根序列重建二叉树 | tree           |
| 超级备忘录                     | splay tree     |



## 08183:画矩形

http://cs101.openjudge.cn/practice/08183/

根据参数，画出矩形。

输入

输入一行，包括四个参数：前两个参数为整数，依次代表矩形的高和宽（高不少于3行不多于10行，宽不少于5列不多于10列）；第三个参数是一个字符，表示用来画图的矩形符号；第四个参数为1或0，0代表空心，1代表实心。

输出

输出画出的图形。

样例输入

```
7 7 @ 0
```

样例输出

```
@@@@@@@
@     @
@     @
@     @
@     @
@     @
@@@@@@@
```





```python
def draw_rectangle(height, width, char, is_filled):
    for i in range(height):
        for j in range(width):
            if i == 0 or i == height - 1 or j == 0 or j == width - 1 or is_filled:
                print(char, end="")
            else:
                print(" ", end="")
        print()

# Test the function
h,w,c,f = input().split()
draw_rectangle(int(h), int(w), c, int(f))

```



## 07902:花生采摘

http://cs101.openjudge.cn/practice/07902/

鲁宾逊先生有一只宠物猴，名叫多多。这天，他们两个正沿着乡间小路散步，突然发现路边的告示牌上贴着一张小小的纸条：“欢迎免费品尝我种的花生！——熊字”。

鲁宾逊先生和多多都很开心，因为花生正是他们的最爱。在告示牌背后，路边真的有一块花生田，花生植株整齐地排列成矩形网格（如图1）。有经验的多多一眼就能看出，每棵花生植株下的花生有多少。为了训练多多的算术，鲁宾逊先生说：“你先找出花生最多的植株，去采摘它的花生；然后再找出剩下的植株里花生最多的，去采摘它的花生；依此类推，不过你一定要在我限定的时间内回到路边。”

我们假定多多在每个单位时间内，可以做下列四件事情中的一件：

1) 从路边跳到最靠近路边（即第一行）的某棵花生植株；
2) 从一棵植株跳到前后左右与之相邻的另一棵植株；
3) 采摘一棵植株下的花生；
4) 从最靠近路边（即第一行）的某棵花生植株跳回路边。

![img](http://media.openjudge.cn/images/upload/1446616081.jpg)

现在给定一块花生田的大小和花生的分布，请问在限定时间内，多多最多可以采到多少个花生？注意可能只有部分植株下面长有花生，假设这些植株下的花生个数各不相同。

例如在图2所示的花生田里，只有位于(2, 5), (3, 7), (4, 2), (5, 4)的植株下长有花生，个数分别为13, 7, 15, 9。沿着图示的路线，多多在21个单位时间内，最多可以采到37个花生。

输入

第一行包括三个整数，M, N和K，用空格隔开；表示花生田的大小为M * N（1 <= M, N <= 20），多多采花生的限定时间为K（0 <= K <= 1000）个单位时间。接下来的M行，每行包括N个非负整数，也用空格隔开；第i + 1行的第j个整数Pij（0 <= Pij <= 500）表示花生田里植株(i, j)下花生的数目，0表示该植株下没有花生。

输出

包括一行，这一行只包含一个整数，即在限定时间内，多多最多可以采到花生的个数。

样例输入

```
样例 #1：
6 7 21
0 0 0 0 0 0 0
0 0 0 0 13 0 0
0 0 0 0 0 0 7
0 15 0 0 0 0 0
0 0 0 9 0 0 0
0 0 0 0 0 0 0

样例 #2：
6 7 20
0 0 0 0 0 0 0
0 0 0 0 13 0 0
0 0 0 0 0 0 7
0 15 0 0 0 0 0
0 0 0 9 0 0 0
0 0 0 0 0 0 0
```

样例输出

```
样例 #1：
37

样例 #2：
28
```

来源

NOIP2004复赛 普及组 第二题



```python
def max_peanuts(M, N, K, field):
    # 提取所有有花生的位置及其数量
    peanuts = []
    for i in range(M):
        for j in range(N):
            if field[i][j] > 0:
                peanuts.append((field[i][j], i, j))
    
    # 按照花生数量从大到小排序
    peanuts.sort(reverse=True, key=lambda x: x[0])
    
    # 初始化当前时间和采摘的花生总数
    current_time = 0
    total_peanuts = 0
    
    # 初始位置设为路边
    current_pos = (-1, 0)
    
    for peanut in peanuts:
        amount, x, y = peanut
        
        # 计算从当前位置到该位置的时间
        if current_pos[0] == -1:  # 从路边跳到第一行
            time_to_reach = x + 1 + abs(current_pos[1] - y)
        else:
            time_to_reach = abs(current_pos[0] - x) + abs(current_pos[1] - y)
        
        if current_pos == (-1, 0):  # 从路边跳到第一行的时间
            current_time += (x + 1)
        else:
            current_time += time_to_reach
        
        # 采摘花生需要1单位时间
        current_time += 1
        
        if current_time + x + 1 <= K:
            total_peanuts += amount
            current_pos = (x, y)
        else:
            break
    
    return total_peanuts

# 读取输入
M, N, K = map(int, input().split())
field = []
for _ in range(M):
    field.append(list(map(int, input().split())))

# 计算并输出结果
result = max_peanuts(M, N, K, field)
print(result)

```



## 01321:棋盘问题

http://cs101.openjudge.cn/practice/01321/

在一个给定形状的棋盘（形状可能是不规则的）上面摆放棋子，棋子没有区别。要求摆放时任意的两个棋子不能放在棋盘中的同一行或者同一列，请编程求解对于给定形状和大小的棋盘，摆放k个棋子的所有可行的摆放方案C。

输入

输入含有多组测试数据。
每组数据的第一行是两个正整数，n k，用一个空格隔开，表示了将在一个n*n的矩阵内描述棋盘，以及摆放棋子的数目。 n <= 8 , k <= n
当为-1 -1时表示输入结束。
随后的n行描述了棋盘的形状：每行有n个字符，其中 # 表示棋盘区域， . 表示空白区域（数据保证不出现多余的空白行或者空白列）。

输出

对于每一组数据，给出一行输出，输出摆放的方案数目C （数据保证C<2^31）。

样例输入

```
2 1
#.
.#
4 4
...#
..#.
.#..
#...
-1 -1
```

样例输出

```
2
1
```

来源

蔡错@pku





```python
# https://www.cnblogs.com/Ayanowww/p/11555193.html
'''
本题知识点：深度优先搜索 + 枚举 + 回溯

题意是要求我们把棋子放在棋盘的'#'上，但不能把两枚棋子放在同一列或者同一行上，问摆好这k枚棋子有多少种情况。

我们可以一行一行地找，当在某一行上找到一个可放入的'#'后，就开始找下一行的'#'，如果下一行没有，就再从下一行找。这样记录哪个'#'已放棋子就更简单了，只需要记录一列上就可以了。
'''
n, k, ans = 0, 0, 0
chess = [['' for _ in range(10)] for _ in range(10)]
take = [False] * 10

def dfs(h, t):
    global ans

    if t == k:
        ans += 1
        return

    if h == n:
        return

    for i in range(h, n):
        for j in range(n):
            if chess[i][j] == '#' and not take[j]:
                take[j] = True
                dfs(i + 1, t + 1)
                take[j] = False

while True:
    n, k = map(int, input().split())
    if n == -1 and k == -1:
        break

    for i in range(n):
        chess[i] = list(input())

    take = [False] * 10
    ans = 0
    dfs(0, 0)
    print(ans)
```





## 03532:最大上升子序列和

http://cs101.openjudge.cn/practice/03532/

一个数的序列bi，当b1 < b2 < ... < bS的时候，我们称这个序列是上升的。对于给定的一个序列(a1, a2, ...,aN)，我们可以得到一些上升的子序列(ai1, ai2, ..., aiK)，这里1 <= i1 < i2 < ... < iK <= N。比如，对于序列(1, 7, 3, 5, 9, 4, 8)，有它的一些上升子序列，如(1, 7), (3, 4, 8)等等。这些子序列中序列和最大为18，为子序列(1, 3, 5, 9)的和.

你的任务，就是对于给定的序列，求出最大上升子序列和。注意，最长的上升子序列的和不一定是最大的，比如序列(100, 1, 2, 3)的最大上升子序列和为100，而最长上升子序列为(1, 2, 3)

输入

输入的第一行是序列的长度N (1 <= N <= 1000)。第二行给出序列中的N个整数，这些整数的取值范围都在0到10000（可能重复）。

输出

最大上升子序列和

样例输入

```
7
1 7 3 5 9 4 8
```

样例输出

```
18
```





```python
input()
a = [int(x) for x in input().split()]

n = len(a)
dp = [0]*n

for i in range(n):
    dp[i] = a[i]
    for j in range(i):
        if a[j]<a[i]:
            dp[i] = max(dp[j]+a[i], dp[i])
    
print(max(dp))
```





## 02775:文件结构“图”

在计算机上看到文件系统的结构通常很有用。Microsoft Windows上面的"explorer"程序就是这样的一个例子。但是在有图形界面之前，没有图形化的表示方法的，那时候最好的方式是把目录和文件的结构显示成一个"图"的样子，而且使用缩排的形式来表示目录的结构。比如：



```
ROOT
|     dir1
|     file1
|     file2
|     file3
|     dir2
|     dir3
|     file1
file1
file2
```

这个图说明：ROOT目录包括三个子目录和两个文件。第一个子目录包含3个文件，第二个子目录是空的，第三个子目录包含一个文件。

输入

你的任务是写一个程序读取一些测试数据。每组测试数据表示一个计算机的文件结构。每组测试数据以'*'结尾，而所有合理的输入数据以'#'结尾。一组测试数据包括一些文件和目录的名字（虽然在输入中我们没有给出，但是我们总假设ROOT目录是最外层的目录）。在输入中,以']'表示一个目录的内容的结束。目录名字的第一个字母是'd'，文件名字的第一个字母是'f'。文件名可能有扩展名也可能没有（比如fmyfile.dat和fmyfile）。文件和目录的名字中都不包括空格,长度都不超过30。一个目录下的子目录个数和文件个数之和不超过30。

输出

在显示一个目录中内容的时候，先显示其中的子目录（如果有的话），然后再显示文件（如果有的话）。文件要求按照名字的字母表的顺序显示（目录不用按照名字的字母表顺序显示，只需要按照目录出现的先后显示）。对每一组测试数据，我们要先输出"DATA SET x:"，这里x是测试数据的编号（从1开始）。在两组测试数据之间要输出一个空行来隔开。

你需要注意的是，我们使用一个'|'和5个空格来表示出缩排的层次。

样例输入

```
file1
file2
dir3
dir2
file1
file2
]
]
file4
dir1
]
file3
*
file2
file1
*
#
```

样例输出

```
DATA SET 1:
ROOT
|     dir3
|     |     dir2
|     |     file1
|     |     file2
|     dir1
file1
file2
file3
file4

DATA SET 2:
ROOT
file1
file2
```

提示

一个目录和它的子目录处于不同的层次
一个目录和它的里面的文件处于同一层次

来源

翻译自 Pacific Northwest 1998 的试题



```python
class Node:
    def __init__(self, name):
        self.name = name
        self.dirs = []
        self.files = []

def print_structure(node, indent=0):
    prefix = '|     ' * indent
    print(prefix + node.name)
    for dir in node.dirs:
        print_structure(dir, indent + 1)
    for file in sorted(node.files):
        print(prefix + file)

dataset = 1
datas = []
temp = []
while True:
    line = input()
    if line == '#':
        break
    if line == '*':
        datas.append(temp)
        temp = []
    else:
        temp.append(line)

for data in datas:
    print(f'DATA SET {dataset}:')
    root = Node('ROOT')
    stack = [root]
    for line in data:
        if line[0] == 'd':
            dir = Node(line)
            stack[-1].dirs.append(dir)
            stack.append(dir)
        elif line[0] == 'f':
            stack[-1].files.append(line)
        elif line == ']':
            stack.pop()
    print_structure(root)
    if dataset < len(datas):
        print()
    dataset += 1
```



## 01125:Stockbroker Grapevine

http://cs101.openjudge.cn/practice/01125/

Stockbrokers are known to overreact to rumours. You have been contracted to develop a method of spreading disinformation amongst the stockbrokers to give your employer the tactical edge in the stock market. For maximum effect, you have to spread the rumours in the fastest possible way.

Unfortunately for you, stockbrokers only trust information coming from their "Trusted sources" This means you have to take into account the structure of their contacts when starting a rumour. It takes a certain amount of time for a specific stockbroker to pass the rumour on to each of his colleagues. Your task will be to write a program that tells you which stockbroker to choose as your starting point for the rumour, as well as the time it will take for the rumour to spread throughout the stockbroker community. This duration is measured as the time needed for the last person to receive the information.

输入

Your program will input data for different sets of stockbrokers. Each set starts with a line with the number of stockbrokers. Following this is a line for each stockbroker which contains the number of people who they have contact with, who these people are, and the time taken for them to pass the message to each person. The format of each stockbroker line is as follows: The line starts with the number of contacts (n), followed by n pairs of integers, one pair for each contact. Each pair lists first a number referring to the contact (e.g. a '1' means person number one in the set), followed by the time in minutes taken to pass a message to that person. There are no special punctuation symbols or spacing rules.

Each person is numbered 1 through to the number of stockbrokers. The time taken to pass the message on will be between 1 and 10 minutes (inclusive), and the number of contacts will range between 0 and one less than the number of stockbrokers. The number of stockbrokers will range from 1 to 100. The input is terminated by a set of stockbrokers containing 0 (zero) people.



输出

For each set of data, your program must output a single line containing the person who results in the fastest message transmission, and how long before the last person will receive any given message after you give it to this person, measured in integer minutes.
It is possible that your program will receive a network of connections that excludes some persons, i.e. some people may be unreachable. If your program detects such a broken network, simply output the message "disjoint". Note that the time taken to pass the message from person A to person B is not necessarily the same as the time taken to pass it from B to A, if such transmission is possible at all. 

样例输入

```
3
2 2 4 3 5
2 1 2 3 6
2 1 2 2 2
5
3 4 4 2 8 5 3
1 5 8
4 1 6 4 10 2 7 5 2
0
2 2 5 1 5
0
```

样例输出

```
3 2
3 10
```

来源

Southern African 2001





```python
# 23n2300011072(蒋子轩)
def floyd_warshall(graph):
    """
    实现Floyd-Warshall算法，找到所有顶点对之间的最短路径。
    """
    n = len(graph)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if i != j and j != k and i != k:
                    # 通过顶点i更新顶点j和k之间的最短路径
                    graph[j][k] = min(graph[j][k], graph[j][i] + graph[i][k])
    return graph

def find_best_broker(graph):
    """
    找到最佳经纪人开始传播谣言，以使其以最短时间到达所有其他人。
    """
    n = len(graph)
    mmin = float('inf')
    best_broker = -1
    for i in range(n):
        # 查找从经纪人i向所有其他人传播信息所需的最大时间
        mmax = max(graph[i])
        if mmin > mmax:
            mmin = mmax
            best_broker = i
    return best_broker, mmin


while True:
    # 读取经纪人的数量
    n = int(input())
    if n == 0:
        break

    # 用'inf'初始化图（代表无直接连接）
    graph = [[float('inf') for _ in range(n)] for _ in range(n)]
    for i in range(n):
        graph[i][i] = 0  # 经纪人将消息传递给自己的时间为0
        data = list(map(int, input().split()))
        for j in range(1, len(data), 2):
            # 用直接连接和传递消息所需的时间更新图
            graph[i][data[j] - 1] = data[j + 1]
    
    # 计算所有经纪人对之间的最短路径
    graph = floyd_warshall(graph)
    # 查找开始传播谣言的最佳经纪人
    broker, time = find_best_broker(graph)
    
    # 打印结果
    if time == float('inf'):
        print("disjoint")
    else:
        print(broker + 1, time)
```



## 05414:由中根序列和后根序列重建二叉树

http://cs101.openjudge.cn/practice/05414/

我们知道如何按照三种深度优先次序来周游一棵二叉树，来得到中根序列、前根序列和后根序列。反过来，如果给定二叉树的中根序列和后根序列，或者给定中根序列和前根序列，可以重建一二叉树。本题输入一棵二叉树的中根序列和后根序列，要求在内存中重建二叉树，最后输出这棵二叉树的前根序列。

用不同的整数来唯一标识二叉树的每一个结点，下面的二叉树

![img](http://media.openjudge.cn/images/upload/1351670567.png)

中根序列是9 5 32 67

后根序列9 32 67 5

前根序列5 9 67 32

输入

两行。第一行是二叉树的中根序列，第二行是后根序列。每个数字表示的结点之间用空格隔开。结点数字范围0～65535。暂不必考虑不合理的输入数据。

输出

一行。由输入中的中根序列和后根序列重建的二叉树的前根序列。每个数字表示的结点之间用空格隔开。

样例输入

```
9 5 32 67
9 32 67 5
```

样例输出

```
5 9 67 32
```





```python
def buildTree(inorder, postorder):
    if not inorder or not postorder:
        return []
    root = postorder[-1]
    rootIndex = inorder.index(root)
    leftInorder = inorder[:rootIndex]
    rightInorder = inorder[rootIndex+1:]
    leftPostorder = postorder[:len(leftInorder)]
    rightPostorder = postorder[len(leftInorder):-1]
    return [root] + buildTree(leftInorder, leftPostorder) + buildTree(rightInorder, rightPostorder)

inorder = list(map(int, input().split()))
postorder = list(map(int, input().split()))
preorder = buildTree(inorder, postorder)
print(' '.join(map(str, preorder)))
```





## 04090:超级备忘录

http://cs101.openjudge.cn/practice/04090/

你的朋友Jackson被邀请参加一个叫做“超级备忘录”的电视节目。在这个节目中，参与者需要玩一个记忆游戏。在一开始，主持人会告诉所有参与者一个数列，{A1, A2, ..., An}。接下来，主持人会在数列上做一些操作，操作包括以下几种：

1. ADD x y D：给子序列{Ax, ..., Ay}统一加上一个数D。例如，在{1, 2, 3, 4, 5}上进行操作"ADD 2 4 1"会得到{1, 3, 4, 5, 5}。
2. REVERSE x y：将子序列{Ax, ..., Ay}逆序排布。例如，在{1, 2, 3, 4, 5}上进行操作"REVERSE 2 4"会得到{1, 4, 3, 2, 5}。
3. REVOLVE x y T：将子序列{Ax, ..., Ay}轮换T次。例如，在{1, 2, 3, 4, 5}上进行操作"REVOLVE 2 4 2"会得到{1, 3, 4, 2, 5}。
4. INSERT x P：在Ax后面插入P。例如，在{1, 2, 3, 4, 5}上进行操作"INSERT 2 4"会得到{1, 2, 4, 3, 4, 5}。
5. DELETE x：删除Ax。在Ax后面插入P。例如，在{1, 2, 3, 4, 5}上进行操作"DELETE 2"会得到{1, 3, 4, 5}。
6. MIN x y：查询子序列{Ax, ..., Ay}中的最小值。例如，{1, 2, 3, 4, 5}上执行"MIN 2 4"的正确答案为2。

为了使得节目更加好看，每个参赛人都有机会在觉得困难时打电话请求场外观众的帮助。你的任务是看这个电视节目，然后写一个程序对于每一个询问计算出结果，这样可以使得Jackson在任何时候打电话求助你的时候，你都可以给出正确答案。

输入

第一行包含一个数n (n ≤ 100000)。
接下来n行给出了序列中的数。
接下来一行包含一个数M (M ≤ 100000)，描述操作和询问的数量。
接下来M行给出了所有的操作和询问。

输出

对于每一个"MIN"询问，输出正确答案。

样例输入

```
5
1 
2 
3 
4 
5
2
ADD 2 4 1
MIN 4 5
```

样例输出

```
5
```





```python

```

