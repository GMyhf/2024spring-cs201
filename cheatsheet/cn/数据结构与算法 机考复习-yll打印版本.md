

# 数据结构与算法 机考复习-yll

## 一  python入门

## 1模拟

#### 01936 全在其中 http://cs101.openjudge.cn/2024sp_routine/01936/

判断两个字符串是不是互为子串

```python
try:a=t.index(s[0])
except ValueError:
```

#### 04067 回文数字 http://cs101.openjudge.cn/2024sp_routine/04067/

```python
try:s=input()
except EOFError:
```

#### 02734 从十进制到八进制 http://cs101.openjudge.cn/2024sp_routine/02734/

```python
b=bin(a)
b=oct(a)
b=hex(a)
```

#### 02783 Holiday Hotel http://cs101.openjudge.cn/2024sp_routine/02783/

找出最划算的那些酒店：不存在任何一个酒店，距离和价格同时小于它

```python
h.sort(key=lambda x:(x[0],x[1]))
```

#### 04093 倒排索引查询 http://cs101.openjudge.cn/2024sp_routine/04093/

```python
d.keys()
d.value()
```

#### 04110 圣诞老人的礼物 http://cs101.openjudge.cn/2024sp_routine/04110/

输出m 保留一位小数

```python
print(f'{m:.1f}')
```

#### 07745:整数奇偶排序 http://cs101.openjudge.cn/2024sp_routine/07745/

把a从大到小排

```python
a.sort(reverse=True)
```

#### 21554:排队做实验 (greedy)v0.2 http://cs101.openjudge.cn/2024sp_routine/21554/

```python
print('%o' % 20) #八进制
24
print('%d' % 20) #十进制
20
print('%x' % 20) #十六进制
14

print('%f' % 1.11)  # 默认保留6位小数
1.110000
print('%.1f' % 1.11)  # 取1位小数
1.1
print('%e' % 1.11)  # 默认6位小数，用科学计数法
1.110000e+00
print('%.3e' % 1.11)  # 取3位小数，用科学计数法
1.110e+00
print('%g' % 1111.1111)  # 默认6位有效数字
1111.11
print('%.7g' % 1111.1111)  # 取7位有效数字
1111.111
print('%.2g' % 1111.1111)  # 取2位有效数字，自动转换为科学计数法
1.1e+03

round(1.1125)  # 四舍五入，不指定位数，取整
1
round(1.1135,3)  # 取3位小数，由于3为奇数，则向下“舍”
1.113
round(1.1125,3)  # 取3位小数，由于2为偶数，则向上“入”
1.113
round(1.5)  # 无法理解，查阅一些资料是说python会对数据进行截断，没有深究
2
round(2.5)  # 无法理解
2
round(1.675,2)  # 无法理解
1.68
round(2.675,2)  # 无法理解
2.67

print('%s' % 'hello world')  # 字符串输出
hello world
print('%20s' % 'hello world')  # 右对齐，取20位，不够则补位
hello world
print('%-20s' % 'hello world')  # 左对齐，取20位，不够则补位
hello world         
print('%.2s' % 'hello world')  # 取2位
he
print('%10.2s' % 'hello world')  # 右对齐，取2位
he
print('%-10.2s' % 'hello world')  # 左对齐，取2位
he 

```

![78181717561008_.pic](/Users/yuanlai/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/20ef5b342c28f3861251200692178fd7/Message/MessageTemp/e3257cdfaf9538554e5ffa7e5d360b8d/Image/78181717561008_.pic.jpg)

![78191717561027_.pic](/Users/yuanlai/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/20ef5b342c28f3861251200692178fd7/Message/MessageTemp/e3257cdfaf9538554e5ffa7e5d360b8d/Image/78191717561027_.pic.jpg)

#### 22271:绿水青山之植树造林活动 http://cs101.openjudge.cn/2024sp_routine/22271/

```python
print("{:.4f}".format(t))
```

#### 23563:多项式时间复杂度 http://cs101.openjudge.cn/2024sp_routine/23563/

```python
net.remove(a)
```

#### 26977:接雨水 http://cs101.openjudge.cn/2024sp_routine/26977/

```python
lr.insert(0,m2)
```

#### 27273:简单的数学题 http://cs101.openjudge.cn/2024sp_routine/27273/

```python
from math import log
log(n,2)
```

#### 27625:AVL树至少有几个结点 http://cs101.openjudge.cn/2024sp_routine/27625/

```python
from functools import lru_cache
@lru_cache(maxsize=None)
```

## 2 动态规划

#### 23997:奇数拆分 http://cs101.openjudge.cn/2024sp_routine/23997/

```python
net.extend(net_0)
import math
math.sqrt(a)
```

## 二 时间复杂度

### 1 @lru_cache

#### 02192 Zipper http://cs101.openjudge.cn/2024sp_routine/02192/

```python
from functools import lru_cache
@lru_cache
```

### 2 MergeSort

```python
def mergeSort(arr):
	if len(arr) > 1:
		mid = len(arr)//2

		L = arr[:mid]	# Dividing the array elements
		R = arr[mid:] # Into 2 halves

		mergeSort(L) # Sorting the first half
		mergeSort(R) # Sorting the second half

		i = j = k = 0
		# Copy data to temp arrays L[] and R[]
		while i < len(L) and j < len(R):
			if L[i] <= R[j]:
				arr[k] = L[i]
				i += 1
			else:
				arr[k] = R[j]
				j += 1
			k += 1

		# Checking if any element was left
		while i < len(L):
			arr[k] = L[i]
			i += 1
			k += 1

		while j < len(R):
			arr[k] = R[j]
			j += 1
			k += 1


if __name__ == '__main__':
	arr = [12, 11, 13, 5, 6, 7]
	mergeSort(arr)
	print(' '.join(map(str, arr)))
```

### 3 二分查找法

#### 02774 木材加工 http://cs101.openjudge.cn/2024sp_routine/02774/

木材厂有一些原木，现在想把这些木头切割成一些长度相同的小段木头，需要得到的小段的数目是给定了。当然，我们希望得到的小段越长越好，你的任务是计算能够得到的小段木头的最大长度。

木头长度的单位是厘米。原木的长度都是正整数，我们要求切割得到的小段木头的长度也要求是正整数。

**输入**

第一行是两个正整数*N*和*K*(1 ≤ *N* ≤ 10000, 1 ≤ *K* ≤ 10000)，*N*是原木的数目，*K*是需要得到的小段的数目。
接下来的*N*行，每行有一个1到10000之间的正整数，表示一根原木的长度。
　

**输出**

输出能够切割得到的小段的最大长度。如果连1厘米长的小段都切不出来，输出"0"。

样例输入

```
3 7
232
124
456
```

样例输出

```
114
```

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



### 4 T-prime问题

#### 18176:2050年成绩计算 http://cs101.openjudge.cn/2024sp_routine/18176/

```python
p=[True]*10001
for x in range(2,101):
    d=x**2
    if p[x]:
        while d<10001:
            p[d]=False
            d+=x
```

## 三 线性数据结构

### 1 栈实现

#### 1.回溯

##### 01321 棋盘问题 http://cs101.openjudge.cn/2024sp_routine/01321/

描述

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
        print(solve_n_queens(a,b,net))
```

##### 02754 八皇后问题 http://cs101.openjudge.cn/2024sp_routine/solution/44929849/

描述：会下国际象棋的人都很清楚：皇后可以在横、竖、斜线上不限步数地吃掉其他棋子。如何将8个皇后放在棋盘上（有8 * 8个方格），使它们谁也不能被吃掉！这就是著名的八皇后问题。 对于某个满足要求的8皇后的摆放方法，定义一个皇后串a与之对应，即$a=b_1b_2...b_8~$,其中$b_i$为相应摆法中第i行皇后所处的列数。已经知道8皇后问题一共有92组解（即92个不同的皇后串）。 给出一个数b，要求输出第b个串。串的比较是这样的：皇后串x置于皇后串y之前，当且仅当将x视为整数时比y小。

 八皇后是一个古老的经典问题：**如何在一张国际象棋的棋盘上，摆放8个皇后，使其任意两个皇后互相不受攻击。该问题由一位德国国际象棋排局家** **Max Bezzel** 于 1848年提出。严格来说，那个年代，还没有“德国”这个国家，彼时称作“普鲁士”。1850年，**Franz Nauck** 给出了第一个解，并将其扩展成了“ **n皇后** ”问题，即**在一张 n** x **n 的棋盘上，如何摆放 n 个皇后，使其两两互不攻击**。历史上，八皇后问题曾惊动过“数学王子”高斯(Gauss)，而且正是 Franz Nauck 写信找高斯请教的。

**输入**

第1行是测试数据的组数n，后面跟着n行输入。每组测试数据占1行，包括一个正整数b(1 ≤  b ≤  92)

**输出**

输出有n行，每行输出对应一个输入。输出应是一个正整数，是对应于b的皇后串。

样例输入

```
2
1
92
```



样例输出

```
15863724
84136275
```



先给出两个dfs回溯实现的八皇后，接着给出两个stack迭代实现的八皇后。

八皇后思路：回溯算法通过尝试不同的选择，逐步构建解决方案，并在达到某个条件时进行回溯，以找到所有的解决方案。从第一行第一列开始放置皇后，然后在每一行的不同列都放置，如果与前面不冲突就继续，有冲突则回到上一行继续下一个可能性。

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

##### 24591:中序表达式转后序表达式 http://cs101.openjudge.cn/2024sp_routine/solution/44077911/

中序表达式是运算符放在两个数中间的表达式。乘、除运算优先级高于加减。可以用"()"来提升优先级 --- 就是小学生写的四则算术运算表达式。中序表达式可用如下方式递归定义：

1）一个数是一个中序表达式。该表达式的值就是数的值。

1. 若a是中序表达式，则"(a)"也是中序表达式(引号不算)，值为a的值。
2. 若a,b是中序表达式，c是运算符，则"acb"是中序表达式。"acb"的值是对a和b做c运算的结果，且a是左操作数，b是右操作数。

输入一个中序表达式，要求转换成一个后序表达式输出。

**输入**

第一行是整数n(n<100)。接下来n行，每行一个中序表达式，数和运算符之间没有空格，长度不超过700。

**输出**

对每个中序表达式，输出转成后序表达式后的结果。后序表达式的数之间、数和运算符之间用一个空格分开。

样例输入

```
3
7+8.3 
3+4.5*(7+2)
(3)*((3+4)*(2+3.5)/(4+5)) 
```



样例输出

```
7 8.3 +
3 4.5 7 2 + * +
3 3 4 + 2 3.5 + * 4 5 + / *
```

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

##### 24588:后序表达式求值 http://cs101.openjudge.cn/2024sp_routine/solution/44076307/

后序表达式由操作数和运算符构成。操作数是整数或小数，运算符有 + - * / 四种，其中 * / 优先级高于 + -。后序表达式可用如下方式递归定义：

1. 一个操作数是一个后序表达式。该表达式的值就是操作数的值。
2. 若a,b是后序表达式，c是运算符，则"a b c"是后序表达式。“a b c”的值是 (a) c (b),即对a和b做c运算，且a是第一个操作数，b是第二个操作数。下面是一些后序表达式及其值的例子(操作数、运算符之间用空格分隔)：

3.4 值为：3.4 5 值为：5 5 3.4 + 值为：5 + 3.4 5 3.4 + 6 / 值为：(5+3.4)/6 5 3.4 + 6 * 3 + 值为：(5+3.4)*6+3

**输入**

第一行是整数n(n<100)，接下来有n行，每行是一个后序表达式，长度不超过1000个字符

**输出**

对每个后序表达式，输出其值，保留小数点后面2位

样例输入

```
3
5 3.4 +
5 3.4 + 6 /
5 3.4 + 6 * 3 +
```



样例输出

```
8.40
1.40
53.40
```



来源: Guo wei

要解决这个问题，需要理解如何计算后序表达式。后序表达式的计算可以通过使用一个栈来完成，按照以下步骤：

1. 从左到右扫描后序表达式。
2. 遇到数字时，将其压入栈中。
3. 遇到运算符时，从栈中弹出两个数字，先弹出的是右操作数，后弹出的是左操作数。将这两个数字进行相应的运算，然后将结果压入栈中。
4. 当表达式扫描完毕时，栈顶的数字就是表达式的结果。

```python
def evaluate_postfix(expression):
    stack = []
    tokens = expression.split()
    
    for token in tokens:
        if token in '+-*/':
            # 弹出栈顶的两个元素
            right_operand = stack.pop()
            left_operand = stack.pop()
            # 执行运算
            if token == '+':
                stack.append(left_operand + right_operand)
            elif token == '-':
                stack.append(left_operand - right_operand)
            elif token == '*':
                stack.append(left_operand * right_operand)
            elif token == '/':
                stack.append(left_operand / right_operand)
        else:
            # 将操作数转换为浮点数后入栈
            stack.append(float(token))
    
    # 栈顶元素就是表达式的结果
    return stack[0]

# 读取输入行数
n = int(input())

# 对每个后序表达式求值
for _ in range(n):
    expression = input()
    result = evaluate_postfix(expression)
    # 输出结果，保留两位小数
    print(f"{result:.2f}")
```

### 2 双端队列

#### 05902 双端队列 http://cs101.openjudge.cn/2024sp_routine/05902/

定义一个双端队列，进队操作与普通队列一样，从队尾进入。出队操作既可以从队头，也可以从队尾。编程实现这个数据结构。

**输入** 第一行输入一个整数t，代表测试数据的组数。 每组数据的第一行输入一个整数n，表示操作的次数。 接着输入n行，每行对应一个操作，首先输入一个整数type。 当type=1，进队操作，接着输入一个整数x，表示进入队列的元素。 当type=2，出队操作，接着输入一个整数c，c=0代表从队头出队，c=1代表从队尾出队。 n <= 1000

**输出** 对于每组测试数据，输出执行完所有的操作后队列中剩余的元素,元素之间用空格隔开，按队头到队尾的顺序输出，占一行。如果队列中已经没有任何的元素，输出NULL。

样例输入

```
2
5
1 2
1 3
1 4
2 0
2 1
6
1 1
1 2
1 3
2 0
2 1
2 0
```



样例输出

```
3
NULL
```



```python
from collections import deque

for _ in range(int(input())):
    n=int(input())
    q=deque([])
    for i in range(n):
        a,b=map(int,input().split())
        if a==1:
            q.append(b)
        else:
            if b==0:
                q.popleft()
            else:
                q.pop()
    if q:
        print(*q)
    else:
        print('NULL')
```

### 3 查询问题

#### 

## 三 树

### 1 树的表示方法

#### 24729: 括号嵌套树

http://cs101.openjudge.cn/practice/24729/

可以用括号嵌套的方式来表示一棵树。表示方法如下：

1. 如果一棵树只有一个结点，则该树就用一个大写字母表示，代表其根结点。
2. 如果一棵树有子树，则用“树根(子树1,子树2,...,子树n)”的形式表示。树根是一个大写字母，子树之间用逗号隔开，没有空格。子树都是用括号嵌套法表示的树。

给出一棵不超过26个结点的树的括号嵌套表示形式，请输出其前序遍历序列和后序遍历序列。

输入样例代表的树如下图：

[![img](https://camo.githubusercontent.com/381bcd178cc5ef03507c7526c4738e95e168791f072079091c89716d45c63679/687474703a2f2f6d656469612e6f70656e6a756467652e636e2f696d616765732f75706c6f61642f353830352f313635333437323137332e706e67)](https://camo.githubusercontent.com/381bcd178cc5ef03507c7526c4738e95e168791f072079091c89716d45c63679/687474703a2f2f6d656469612e6f70656e6a756467652e636e2f696d616765732f75706c6f61642f353830352f313635333437323137332e706e67)

**输入**

一行，一棵树的括号嵌套表示形式

**输出**

两行。第一行是树的前序遍历序列，第二行是树的后序遍历序列

样例输入

```
A(B(E),C(F,G),D(H(I)))
```



样例输出

```
ABECFGDHI
EBFGCIHDA
```



来源：Guo Wei

题面提到了遍历，但是没有给出定义。定义在3.2 树的遍历 一节。

下面两个代码。先给出用类表示node

```python
class TreeNode:
    def __init__(self, value): #类似字典
        self.value = value
        self.children = []

def parse_tree(s):
    stack = []
    node = None
    for char in s:
        if char.isalpha():  # 如果是字母，创建新节点
            node = TreeNode(char)
            if stack:  # 如果栈不为空，把节点作为子节点加入到栈顶节点的子节点列表中
                stack[-1].children.append(node)
        elif char == '(':  # 遇到左括号，当前节点可能会有子节点
            if node:
                stack.append(node)  # 把当前节点推入栈中
                node = None
        elif char == ')':  # 遇到右括号，子节点列表结束
            if stack:
                node = stack.pop()  # 弹出当前节点
    return node  # 根节点


def preorder(node):
    output = [node.value]
    for child in node.children:
        output.extend(preorder(child))
    return ''.join(output)

def postorder(node):
    output = []
    for child in node.children:
        output.extend(postorder(child))
    output.append(node.value)
    return ''.join(output)

# 主程序
def main():
    s = input().strip()
    s = ''.join(s.split())  # 去掉所有空白字符
    root = parse_tree(s)  # 解析整棵树
    if root:
        print(preorder(root))  # 输出前序遍历序列
        print(postorder(root))  # 输出后序遍历序列
    else:
        print("input tree string error!")

if __name__ == "__main__":
    main()
```



#### 02775 文件结构“图” http://cs101.openjudge.cn/2024sp_routine/02775/

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

**输入**

你的任务是写一个程序读取一些测试数据。每组测试数据表示一个计算机的文件结构。每组测试数据以`*`结尾，而所有合理的输入数据以`#`结尾。一组测试数据包括一些文件和目录的名字（虽然在输入中我们没有给出，但是我们总假设ROOT目录是最外层的目录）。在输入中,以`]`表示一个目录的内容的结束。目录名字的第一个字母是'd'，文件名字的第一个字母是`f`。文件名可能有扩展名也可能没有（比如`fmyfile.dat`和`fmyfile`）。文件和目录的名字中都不包括空格,长度都不超过30。一个目录下的子目录个数和文件个数之和不超过30。

**输出**

在显示一个目录中内容的时候，先显示其中的子目录（如果有的话），然后再显示文件（如果有的话）。文件要求按照名字的字母表的顺序显示（目录不用按照名字的字母表顺序显示，只需要按照目录出现的先后显示）。对每一组测试数据，我们要先输出`DATA SET x:`，这里`x`是测试数据的编号（从1开始）。在两组测试数据之间要输出一个空行来隔开。

你需要注意的是，我们使用一个`|`和5个空格来表示出缩排的层次。

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

一个目录和它的子目录处于不同的层次 一个目录和它的里面的文件处于同一层次

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

### 2 二叉树

#### 18164: 剪绳子



greedy/huffman, http://cs101.openjudge.cn/practice/18164/

小张要将一根长度为L的绳子剪成N段。准备剪的绳子的长度为L1,L2,L3...,LN，未剪的绳子长度恰好为剪后所有绳子长度的和。

每次剪断绳子时，需要的开销是此段绳子的长度。

比如，长度为10的绳子要剪成长度为2,3,5的三段绳子。长度为10的绳子切成5和5的两段绳子时，开销为10。再将5切成长度为2和3的绳子，开销为5。因此总开销为15。

请按照目标要求将绳子剪完最小的开销时多少。

已知，1<=N <= 20000，0<=Li<= 50000

**输入**

第一行：N，将绳子剪成的段数。 第二行：准备剪成的各段绳子的长度。

**输出**

最小开销

样例输入

```
3
2 3 5
```



样例输出

```
15
```

```python
import sys
try: fin = open('test.in','r').readline
except: fin = sys.stdin.readline

n = int(fin())
import heapq
a = list(map(int, fin().split()))
heapq.heapify(a)
ans = 0
for i in range(n-1):
    x = heapq.heappop(a)
    y = heapq.heappop(a)
    z = x + y
    heapq.heappush(a, z)
    ans += z
print(ans)
```

### 3 二叉树的遍历

#### 22158:根据二叉树前中序序列建树 http://cs101.openjudge.cn/2024sp_routine/22158/

假设二叉树的节点里包含一个大写字母，每个节点的字母都不同。

给定二叉树的前序遍历序列和中序遍历序列(长度均不超过26)，请输出该二叉树的后序遍历序列

**输入**

多组数据 每组数据2行，第一行是前序遍历序列，第二行是中序遍历序列

**输出**

对每组序列建树，输出该树的后序遍历序列

样例输入

```
DURPA
RUDPA
XTCNB
CTBNX
```



样例输出

```
RUAPD
CBNTX
```



来源：郭炜

The problem is asking to construct a binary tree from given preorder and inorder traversal sequences, and then output the postorder traversal sequence of the constructed tree.

Here is a step-by-step plan:
1.Create a TreeNode class to represent each node in the tree. 2.Create a function build_tree that takes the preorder and inorder sequences as input and returns the root of the constructed tree. The first character of the preorder sequence is the root of the tree. Find the position of the root in the inorder sequence. Recursively construct the left subtree using the left part of the inorder sequence and the corresponding part of the preorder sequence. Recursively construct the right subtree using the right part of the inorder sequence and the corresponding part of the preorder sequence. 3.Create a function postorder_traversal that takes the root of the tree as input and returns the postorder traversal sequence of the tree. 4.For each pair of preorder and inorder sequences, construct the tree and output the postorder traversal sequence. Here is the Python code that implements this plan:

```python
class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

def build_tree(preorder, inorder):
    if not preorder or not inorder:
        return None
    root_value = preorder[0]
    root = TreeNode(root_value)
    root_index_inorder = inorder.index(root_value)
    root.left = build_tree(preorder[1:1+root_index_inorder], inorder[:root_index_inorder])
    root.right = build_tree(preorder[1+root_index_inorder:], inorder[root_index_inorder+1:])
    return root

def postorder_traversal(root):
    if root is None:
        return ''
    return postorder_traversal(root.left) + postorder_traversal(root.right) + root.value

while True:
    try:
        preorder = input().strip()
        inorder = input().strip()
        root = build_tree(preorder, inorder)
        print(postorder_traversal(root))
    except EOFError:
        break
```

### 4 树的遍历

```python
class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

def build_tree(preorder, inorder):
    if not preorder or not inorder:
        return None
    root_value = preorder[0]
    root = TreeNode(root_value)
    root_index_inorder = inorder.index(root_value)
    root.left = build_tree(preorder[1:1+root_index_inorder], inorder[:root_index_inorder])
    root.right = build_tree(preorder[1+root_index_inorder:], inorder[root_index_inorder+1:])
    return root

def postorder_traversal(root):
    if root is None:
        return ''
    return postorder_traversal(root.left) + postorder_traversal(root.right) + root.value

while True:
    try:
        preorder = input().strip()
        inorder = input().strip()
        root = build_tree(preorder, inorder)
        print(postorder_traversal(root))
    except EOFError:
        break
```

### 5 并查集

#### 07734:虫子的生活 http://cs101.openjudge.cn/2024sp_routine/07734/

Hopper 博士正在研究一种罕见种类的虫子的性行为。他假定虫子只表现为两种性别，并且虫子只与异性进行交互。在他的实验中，不同的虫子个体和虫子的交互行为很容易区分开来，因为这些虫子的背上都被标记有一些标号。

现在给定一系列的虫子的交互，现在让你判断实验的结果是否验证了他的关于没有同性恋的虫子的假设或者是否存在一些虫子之间的交互证明假设是错的。

**输入**

输入的第一行包含实验的组数。每组实验数据第一行是虫子的个数（至少1个，最多2000个) 和交互的次数 (最多1000000次) ，以空格间隔. 在下面的几行中,每次交互通过给出交互的两个虫子的标号来表示，标号之间以空格间隔。已知虫子从1开始连续编号。

**输出**

每组测试数据的输出为2行，第一行包含 "Scenario #i:", 其中 i 是实验数据组数的标号，从1开始,第二行为 "No suspicious bugs found!" 如果实验结果和博士的假设相符,或 "Suspicious bugs found!" 如果Hopper的博士的假设是错误的

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

### 6 huffman树

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

#### 18164:剪绳子 http://cs101.openjudge.cn/2024sp_routine/18164/

### 7 二叉搜索树

#### 22275 二叉搜索树的遍历 http://cs101.openjudge.cn/2024sp_routine/22275/

**输入**

第一行一个正整数n（n<=2000）表示这棵二叉搜索树的结点个数 第二行n个正整数，表示这棵二叉搜索树的前序遍历 保证第二行的n个正整数中，1~n的每个值刚好出现一次

**输出**

一行n个正整数，表示这棵二叉搜索树的后序遍历

样例输入

```
5
4 2 1 3 5
```



样例输出

```
1 3 2 5 4
```



提示

树的形状为 4
/ \ 2 5 / \
1 3

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

### AVL

##### 晴问9.5: 平衡二叉树的建立



https://sunnywhy.com/sfbj/9/5/359

将 n 个互不相同的正整数先后插入到一棵空的AVL树中，求最后生成的AVL树的先序序列。

**输入**

第一行一个整数 𝑛(1≤𝑛≤50)，表示AVL树的结点个数；

第二行 n 个整数$a_i (1 \le a_i \le 100)$，表示表示插入序列。

**输出**

输出 n 个整数，表示先序遍历序列，中间用空格隔开，行末不允许有多余的空格。

样例1

输入

```
5
5 2 3 6 8
```



输出

```
3 2 6 5 8
```



解释

插入的过程如下图所示。

[![平衡二叉树的建立.png](https://raw.githubusercontent.com/GMyhf/img/main/img/202403210041932.png)](https://raw.githubusercontent.com/GMyhf/img/main/img/202403210041932.png)

To solve this problem, you can follow these steps:

1. Read the input sequence.
2. Insert the values into an AVL tree. An AVL tree is a self-balancing binary search tree, and the heights of the two child subtrees of any node differ by at most one.
3. Perform a preorder traversal of the AVL tree and print the result.

Here is the Python code that implements this plan:

```
class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        self.height = 1

class AVL:
    def __init__(self):
        self.root = None

    def insert(self, value):
        if not self.root:
            self.root = Node(value)
        else:
            self.root = self._insert(value, self.root)

    def _insert(self, value, node):
        if not node:
            return Node(value)
        elif value < node.value:
            node.left = self._insert(value, node.left)
        else:
            node.right = self._insert(value, node.right)

        node.height = 1 + max(self._get_height(node.left), self._get_height(node.right))

        balance = self._get_balance(node)

        if balance > 1:
            if value < node.left.value:	# 树形是 LL
                return self._rotate_right(node)
            else:	# 树形是 LR
                node.left = self._rotate_left(node.left)
                return self._rotate_right(node)

        if balance < -1:
            if value > node.right.value:	# 树形是 RR
                return self._rotate_left(node)
            else:	# 树形是 RL
                node.right = self._rotate_right(node.right)
                return self._rotate_left(node)

        return node

    def _get_height(self, node):
        if not node:
            return 0
        return node.height

    def _get_balance(self, node):
        if not node:
            return 0
        return self._get_height(node.left) - self._get_height(node.right)

    def _rotate_left(self, z):
        y = z.right
        T2 = y.left
        y.left = z
        z.right = T2
        z.height = 1 + max(self._get_height(z.left), self._get_height(z.right))
        y.height = 1 + max(self._get_height(y.left), self._get_height(y.right))
        return y

    def _rotate_right(self, y):
        x = y.left
        T2 = x.right
        x.right = y
        y.left = T2
        y.height = 1 + max(self._get_height(y.left), self._get_height(y.right))
        x.height = 1 + max(self._get_height(x.left), self._get_height(x.right))
        return x

    def preorder(self):
        return self._preorder(self.root)

    def _preorder(self, node):
        if not node:
            return []
        return [node.value] + self._preorder(node.left) + self._preorder(node.right)

n = int(input().strip())
sequence = list(map(int, input().strip().split()))

avl = AVL()
for value in sequence:
    avl.insert(value)

print(' '.join(map(str, avl.preorder())))
```



This code reads the sequence from the input, inserts its values into an AVL tree, performs a preorder traversal of the AVL tree, and then prints the result.

#### 6.2.2 AVL树中删除节点



要实现从AVL树中删除节点，需要添加一个删除方法，并确保在删除节点后重新平衡树。

下面是更新后的代码，包括删除方法 `_delete`：

```
class AVL:
    # Existing code...
    
    def delete(self, value):
        self.root = self._delete(value, self.root)

    def _delete(self, value, node):
        if not node:
            return node

        if value < node.value:
            node.left = self._delete(value, node.left)
        elif value > node.value:
            node.right = self._delete(value, node.right)
        else:
            if not node.left:
                temp = node.right
                node = None
                return temp
            elif not node.right:
                temp = node.left
                node = None
                return temp

            temp = self._min_value_node(node.right)
            node.value = temp.value
            node.right = self._delete(temp.value, node.right)

        if not node:
            return node

        node.height = 1 + max(self._get_height(node.left), self._get_height(node.right))

        balance = self._get_balance(node)

        # Rebalance the tree
        if balance > 1:
            if self._get_balance(node.left) >= 0:
                return self._rotate_right(node)
            else:
                node.left = self._rotate_left(node.left)
                return self._rotate_right(node)

        if balance < -1:
            if self._get_balance(node.right) <= 0:
                return self._rotate_left(node)
            else:
                node.right = self._rotate_right(node.right)
                return self._rotate_left(node)

        return node

    def _min_value_node(self, node):
        current = node
        while current.left:
            current = current.left
        return current

    # Existing code...
```



这段代码中的 `_delete` 方法用于删除节点。它首先检查树中是否存在要删除的节点，然后根据节点的左右子树情况执行相应的操作，以保持AVL树的平衡。

在 AVL 树中，删除节点时，当被删除的节点有两个子节点时，需要一些额外的步骤来保持树的平衡性。让我们详细讲解 `else` 分支中的情况：

```
else:
    if not node.left:
        temp = node.right
        node = None
        return temp
    elif not node.right:
        temp = node.left
        node = None
        return temp

    temp = self._min_value_node(node.right)
    node.value = temp.value
    node.right = self._delete(temp.value, node.right)
```



1. 如果要删除的节点 `node` 没有左子节点，那么我们只需返回其右子节点。这是因为右子节点（如果存在）将占据 `node` 的位置，而不会影响树的平衡性。所以我们将 `node` 设置为 `None`，然后返回其右子节点即可。
2. 如果要删除的节点 `node` 没有右子节点，那么我们只需返回其左子节点。这与上述情况类似。
3. 如果要删除的节点 `node` 既有左子节点又有右子节点，那么我们需要找到 `node` 的右子树中的最小值节点，并将其值替换到 `node` 中，然后在右子树中删除这个最小值节点。这是因为右子树中的最小值节点是大于左子树中所有节点值且小于右子树中所有节点值的节点，它在替代被删除节点后能够保持树的平衡性。

函数 `_min_value_node` 用于找到树中的最小值节点，其实现如下：

```
def _min_value_node(self, node):
    current = node
    while current.left:
        current = current.left
    return current
```



这样，当我们删除带有两个子节点的节点时，我们选择将右子树中的最小值节点的值替换到要删除的节点中，然后递归地在右子树中删除这个最小值节点。

## 四 图

### 1 bfs算法

#### 28046:词梯 http://cs101.openjudge.cn/2024sp_routine/28046/

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

### 2 dfs算法

#### 28170:算鹰 http://cs101.openjudge.cn/2024sp_routine/28170/

```python
def dfs(x,y):
    graph[x][y] = "-"
    for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
        if 0<=x+dx<10 and 0<=y+dy<10 and graph[x+dx][y+dy] == ".":
            dfs(x+dx,y+dy)
graph = []
result = 0
for i in range(10):
    graph.append(list(input()))
for i in range(10):
    for j in range(10):
        if graph[i][j] == ".":
            result += 1
            dfs(i,j)
print(result)
```



### 2 拓扑排序

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

### 3 dijkstra算法

假设二叉树的节点里包含一个大写字母，每个节点的字母都不同。

给定二叉树的前序遍历序列和中序遍历序列(长度均不超过26)，请输出该二叉树的后序遍历序列

**输入**

多组数据 每组数据2行，第一行是前序遍历序列，第二行是中序遍历序列

**输出**

对每组序列建树，输出该树的后序遍历序列

样例输入

```
DURPA
RUDPA
XTCNB
CTBNX
```



样例输出

```
RUAPD
CBNTX
```



来源：郭炜

The problem is asking to construct a binary tree from given preorder and inorder traversal sequences, and then output the postorder traversal sequence of the constructed tree.

Here is a step-by-step plan:
1.Create a TreeNode class to represent each node in the tree. 2.Create a function build_tree that takes the preorder and inorder sequences as input and returns the root of the constructed tree. The first character of the preorder sequence is the root of the tree. Find the position of the root in the inorder sequence. Recursively construct the left subtree using the left part of the inorder sequence and the corresponding part of the preorder sequence. Recursively construct the right subtree using the right part of the inorder sequence and the corresponding part of the preorder sequence. 3.Create a function postorder_traversal that takes the root of the tree as input and returns the postorder traversal sequence of the tree. 4.For each pair of preorder and inorder sequences, construct the tree and output the postorder traversal sequence. Here is the Python code that implements this plan:

```python
class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

def build_tree(preorder, inorder):
    if not preorder or not inorder:
        return None
    root_value = preorder[0]
    root = TreeNode(root_value)
    root_index_inorder = inorder.index(root_value)
    root.left = build_tree(preorder[1:1+root_index_inorder], inorder[:root_index_inorder])
    root.right = build_tree(preorder[1+root_index_inorder:], inorder[root_index_inorder+1:])
    return root

def postorder_traversal(root):
    if root is None:
        return ''
    return postorder_traversal(root.left) + postorder_traversal(root.right) + root.value

while True:
    try:
        preorder = input().strip()
        inorder = input().strip()
        root = build_tree(preorder, inorder)
        print(postorder_traversal(root))
    except EOFError:
        break
```

#### 01724 ROADS http://cs101.openjudge.cn/2024sp_routine/01724/

样例输入

```
5
6
7
1 2 2 3
2 4 3 3
3 4 2 4
1 3 4 1
4 6 2 1
3 5 2 0
5 4 3 2
```

样例输出

```
11
```

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

## 其他

#### 27778:MD5加密验证系统 http://cs101.openjudge.cn/2024sp_routine/27778/

哈希算法

描述

在数字安全领域，MD5加密是一种广泛使用的哈希算法，用于将任意长度的数据“压缩”成128位的加密串（通常表示为32位的十六进制数）。尽管MD5因安全漏洞而不再推荐用于敏感加密场合，它在教学和非安全领域的应用仍然广泛。你的任务是实现一个MD5加密验证系统，用于比较两串文本是否具有相同的MD5加密值。

输入

首先输入一个整数T，表示接下来有T组输入，其中T小于等于10。
接着是T组输入，每组输入包含两行，分别代表两串需要进行MD5加密比较的文本。每行文本的长度不超过1000个字符。

输出

对于每组输入，输出一行结果。如果两串文本的MD5加密值相同，则输出"Yes"；否则输出"No"。

样例输入

```
2
helloworld
worldhello
helloworld
helloworld
```

样例输出

```
No
Yes
```

```python
import hashlib

def f(x,y):
    mdx=hashlib.md5(x.encode())
    md5x=mdx.hexdigest()
    mdy=hashlib.md5(y.encode())
    md5y=mdy.hexdigest()
    if md5x==md5y:
        return 'Yes'
    else:
        return 'No'

a=int(input())
for i in range(a):
    m=input()
    n=input()
    print(f(m,n))
```

## 06263:布尔表达式

- 描述

  输入一个布尔表达式，请你输出它的真假值。  比如：( V | V ) & F & ( F | V )  V表示true，F表示false，&表示与，|表示或，!表示非。  上式的结果是F 

- 输入

  输入包含多行，每行一个布尔表达式，表达式中可以有空格，总长度不超过1000

- 输出

  对每行输入，如果表达式为真，输出"V",否则出来"F"

- 样例输入

  `( V | V ) & F & ( F| V) !V | V & V & !F & (F | V ) & (!F | F | !V & V) (F&F|V|!V&!F&!(F|F&V))`

- 样例输出

  `F V V`

```python
while True:
    try:
        s = input()
        s0 = s.replace("!", " not ").replace("|", " or ").replace("&", " and ")
        s1 = s0.replace("V", "True").replace("F", "False")
        ans = eval(s1)
        if ans:
            print("V")
        else:
            print("F")
    except EOFError:
        break
```



#### 递归次数扩展：

```python
import sys
sys.setrecursionlimit(100000)
```



## 总结

打印版本汇集了一些笔者认为最容易错的简单代码块，以及一些容易忘记的基本内置函数的用法，一些难题笔者加上了题干以便考场上理解和抄写。