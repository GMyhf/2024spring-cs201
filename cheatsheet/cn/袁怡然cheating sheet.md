# **队列、链表、栈**
## **！调度场算法**
shunting yard -->典型 中序表达式转为后序表达式
主要思想是使用两个栈（运算符栈和输出栈）来处理表达式的符号。算法按照运算符的优先级和结合性，将符号逐个处理并放置到正确的位置。最终，输出栈中的元素就是转换后的后缀表达式。

## **中序表达式转后序表达式**
```
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
## **！重点 单调栈**

## **八皇后**
（dfs回溯实现或stack迭代实现）
### dfs方法
def solve_n_queens(n):
    solutions = []  # 存储所有解决方案的列表
    queens = [-1] * n  # 存储每一行皇后所在的列数
    def backtrack(row):
        if row == n:  # 找到一个合法解决方案
            solutions.append(queens.copy())
        else:
            for col in range(n):
                if is_valid(row, col):  # 检查当前位置是否合法
                    queens[row] = col  # 在当前行放置皇后
                    backtrack(row + 1)  # 递归处理下一行
                    queens[row] = -1  # 回溯，撤销当前行的选择
    def is_valid(row, col):
        for r in range(row):
            if queens[r] == col or abs(row - r) == abs(col - queens[r]):
                return False
        return True
    backtrack(0)  # 从第一行开始回溯
    return solutions
#获取第 b 个皇后串
def get_queen_string(b):
    solutions = solve_n_queens(8)
    if b > len(solutions):
        return None
    queen_string = ''.join(str(col + 1) for col in solutions[b - 1])
    return queen_string

### stack迭代方法
def solve_n_queens(n):
    stack = []  # 用于保存状态的栈
    solutions = []  # 存储所有解决方案的列表
    stack.append((0, [-1] * n))  # 初始状态为第一行，所有列都未放置皇后
    while stack:
        row, queens = stack.pop()
        if row == n:  # 找到一个合法解决方案
            solutions.append(queens.copy())
        else:
            for col in range(n):
                if is_valid(row, col, queens):  # 检查当前位置是否合法
                    new_queens = queens.copy()
                    new_queens[row] = col  # 在当前行放置皇后
                    stack.append((row + 1, new_queens))  # 推进到下一行
    return solutions
def is_valid(row, col, queens):

## **合法出栈序列**
给定一个由大小写字母和数字构成的，没有重复字符的长度不超过62的字符串x，现在要将该字符串的字符依次压入栈中，然后再全部弹出。
要求左边的字符一定比右边的字符先入栈，出栈顺序无要求。
再给定若干字符串，对每个字符串，判断其是否是可能的x中的字符的出栈序列
```
def isPopSeq(s1,s2):#判断s2是不是s1经出入栈得到的出栈序列
	stack = []
	if len(s1) != len(s2):
		return False
	else:
		L = len(s1)
		stack.append(s1[0])
		p1,p2 = 1,0
		while p1 < L:
			if len(stack) > 0 and stack[-1] == s2[p2]:
				stack.pop()
				p2 += 1
			else:
				stack.append(s1[p1])
				p1 += 1
		return "".join(stack[::-1]) == s2[p2:]
```

## **约瑟夫**
**（双端队列法，或双向链表）**
n 个小孩围坐成一圈，并按顺时针编号为1,2,…,n，从编号为 p 的小孩顺时针依次报数，由1报到m ，当报到 m 时，该小孩从圈中出去，然后下一个再从1报数，当报到 m 时再出去。如此反复，直至所有的小孩都从圈中出去。请按出去的先后顺序输出小孩的编号。
```
while True:
    n,p,m=map(int,input().split())
    if {n,p,m}=={0}:
        break
    monkey=[i for i in range(1,n+1)]
    for _ in range(p-1):
        monkey.append(monkey.pop(0))
    index=0
    res=[]
    while len(monkey)!=1:
        temp=monkey.pop(0) #将不用取出的元素存入暂时列表中，等取出第m个元素后将这些加入原列表中
        index+=1
        if index==m:
            index=0
            res.append(temp)
            continue
        monkey.append(temp)
    res.extend(monkey)
    print(','.join(map(str,res)))
```
## **埃拉托斯特尼筛法**
可以先找到n可能的最大值，打表防超时，可用math.sqrt
```
def prime_sieve(n):
    sieve=[True]*(n+1)
    sieve[0]=sieve[1]=False
    for i in range(2,int(n**0.5)+1):
        if sieve[i]:
            sieve[i*i:n+1:i]=[False*len(range(i*i,n+1,i))]
    return [i for i in range(n+1) if sieve[i]]
```
## **递推程序**
**自下至上递堆**
```
n = int(input())
D = []
maxSum = [[-1 for j in range(i+1)] for i in range(n)]
def main():
    for i in range(n):
        lst = list(map(int,input().split()))
        D.append(lst)
    for i in range(n):
        maxSum[n-1][i] = D[n-1][i]
    for i in range(n-2,-1,-1):
        for j in range(0,i+1):
            maxSum[i][j] = max(maxSum[i+1][j],maxSum[i+1][j+1]) + D[i][j]
    print(maxSum[0][0])

main()
```
## **汉诺塔问题**
```
def hanoi(n,a,b,c):
    if n==1:
        print(f'1:{a}->{c}')
    elif n>1:
        hanoi(n-1,a,c,b)
        print(f'{n}:{a}->{c}')
        hanoi(n-1,b,a,c)
n,a,b,c=input().split()
n=int(n)
hanoi(n,a,b,c)
```

## **归并排序**
```
def merge(left,right):
    merged=[]
    inv_count=0
    i=j=0
    while i<len(left) and j<len(right):
        if left[i]<=right[j]:
            merged.append(left[i])
            i+=1
        else:
            merged.append(right[j])
            j+=1
            inv_count+=len(left)-i
    merged+=left[i:]
    merged+=right[j:]
    return merged,inv_count
def merge_sort(lst):
    if len(lst)<=1:
        return lst,0
    middle=len(lst)//2
    left,inv_left=merge_sort(lst[:middle])
    right,inv_right=merge_sort(lst[middle:])
    merged,inv_merged=merge(left,right)
    return merged,inv_left+inv_right+inv_merged
```

# **递归、动规**
## **数字三角形的记忆递归型动规程序**
```
n = int(input())
D = []
maxSum = [[-1 for j in range(i+1)] for i in range(n)]
def MaxSum(i,j):
    if i == n-1:
        return D[i][j]
    if maxSum[i][j] != -1:
        return maxSum[i][j]
    x = MaxSum(i+1,j)
    y = MaxSum(i+1,j+1)
    maxSum[i][j] = max(x,y) + D[i][j]
    return maxSum[i][j]
for i in range(n):
       lst = list(map(int,input().split()))
       D.append(lst)
print(MaxSum(0,0))
```

#二叉树的深度
#扩展二叉树（嵌套树）
#文件结构图
#根据后序表达式建立队列表达式（解析树）
#二叉搜索树的遍历
#树的镜面映射


#有向图判断环
def has_cycle(n,edges):
    graph=[[] for _ in range(n)]
    for u,v in edges:
        graph[u].append(v)
    color=[0]*n
    def dfs(node):
        if color[node]==1:
            return True
        if color[node]==2:
            return False
        color[node]=1
        for neighbor in graph[node]:
            if dfs(neighbor):
                return True
        color[node]=2
        return False
