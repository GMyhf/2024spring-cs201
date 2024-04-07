# 晴问编程题目

Updated 2359 GMT+8 Apr 5, 2024

2024 spring, Complied by Hongfei Yan



晴问《算法笔记》题目练习网址，https://sunnywhy.com/sfbj



# 栈和队列（20题）

## 1 栈的应用 7题

### 1.1 栈的操作序列 简单

现有一个空栈s，按顺序执行n个操作序列，每个操作是下面的两种之一：

1. 往s中压入一个正整数k
2. 弹出的栈顶元素，同时将其输出

**输入**

第一行一个整数 n（$1 \le n \le 100$），表示操作序列的个数；

接下来行，每行一个操作序列，格式为以下两种之一，分别对应入栈和出栈的操作，其中`push k`表示需要将整数k（$1 \le k \le 100$）压入栈，而`pop`表示需要弹出栈顶元素：

1. `push k`
2. `pop`

**输出**

输出多行，每次弹出时输出一行，表示弹出的栈顶元素。如果无法弹出栈顶元素，那么输出`-1`。

样例1

输入

```
8
push 2
pop
pop
push 5
push 3
push 6
pop
pop
```

输出

```
2
-1
6
3
```

解释

初始时栈为空，即`[]`。接下来进行以下`8`个操作：

1. `push 2`：将`2`入栈，此时栈为`[2]`；
2. `pop`：栈顶元素为`2`，因此输出`2`，弹出后栈为`[]`；
3. `pop`：由于栈空，无法弹出元素，因此输出`-1`；
4. `push 5`：将`5`入栈，此时栈为`[5]`；
5. `push 3`：将`3`入栈，此时栈为`[5,3]`；
6. `push 6`：将`6`入栈，此时栈为`[5,3,6]`；
7. `pop`：栈顶元素为`6`，因此输出`6`，弹出后栈为`[5,3]`；
8. `pop`：栈顶元素为`3`，因此输出`3`，弹出后栈为`[5]`。



The user is asking for a Python program that simulates a stack with push and pop operations. The program should read a sequence of operations from the input, perform these operations on the stack, and print the result of each pop operation. If a pop operation is attempted on an empty stack, the program should print `-1`.

Here is a Python program that accomplishes this:

```python
class Stack:
    def __init__(self):
        self.stack = []

    def push(self, x):
        self.stack.append(x)

    def pop(self):
        return self.stack.pop() if self.stack else -1

def main():
    n = int(input().strip())
    stack = Stack()
    for _ in range(n):
        operation = input().strip().split()
        if operation[0] == 'push':
            stack.push(int(operation[1]))
        elif operation[0] == 'pop':
            print(stack.pop())

if __name__ == "__main__":
    main()
```

In this program, we define a `Stack` class with `push` and `pop` methods. The `push` method appends an element to the end of the stack, and the `pop` method removes and returns the last element of the stack, or `-1` if the stack is empty. In the `main` function, we read the number of operations from the input, create a `Stack` object, and perform the operations on the stack. If the operation is a `pop`, we print the result.



### 1.2 合法的出栈序列 简单

现有一个空栈s和一个正整数n，将1,2,3,...,n依次入栈，期间任意时刻出栈。然后给定一个出栈序列，问其是否是一个合法的出栈序列。

**输入**

第一行一个整数n（$1 \le n \le 100$），表示需要入栈的整数个数；

第二行为[1,n]的一个排列，表示一个出栈序列，整数间用空格隔开。

**输出**

如果出栈序列合法，那么输出`Yes`，否则输出`No`。

样例1

输入

```
4
3 2 4 1
```

输出

```
Yes
```

解释

初始时栈为空，即`[]`。接下来按下面的顺序执行即可得到出栈序列`3 2 4 1`：

1. `1`入栈，此时栈为`[1]`；
2. `2`入栈，此时栈为`[1,2]`；
3. `3`入栈，此时栈为`[1,2,3]`；
4. `3`出栈，此时栈为`[1,2]`；
5. `2`出栈，此时栈为`[1]`；
6. `4`入栈，此时栈为`[1,4]`；
7. `4`出栈，此时栈为`[1]`；
8. `1`出栈，此时栈为`[]`。

样例2

输入

```
4
3 4 1 2
```

输出

```
No
```

解释

无法达成出栈序列`3 4 1 2`



The user is asking for a Python program that checks if a given sequence of numbers can be a valid sequence of popped elements from a stack. The stack initially contains a sequence of integers from 1 to n, and the program should simulate pushing these integers onto the stack and popping them off in the order given by the input sequence. If it's possible to pop off the integers in the given order, the program should print `Yes`, otherwise it should print `No`.

Here is a Python program that accomplishes this:

```python
def is_valid_sequence(n, sequence):
    stack = []
    j = 0

    for i in range(1, n + 1):
        stack.append(i)
        while stack and stack[-1] == sequence[j]:
            stack.pop()
            j += 1

    return 'Yes' if not stack else 'No'

def main():
    n = int(input().strip())
    sequence = list(map(int, input().strip().split()))
    print(is_valid_sequence(n, sequence))

if __name__ == "__main__":
    main()
```

In this program, we define a function `is_valid_sequence` that simulates pushing integers onto a stack and popping them off in the order given by the input sequence. If it's possible to pop off the integers in the given order, the function returns `Yes`, otherwise it returns `No`. In the `main` function, we read the number of integers and the sequence from the input, and print the result of `is_valid_sequence`.



### 1.3 可能的出栈序列 中等

现有一个空栈s和一个正整数n，将1,2,3,...,n依次入栈，期间任意时刻出栈。求所有可能的出栈序列。

**输入**

一个整数n（$1 \le n \le 8$），表示需要入栈的整数个数。

**输出**

按字典序从小到大的顺序，每行输出一组可能的出栈序列，整数之间用空格隔开，行末不允许有多余的空格。

样例1

输入

```
3
```

输出

```
1 2 3
1 3 2
2 1 3
2 3 1
3 2 1
```

解释

`3 1 2`是无法达成的，因为第一个出栈的是`3`，但此时栈顶为`2`，不可能把`1`出栈。



The user is asking for a Python program that generates all possible sequences of popped elements from a stack. The stack initially contains a sequence of integers from 1 to n, and the program should simulate pushing these integers onto the stack and popping them off in any order. The program should print all possible sequences of popped elements, sorted in lexicographical order.

Here is a Python program that accomplishes this:

```python
def generate_sequences(stack, in_stack, out_stack, output):
    if len(in_stack) == 0 and len(stack) == 0:
        output.append(out_stack)
    if len(in_stack) > 0:
        generate_sequences(stack + [in_stack[0]], in_stack[1:], out_stack, output)
    if len(stack) > 0:
        generate_sequences(stack[:-1], in_stack, out_stack + [stack[-1]], output)

def main():
    n = int(input().strip())
    in_stack = list(range(1, n + 1))
    output = []
    generate_sequences([], in_stack, [], output)
    output.sort()
    for sequence in output:
        print(' '.join(map(str, sequence)))

if __name__ == "__main__":
    main()
```

In this program, we define a function `generate_sequences` that recursively simulates pushing integers onto a stack and popping them off in any order, and adds the resulting sequences to the `output` list. The function is called with an empty stack, the sequence of integers to be pushed onto the stack, an empty output sequence, and an empty `output` list. In the `main` function, we read the number of integers from the input, generate all possible sequences of popped elements, sort them in lexicographical order, and print them.



### 1.4 后缀表达式-无优先级 简单

给定一个仅由+、-、整数、空格组成的计算式（中缀表达式），将其转换为对应的后缀表达式（即逆波兰表达式）。

**输入**

一个长度不超过`100`的字符串，仅由+、-、整数（不小于`1`且不大于`9`）、空格组成，其中操作符和操作数之间用空格分隔。数据保证表达式一定合法。

**输出**

输出对应的后缀表达式，其中操作数和操作符之间用空格隔开，行末不允许有多余的空格。

样例1

输入

```
3 + 4 - 5
```

输出

```
3 4 + 5 -
```

解释

`3 + 4`转换为后缀表达式为`3 4 +`，记为`P`；

`P - 5`转换为后缀表达式为`P 5 -`，展开后即为`3 4 + 5 -`



The user is asking for a Python program that converts an infix expression to a postfix (Reverse Polish Notation) expression. The program should read an infix expression from the input, convert it to a postfix expression, and print the result.

Here is a Python program that accomplishes this:

```python
def infix_to_postfix(infix):
    precedence = {'+': 1, '-': 1}
    stack = []
    postfix = []

    for token in infix:
        if token.isdigit():
            postfix.append(token)
        else:
            while stack and precedence[stack[-1]] >= precedence[token]:
                postfix.append(stack.pop())
            stack.append(token)

    while stack:
        postfix.append(stack.pop())

    return ' '.join(postfix)

def main():
    infix = input().strip().split()
    print(infix_to_postfix(infix))

if __name__ == "__main__":
    main()
```

In this program, we define a function `infix_to_postfix` that converts an infix expression to a postfix expression. The function uses a stack to hold operators and a list to hold the postfix expression. For each token in the infix expression, if the token is a digit, it is added to the postfix expression. If the token is an operator, operators are popped from the stack and added to the postfix expression until the stack is empty or the operator at the top of the stack has lower precedence than the current operator. Then the current operator is pushed onto the stack. After all tokens have been processed, any remaining operators are popped from the stack and added to the postfix expression. In the `main` function, we read the infix expression from the input, convert it to a postfix expression, and print the result.



### 1.5 后缀表达式-有优先级 中等

给定一个仅由+、-、*、/、整数、空格组成的计算式（中缀表达式），将其转换为对应的后缀表达式（即逆波兰表达式）。

**输入**

一个长度不超过`100`的字符串，仅由+、-、*、/、整数（不小于`1`且不大于`9`）、空格组成，其中操作符和操作数之间用空格分隔。数据保证表达式一定合法。

**输出**

输出对应的后缀表达式，其中操作数和操作符之间用空格隔开，行末不允许有多余的空格。

样例1

输入

```
3 + 4 * 5
```

输出

```
3 4 5 * +
```

解释

由于乘法的优先级比加法高，因此需要先处理乘号：

`4 * 5`转换为后缀表达式为`4 5 *`，记为`P`；

`3 + P`转换为后缀表达式为`3 P +`，展开后即为`3 4 5 * +`



The user is asking for a Python program that converts an infix expression to a postfix (Reverse Polish Notation) expression. The program should read an infix expression from the input, convert it to a postfix expression, and print the result. The infix expression can now include multiplication and division operators.

Here is a Python program that accomplishes this:

```python
def infix_to_postfix(infix):
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2}
    stack = []
    postfix = []

    for token in infix:
        if token.isdigit():
            postfix.append(token)
        else:
            while stack and precedence[stack[-1]] >= precedence[token]:
                postfix.append(stack.pop())
            stack.append(token)

    while stack:
        postfix.append(stack.pop())

    return ' '.join(postfix)

def main():
    infix = input().strip().split()
    print(infix_to_postfix(infix))

if __name__ == "__main__":
    main()
```

In this program, we define a function `infix_to_postfix` that converts an infix expression to a postfix expression. The function uses a stack to hold operators and a list to hold the postfix expression. For each token in the infix expression, if the token is a digit, it is added to the postfix expression. If the token is an operator, operators are popped from the stack and added to the postfix expression until the stack is empty or the operator at the top of the stack has lower precedence than the current operator. Then the current operator is pushed onto the stack. After all tokens have been processed, any remaining operators are popped from the stack and added to the postfix expression. In the `main` function, we read the infix expression from the input, convert it to a postfix expression, and print the result.



### 1.6 后缀表达式-求值 中等

给定一个仅由+、-、*、/、整数组成的计算式（后缀表达式），计算它的值。

**输入**

一个长度不超过`100`的字符串，其中操作符和操作数仅由+、-、*、/、整数（不小于`1`且不大于`9`）构成，且操作符和操作数之间用空格分隔。数据确保表达式一定合法，且计算过程的所有结果不会超过。

**输出**

输出求值结果（保留两位小数精度）。

样例1

输入

```
3 4 5 * +
```

输出

```
23.00
```

解释

先计算`4 5 *`，得到`20`；然后计算`3 20 +`，得到`23`



The user is asking for a Python program that evaluates a postfix (Reverse Polish Notation) expression. The program should read a postfix expression from the input, evaluate it, and print the result. The postfix expression can include addition, subtraction, multiplication, and division operators.

Here is a Python program that accomplishes this:

```python
def evaluate_postfix(postfix):
    stack = []

    for token in postfix:
        if token.isdigit():
            stack.append(int(token))
        else:
            b = stack.pop()
            a = stack.pop()
            if token == '+':
                stack.append(a + b)
            elif token == '-':
                stack.append(a - b)
            elif token == '*':
                stack.append(a * b)
            elif token == '/':
                stack.append(a / b)

    return round(stack[0], 2)

def main():
    postfix = input().strip().split()
    print("{:.2f}".format(evaluate_postfix(postfix)))

if __name__ == "__main__":
    main()
```

In this program, we define a function `evaluate_postfix` that evaluates a postfix expression. The function uses a stack to hold operands. For each token in the postfix expression, if the token is a digit, it is pushed onto the stack. If the token is an operator, two operands are popped from the stack, the operation is performed, and the result is pushed back onto the stack. After all tokens have been processed, the result of the expression is the only element left on the stack. In the `main` function, we read the postfix expression from the input, evaluate it, and print the result.



### 1.7 简单计算器 困难

给定一个仅由+、-、*、/、整数组成的计算式（中缀表达式），计算它的值。

**输入**

一个长度不超过`100`的字符串，其中操作符和操作数仅由+、-、*、/、整数（不小于`1`且不大于`9`）构成，且操作符和操作数之间用空格分隔。数据确保表达式一定合法，且计算过程的所有结果不会超过。

**输出**

输出求值结果（保留两位小数精度）。

样例1

输入

```
3 + 4 * 5
```

输出

```
23.00
```

解释

3 + 4 * 5 = 3 + 20 = 23



The user is asking for a Python program that evaluates an infix expression. The program should read an infix expression from the input, convert it to a postfix expression, evaluate it, and print the result. The infix expression can include addition, subtraction, multiplication, and division operators.

Here is a Python program that accomplishes this:

```python
def infix_to_postfix(infix):
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2}
    stack = []
    postfix = []

    for token in infix:
        if token.isdigit():
            postfix.append(token)
        else:
            while stack and precedence[stack[-1]] >= precedence[token]:
                postfix.append(stack.pop())
            stack.append(token)

    while stack:
        postfix.append(stack.pop())

    return postfix

def evaluate_postfix(postfix):
    stack = []

    for token in postfix:
        if token.isdigit():
            stack.append(int(token))
        else:
            b = stack.pop()
            a = stack.pop()
            if token == '+':
                stack.append(a + b)
            elif token == '-':
                stack.append(a - b)
            elif token == '*':
                stack.append(a * b)
            elif token == '/':
                stack.append(a / b)

    return round(stack[0], 2)

def main():
    infix = input().strip().split()
    postfix = infix_to_postfix(infix)
    print("{:.2f}".format(evaluate_postfix(postfix)))

if __name__ == "__main__":
    main()
```

In this program, we first convert the infix expression to a postfix expression using the `infix_to_postfix` function. Then we evaluate the postfix expression using the `evaluate_postfix` function. The `main` function reads the infix expression from the input, converts it to a postfix expression, evaluates it, and prints the result.



## 2 队列的应用 4题

### 2.1 队列的操作序列 简单

现有一个空队列q，按顺序执行n个操作序列，每个操作是下面的两种之一：

1. 让一个正整数k入队
2. 让队首元素出队，同时将其输出

**输入**

第一行一个整数n（$1 \le n \le 100$），表示操作序列的个数；

接下来n行，每行一个操作序列，格式为以下两种之一，分别对应入队和出队的操作，其中`push k`表示需要将整数k（$1 \le k \le 100$）入队，而`pop`表示需要将队首元素出队：

1. `push k`
2. `pop`

**输出**

输出多行，每次出队时输出一行，表示出队的队首元素。如果无法出队，那么输出`-1`。

样例1

输入

```
8
push 2
pop
pop
push 5
push 3
push 6
pop
pop
```

输出

```
2
-1
5
3
```

解释

初始时队列为空，即`[]`。接下来进行以下`8`个操作：

1. `push 2`：将`2`入栈，此时队列为`[2]`；
2. `pop`：队首元素为`2`，因此输出`2`，出队后队列为`[]`；
3. `pop`：由于队列为空，无法出队，因此输出`-1`；
4. `push 5`：将`5`入队，此时队列为`[5]`；
5. `push 3`：将`3`入队，此时队列为`[5,3]`；
6. `push 6`：将`6`入队，此时队列为`[5,3,6]`；
7. `pop`：队首元素为`5`，因此输出`5`，出队后队列为`[3,6]`；
8. `pop`：队首元素为`3`，因此输出`3`，出队后队列为`[6]`。



The user is asking for a Python program that simulates a queue. The program should read a sequence of operations from the input, perform the operations on the queue, and print the result of each "pop" operation. If a "pop" operation is performed on an empty queue, the program should print "-1".

Here is a Python program that accomplishes this:

```python
from collections import deque

def main():
    n = int(input().strip())
    queue = deque()

    for _ in range(n):
        operation = input().strip().split()
        if operation[0] == 'push':
            queue.append(int(operation[1]))
        elif operation[0] == 'pop':
            if queue:
                print(queue.popleft())
            else:
                print(-1)

if __name__ == "__main__":
    main()
```

In this program, we use a deque from the collections module to simulate a queue. The `main` function reads the number of operations from the input, then reads each operation. If the operation is "push", the program appends the number to the right end of the queue. If the operation is "pop", the program removes and prints the number from the left end of the queue. If the queue is empty when a "pop" operation is performed, the program prints "-1".



### 2.2 求和队列 简单

现有一个空队列q，按顺序将n个正整数a1、a2、...、an入队，接着反复执行操作：将队首的两个元素求和，并将结果入队。最后队列中将只剩下一个整数，将这个整数输出。

**输入**

第一行一个整数n（$1 \le n \le 100$），表示正整数的个数；

第二行为空格隔开的n个正整数a1、a2、...、an（$1 \le a_i \le 100$）。

**输出**

输出一个整数，表示队列中剩下的整数。

样例1

输入

```
5
2 5 4 1 6
```

输出

```
18
```

解释

初始时队列为`[2 5 4 1 6]`。接下来进行以下操作：

1. 计算`2+5=7`，并将`7`入队，当前队列为`4 1 6 7`；
2. 计算`4+1=5`，并将`5`入队，当前队列为`6 7 5`；
3. 计算`6+7=13`，并将`13`入队，当前队列为`5 13`；
4. 计算`5+13=18`，并将`18`入队，当前队列为`18`，因此输出`18`；



The user is asking for a Python program that simulates a queue. The program should read a sequence of integers from the input, perform the operations on the queue (sum the first two elements and enqueue the result), and print the final result when only one element is left in the queue.

Here is a Python program that accomplishes this:

```python
from collections import deque

def main():
    n = int(input().strip())
    numbers = list(map(int, input().strip().split()))
    queue = deque(numbers)

    while len(queue) > 1:
        a = queue.popleft()
        b = queue.popleft()
        queue.append(a + b)

    print(queue[0])

if __name__ == "__main__":
    main()
```

In this program, we use a deque from the collections module to simulate a queue. The `main` function reads the number of integers and the integers themselves from the input, then enqueues each integer. While there is more than one integer in the queue, the program removes and sums the first two integers, then enqueues the result. When only one integer is left in the queue, the program prints it.



### 2.3 约瑟夫环-队列 简单

**约瑟夫环**：假设n个人按编号顺时针从小到大排成一圈（编号为从1到n）。接着约定一个正整数，从编号为1的人开始顺时针报数（编号为1的人报数1，编号为2的人报数2……），报到的人离开圈子，然后他的下一个人继续从1开始报数，以此类推，直到圈子里只剩下一个人。

请用队列模拟约瑟夫环的报数过程，并按先后顺序输出离开圈子的人最开始的编号。为了统一起见，圈子里的最后一个人也需要离开圈子。

**输入**

两个整数n、k（$1 \le n \le 100, 1 \le k \le 100$），含义如题意所示。

**输出**

输出n个整数，按先后顺序表示离开圈子的的人最开始的编号。整数之间用空格隔开，行末不允许有多余的空格。

样例1

输入

```
6 3
```

输出

```
3 6 4 2 5 1
```

解释

一开始有`6`个人：`1 2 3 4 5 6`，从`1`号开始报数，报到`3`的人离开圈子。

第一轮：报号顺序为`1->2->3`，`3`号离开圈子，还剩下`1 2 4 5 6`，从`4`号开始报数；

第二轮：报号顺序为`4->5->6`，`6`号离开圈子，还剩下`1 2 4 5`，从`1`号开始报数；

第三轮：报号顺序为`1->2->4`，`4`号离开圈子，还剩下`1 2 5`，从`5`号开始报数；

第四轮：报号顺序为`5->1->2`，`2`号离开圈子，还剩下`1 5`，从`5`号开始报数；

第五轮：报号顺序为`5->1->5`，`5`号离开圈子，还剩下`1`，从`1`号开始报数；

第六轮：报号顺序为`1->1->1`，`1`号离开圈子，过程结束。



The user is asking for a Python program that simulates the Josephus problem using a queue. The program should read two integers n and k from the input, perform the operations on the queue (count k-1 people and remove the kth person), and print the original positions of the people in the order they are removed.

Here is a Python program that accomplishes this:

```python
from collections import deque

def main():
    n, k = map(int, input().strip().split())
    queue = deque(range(1, n+1))

    while queue:
        queue.rotate(-(k-1))
        print(queue.popleft(), end=' ' if len(queue) != 0 else '\n')

if __name__ == "__main__":
    main()
```

Help on method_descriptor:

rotate(...)
    Rotate the deque n steps to the right (default n=1).  If n is negative, rotates left.

In this program, we use a deque from the collections module to simulate a queue. The `main` function reads the number of people and the count from the input, then enqueues each person's position. While there are still people in the queue, the program rotates the queue k-1 positions to the left, then removes and prints the person at the front of the queue. The rotation operation effectively moves the kth person to the front of the queue.



### 2.4 匹配队列 简单

现有两个队列q1、q2，假设q1、q2初始状态下均有个n元素，每个元素为`1`和`2`之一，且q1中元素`1`的个数与q2中元素`1`的个数相同、q1中元素`2`的个数与q2中元素`2`的个数也相同。

接下来循环执行下面的操作，直到两个队列均为空：

1. 如果q1、q2的队首元素相同，那么将两个队列的队首元素出队；
2. 否则，将q2的队首元素移至队尾。

问需要执行多少轮操作，才能达成停止条件。

**输入**

第一行一个整数n（$1 \le n \le 100$），表示正整数的个数；

第二行为空格隔开的n个正整数`1`或`2`，表示q1中从队首到队尾的所有元素；

第三行为空格隔开的n个正整数`1`或`2`，表示q2中从队首到队尾的所有元素。

**输出**

输出一个整数，表示需要执行的操作轮数。

样例1

输入

```
4
1 2 2 1
2 1 1 2
```

输出

```
6
```

解释

初始状态下：

：`1 2 2 1` ：`2 1 1 2`

第一轮：

q1的队首元素为`1`，q2的队首元素为`2`，不相同，因此把q2的队首元素移至队尾，得到：

q1：`1 2 2 1` q2：`1 1 2 2`

第二轮：

q1的队首元素为`1`，q2的队首元素为`1`，相同，因此把队首元素出队，得到：

q1：`2 2 1` q2：`1 2 2`

第三轮：

q1的队首元素为`2`，q2的队首元素为`1`，不相同，因此把的队首元素移至队尾，得到：

q1：`2 2 1` q2：`2 2 1`

第四轮：

q1的队首元素为`2`，q2的队首元素为`2`，相同，因此把队首元素出队，得到：

q1：`2 1` q2：`2 1`

第五轮：

的队首元素为`2`，的队首元素为`2`，相同，因此把队首元素出队，得到：

：`1` ：`1`

第六轮：

q1的队首元素为`1`，q2的队首元素为`1`，相同，因此把队首元素出队，得到：

q1：空 q2：空

因此共需要`6`轮操作。



The user is asking for a Python program that simulates two queues. The program should read two sequences of integers from the input, perform the operations on the queues (if the first elements of the two queues are the same, dequeue them; otherwise, move the first element of the second queue to its end), and print the number of operations until both queues are empty.

Here is a Python program that accomplishes this:

```python
from collections import deque

def main():
    n = int(input().strip())
    q1 = deque(map(int, input().strip().split()))
    q2 = deque(map(int, input().strip().split()))

    count = 0
    while q1 and q2:
        if q1[0] == q2[0]:
            q1.popleft()
            q2.popleft()
        else:
            q2.append(q2.popleft())
        count += 1

    print(count)

if __name__ == "__main__":
    main()
```

In this program, we use a deque from the collections module to simulate a queue. The `main` function reads the number of integers and the integers themselves from the input, then enqueues each integer into the corresponding queue. While there are still integers in both queues, the program checks the first integer of each queue. If they are the same, the program removes them; otherwise, the program moves the first integer of the second queue to its end. The program counts the number of operations and prints it when both queues are empty.



## 3 链表处理 9题

### 3.1 链表遍历 简单

现有n个结点（编号为从`1`到`n`），以及链表的第一个结点的编号，请依次输出这条链表上每个结点的信息。

**输入**

第一行两个整数n、first（$1 \le first \le 100$），分别表示结点的个数、链表第一个结点的编号；

接下来n行，每行给出一个结点的信息：

```text
id data next
```

其中整数id（1 <= id <= n）表示结点的编号（每个id只出现一次，顺序不定），整数data（1 <= data < 10^3^）表示结点的数据域，整数next（1 <= next <= n或 next = -1）表示当前结点指向的结点编号（其中next = -1表示`NULL`）。

**输出**

按链表结点的顺序，每行输出链表中一个结点的`id`、`data`、`next`，中间用空格隔开。

样例1

输入

```
5 2
1 10 3
2 35 5
3 12 -1
4 23 3
5 58 1
```

输出

```
2 35 5
5 58 1
1 10 3
3 12 -1
```

解释

第一个结点编号为`2`，因此整条链表为：`2(35)->5(58)->1(10)->3(12)->NULL`，结点`4`是多余的。



The user is asking for a Python program that simulates a linked list. The program should read a sequence of nodes from the input, build the linked list, and print the nodes in the order they appear in the linked list.

Here is a Python program that accomplishes this:

```python
class Node:
    def __init__(self, id, data, next):
        self.id = id
        self.data = data
        self.next = next

def main():
    n, first = map(int, input().strip().split())
    nodes = {}

    for _ in range(n):
        id, data, next = map(int, input().strip().split())
        nodes[id] = Node(id, data, next)

    current = nodes[first]
    while current:
        print(f"{current.id} {current.data} {current.next}")
        if current.next in nodes:
            current = nodes[current.next]
        else:
            break

if __name__ == "__main__":
    main()
```

In this program, we define a `Node` class to represent a node in the linked list. The `main` function reads the number of nodes and the id of the first node from the input, then reads each node's id, data, and next id, and stores them in a dictionary. The program then starts from the first node and follows the `next` pointers to print each node in the order they appear in the linked list.



### 3.2 链表的结点个数 简单

现有n个结点（编号为从`1`到`n`），以及链表的第一个结点的编号，求这条链表上的结点个数。

**输入**

第一行两个整数n、first（$1 \le first \le 100$），分别表示结点的个数、链表第一个结点的编号；

接下来n行，每行给出一个结点的信息：

```text
id data next
```

其中整数id（1 <=id <= n）表示结点的编号（每个id只出现一次，顺序不定），整数data（1 <= data <= 10^3^）表示结点的数据域，整数next（1 <= next <= n或 next = -1）表示当前结点指向的结点编号（其中next = -1表示`NULL`）。

**输出**

输出一个整数，表示链表上的结点个数。

样例1

输入

```
5 2
1 10 3
2 35 5
3 12 -1
4 23 3
5 58 1
```

输出

```
4
```

解释

第一个结点编号为`2`，因此整条链表为：`2(35)->5(58)->1(10)->3(12)->NULL`。因此总共有`4`个结点。



The user is asking for a Python program that simulates a linked list. The program should read a sequence of nodes from the input, build the linked list, and print the number of nodes in the linked list.

Here is a Python program that accomplishes this:

```python
class Node:
    def __init__(self, id, data, next):
        self.id = id
        self.data = data
        self.next = next

def main():
    n, first = map(int, input().strip().split())
    nodes = {}

    for _ in range(n):
        id, data, next = map(int, input().strip().split())
        nodes[id] = Node(id, data, next)

    current = nodes[first]
    count = 0
    while current:
        count += 1
        if current.next in nodes:
            current = nodes[current.next]
        else:
            break

    print(count)

if __name__ == "__main__":
    main()
```

In this program, we define a `Node` class to represent a node in the linked list. The `main` function reads the number of nodes and the id of the first node from the input, then reads each node's id, data, and next id, and stores them in a dictionary. The program then starts from the first node and follows the `next` pointers to count each node in the order they appear in the linked list. The program prints the count when it reaches a node with no next node.



### 3.3 链表头插法 简单

现有n个结点（编号为从`1`到`n`），以及链表的第一个结点的编号，使用头插法在链表头部依次插入个结点，然后依次输出新链表上每个结点的信息。

**输入**

第一行两个整数n、first（$1 \le first \le 100$），分别表示结点的个数、链表第一个结点的编号；

接下来n行，每行给出一个结点的信息：

```text
id data next
```

其中整数id（1 <=id <= n）表示结点的编号（每个id只出现一次，顺序不定），整数data（1 <= data <= 1000）表示结点的数据域，整数next（1 <= next <= n或 next = -1）表示当前结点指向的结点编号（其中next = -1表示`NULL`）。

接着在一行中给出一个整数m（1 <= m <=1000），表示待插入的结点个数。

接下来m行，每行给出一个待插入的结点信息：

```text
id data
```

其中整数id（1 <=id <= n）表示结点的编号（每个id只出现一次，顺序不定），整数data（1 <= data <= 10^3^）表示待插入的结点的数据域。

**输出**

按链表结点的顺序，每行输出链表中一个结点的`id`、`data`、`next`，中间用空格隔开。

样例1

输入

```
5 2
1 10 3
2 35 5
3 12 -1
4 23 3
5 58 1
2
6 21
7 18
```

输出

```
7 18 6
6 21 2
2 35 5
5 58 1
1 10 3
3 12 -1
```

解释

第一个结点编号为`2`，因此整条链表为：`2(35)->5(58)->1(10)->3(12)->NULL`，结点`4`是多余的。

接着在头部插入结点`6(21)`，链表变为：`6(21)->2(35)->5(58)->1(10)->3(12)->NULL`；

最后在头部插入结点`7(18)`，链表变为：`7(18)->6(21)->2(35)->5(58)->1(10)->3(12)->NULL`。



The user is asking for a Python program that simulates a linked list. The program should read a sequence of nodes from the input, build the linked list, insert new nodes at the head of the list, and print the nodes in the order they appear in the linked list.

Here is a Python program that accomplishes this:

```python
class Node:
    def __init__(self, id, data, next):
        self.id = id
        self.data = data
        self.next = next

def main():
    n, first = map(int, input().strip().split())
    nodes = {}
    for _ in range(n):
        id, data, next = map(int, input().strip().split())
        nodes[id] = Node(id, data, next)

    head = nodes[first]
    m = int(input().strip())
    for _ in range(m):
        id, data = map(int, input().strip().split())
        new_node = Node(id, data, head.id)
        nodes[id] = new_node
        head = new_node

    current = head
    while current:
        print(f"{current.id} {current.data} {current.next}")
        if current.next in nodes:
            current = nodes[current.next]
        else:
            break

if __name__ == "__main__":
    main()
```

In this program, we define a `Node` class to represent a node in the linked list. The `main` function reads the number of nodes and the id of the first node from the input, then reads each node's id, data, and next id, and stores them in a dictionary. The program then reads the number of new nodes and their id and data, creates a new node with the current head as its next node, and updates the head to the new node. The program then starts from the new head and follows the `next` pointers to print each node in the order they appear in the linked list.



### 3.4 链表删除元素 中等

现有n个结点（编号为从`1`到`n`），以及链表的第一个结点的编号，请删去链表上所有数据域等于整数的结点，然后依次输出新链表上每个结点的信息。

**输入**

第一行三个整数n、first、k（1<=first<=n<=100,1<=k<=1000），分别表示结点的个数、链表第一个结点的编号、待删除结点的数据域；

接下来行，每行给出一个结点的信息：

```text
id data next
```

其中整数id（1 <=id <= n）表示结点的编号（每个id只出现一次，顺序不定），整数data（1 <= data <= 1000）表示结点的数据域，整数next（1 <= next <= n或 next = -1）表示当前结点指向的结点编号（其中next = -1表示`NULL`）。

数据保证至少有一个结点的数据域不为。

**输出**

按链表结点的顺序，每行输出链表中一个结点的`id`、`data`、`next`，中间用空格隔开。

样例1

输入

```
5 2 58
1 10 3
2 35 5
3 58 -1
4 23 3
5 58 1
```

输出

```
2 35 1
1 10 -1
```

解释

第一个结点编号为`2`，因此整条链表为：`2(35)->5(58)->1(10)->3(58)->NULL`，结点`4`是多余的。

由于结点`3`和结点`5`的数据域是`58`，因此删除结点`3`和结点`5`，得到新链表为：`2(35)->1(10)->NULL`



The user is asking for a Python program that simulates a linked list. The program should read a sequence of nodes from the input, build the linked list, delete nodes with a specific data value, and print the nodes in the order they appear in the linked list.

Here is a Python program that accomplishes this:

```python
class Node:
    def __init__(self, id, data, next):
        self.id = id
        self.data = data
        self.next = next

def main():
    n, first, k = map(int, input().strip().split())
    nodes = {}

    for _ in range(n):
        id, data, next = map(int, input().strip().split())
        nodes[id] = Node(id, data, next)

    head = nodes[first]
    prev = None
    current = head
    while current:
        if current.data == k:
            if prev is None:
                head = nodes[current.next] if current.next in nodes else None
            else:
                prev.next = current.next
                if prev.next in nodes:
                    prev.next = nodes[prev.next]
        else:
            prev = current
        current = nodes[current.next] if current.next in nodes else None

    current = head
    while current:
        print(f"{current.id} {current.data} {current.next}")
        if current.next in nodes:
            current = nodes[current.next]
        else:
            break

if __name__ == "__main__":
    main()
```

In this program, we define a `Node` class to represent a node in the linked list. The `main` function reads the number of nodes, the id of the first node, and the data value to delete from the input, then reads each node's id, data, and next id, and stores them in a dictionary. The program then starts from the first node and follows the `next` pointers to delete each node with the specified data value. The program prints the remaining nodes in the order they appear in the linked list.



### 3.5 链表反转 中等

现有个结点（编号为从`1`到`n`），以及链表的第一个结点的编号，请将链表反转（结点顺序颠倒），然后依次输出新链表上每个结点的信息。

**输入**

第一行两个整数n、first（1<=first<=n<=100），分别表示结点的个数、链表第一个结点的编号；

接下来n行，每行给出一个结点的信息：

```text
id data next
```

其中整数id（1 <=id <= n）表示结点的编号（每个id只出现一次，顺序不定），整数data（1 <= data <= 1000）表示结点的数据域，整数next（1 <= next <= n或 next = -1）表示当前结点指向的结点编号（其中next = -1表示`NULL`）。

**输出**

按链表结点的顺序，每行输出链表中一个结点的`id`、`data`、`next`，中间用空格隔开。

样例1

输入

```
5 2
1 10 3
2 35 5
3 12 -1
4 23 3
5 58 1
```

输出

```
3 12 1
1 10 5
5 58 2
2 35 -1
```

解释

第一个结点编号为`2`，因此整条链表为：`2(35)->5(58)->1(10)->3(12)->NULL`，结点`4`是多余的。

将链表反转后可以得到新链表：`3(12)->1(10)->5(58)->2(35)->NULL`



 首先，我们需要创建一个Node类来存储节点的信息，包括id、data和next。然后，使用字典来存储所有的节点。接着，我们需要根据输入的链表第一个节点的编号，找到链表的头节点，然后遍历链表，将链表中的节点按照顺序添加到一个新的列表中。反转新列表。最后，我们需要更新每个节点的next属性，使其指向新列表中的下一个节点，然后按照新的顺序输出每个节点的信息。

以下是实现这个过程的Python代码：

```python
class Node:
    def __init__(self, id, data, next):
        self.id = id
        self.data = data
        self.next = next

n, first = map(int, input().split())
nodes = {}
for _ in range(n):
    id, data, next = map(int, input().split())
    nodes[id] = Node(id, data, next)

# Reverse the linked list
reversed_nodes = []
current = nodes[first]
while current is not None:
    reversed_nodes.append(current)
    if current.next != -1:
        current = nodes[current.next]
    else:
        current = None

reversed_nodes.reverse()

# Update the next attribute of each node
for i in range(len(reversed_nodes) - 1):
    reversed_nodes[i].next = reversed_nodes[i + 1].id
reversed_nodes[-1].next = -1

# Output the reversed linked list
for node in reversed_nodes:
    print(node.id, node.data, node.next)
```

这段代码首先读取输入，然后创建一个Node类的实例来存储每个节点的信息。然后，它遍历链表，将链表中的节点按照顺序添加到一个新的列表中。反转新列表。然后，它更新每个节点的next属性，使其指向新列表中的下一个节点。最后，它按照新的顺序输出每个节点的信息。



### 3.6 链表去除重复元素 中等

现有n个结点（编号为从`1`到`n`），以及链表的第一个结点的编号，请去掉链表上所有数据域重复的结点（只保留第一个），然后依次输出新链表上每个结点的信息。

**输入**

第一行两个整数n、first（1<=first<=n<=100），分别表示结点的个数、链表第一个结点的编号；

接下来行，每行给出一个结点的信息：

```text
id data next
```

其中整数id（1 <=id <= n）表示结点的编号（每个id只出现一次，顺序不定），整数data（1 <= data <= 1000）表示结点的数据域，整数next（1 <= next <= n或 next = -1）表示当前结点指向的结点编号（其中next = -1表示`NULL`）。

**输出**

按链表结点的顺序，每行输出链表中一个结点的`id`、`data`、`next`，中间用空格隔开。

样例1

输入

```
5 2
1 10 3
2 35 5
3 58 -1
4 23 3
5 58 1
```

输出

```
2 35 5
5 58 1
1 10 -1
```

解释

第一个结点编号为`2`，因此整条链表为：`2(35)->5(58)->1(10)->3(58)->NULL`，结点`4`是多余的。

由于结点`3`和结点`5`的数据域是`58`，但结点`5`先出现，因此删除结点`3`，可以得到新链表：`2(35)->5(58)->1(10)->NULL`



首先，我们需要创建一个Node类来存储节点的信息，包括id、data和next。然后，我们需要创建一个字典来存储所有的节点，以便于我们可以通过节点的id快速找到节点。接着，我们需要根据输入的链表第一个节点的编号，找到链表的头节点，然后遍历链表，将链表中的节点按照顺序添加到一个新的列表中。在这个过程中，我们需要使用一个集合来存储已经出现过的数据，如果遇到重复的数据，我们就跳过这个节点。最后，我们需要更新每个节点的next属性，使其指向新列表中的下一个节点，然后按照新的顺序输出每个节点的信息。

以下是实现这个过程的Python代码：

```python
class Node:
    def __init__(self, id, data, next):
        self.id = id
        self.data = data
        self.next = next

n, first = map(int, input().split())
nodes = {}
for _ in range(n):
    id, data, next = map(int, input().split())
    nodes[id] = Node(id, data, next)

# Construct the linked list and remove duplicates
unique_data = set()
filtered_nodes = []
current = nodes[first]
while current is not None:
    if current.data not in unique_data:
        unique_data.add(current.data)
        filtered_nodes.append(current)
    if current.next != -1:
        current = nodes[current.next]
    else:
        current = None

# Update the next attribute of each node
for i in range(len(filtered_nodes) - 1):
    filtered_nodes[i].next = filtered_nodes[i + 1].id
filtered_nodes[-1].next = -1

# Output the filtered linked list
for node in filtered_nodes:
    print(node.id, node.data, node.next)
```

这段代码首先读取输入，然后创建一个Node类的实例来存储每个节点的信息。然后，它遍历链表，将链表中的节点按照顺序添加到一个新的列表中，同时跳过数据域重复的节点。然后，它更新每个节点的next属性，使其指向新列表中的下一个节点。最后，它按照新的顺序输出每个节点的信息。



### 3.7 升序链表中位数 中等

现有n个结点（编号为从`1`到`n`），以及升序链表的第一个结点的编号，求链表中结点数据域的中位数。

注：

1. 升序链表指按结点连接顺序，结点的数据域是递增的链表；
2. 如果结点个数为偶数，那么中位数指最中间两个数的平均值。

**输入**

第一行两个整数n、first（1<=first<=n<=100），分别表示结点的个数、链表第一个结点的编号；

接下来n行，每行给出一个结点的信息：

```text
id data next
```

其中整数id（1 <=id <= n）表示结点的编号（每个id只出现一次，顺序不定），整数data（1 <= data <= 1000）表示结点的数据域，整数next（1 <= next <= n或 next = -1）表示当前结点指向的结点编号（其中next = -1表示`NULL`）。

**输出**

输出中位数，保留一位小数。

样例1

输入

```
5 2
1 35 3
2 10 4
3 58 -1
4 12 5
5 23 1
```

输出

```
23.0
```

解释

第一个结点编号为`2`，因此整条链表为：`2(10)->4(12)->5(23)->1(35)->3(58)->NULL`。

由于有奇数个结点，因此中位数即最中间结点的数据域，即结点`5`的数据域`23`。

样例2

输入

```
5 2
1 23 3
2 10 5
3 58 -1
4 35 3
5 12 1
```

输出

```
17.5
```

解释

第一个结点编号为`2`，因此整条链表为：`2(10)->5(12)->1(23)->3(58)->NULL`，结点`4`是多余的。

由于有偶数个结点，因此中位数中间两个结点数据域的平均值，即结点`5`的数据域`12`和和结点`1`的数据域`23`的平均值`17.5`。



首先，我们需要创建一个Node类来存储节点的信息，包括id、data和next。然后，我们需要创建一个字典来存储所有的节点，以便于我们可以通过节点的id快速找到节点。接着，我们需要根据输入的链表第一个节点的编号，找到链表的头节点，然后遍历链表，将链表中的节点按照顺序添加到一个新的列表中。最后，我们需要找到新列表中的中位数。

以下是实现这个过程的Python代码：

```python
class Node:
    def __init__(self, id, data, next):
        self.id = id
        self.data = data
        self.next = next

n, first = map(int, input().split())
nodes = {}
for _ in range(n):
    id, data, next = map(int, input().split())
    nodes[id] = Node(id, data, next)

# Construct the linked list
linked_list = []
current = nodes[first]
while current is not None:
    linked_list.append(current)
    if current.next != -1:
        current = nodes[current.next]
    else:
        current = None

# Find the median
length = len(linked_list)
if length % 2 == 1:
    median = linked_list[length // 2].data
else:
    median = (linked_list[length // 2 - 1].data + linked_list[length // 2].data) / 2

print(f"{median:.1f}")
```

这段代码首先读取输入，然后创建一个Node类的实例来存储每个节点的信息。然后，它遍历链表，将链表中的节点按照顺序添加到一个新的列表中。然后，它找到新列表中的中位数，并以一位小数的形式输出。



### 3.8 链表倒数第k个结点 中等

现有n个结点（编号为从`1`到`n`），以及链表的第一个结点的编号，求该链表的倒数第k个结点（假设表示最后一个结点）。

**输入**

第一行三个整数n、first、k（1<=first<=n<=100,1<=k<=链表长度），分别表示结点的个数、链表第一个结点的编号、倒数的结点序号；

接下来n行，每行给出一个结点的信息：

```text
id data next
```

其中整数id（1 <=id <= n）表示结点的编号（每个id只出现一次，顺序不定），整数data（1 <= data <= 1000）表示结点的数据域，整数next（1 <= next <= n或 next = -1）表示当前结点指向的结点编号（其中next = -1表示`NULL`）。

**输出**

在一行中输出链表倒数第个结点的的`id`、`data`、`next`，中间用空格隔开。

样例1

输入

```
5 2 2
1 10 3
2 35 5
3 12 -1
4 23 3
5 58 1
```

输出

```
1 10 3
```

解释

第一个结点编号为`2`，因此整条链表为：`2(35)->5(58)->1(10)->3(12)->NULL`，结点`4`是多余的。

因此倒数第`2`个结点是结点`1`。



为了找到链表的倒数第k个节点，我们可以首先遍历整个链表，将所有节点存储在一个列表中。然后，我们可以简单地从列表中获取倒数第k个元素。

以下是实现这个过程的Python代码：

```python
class Node:
    def __init__(self, id, data, next):
        self.id = id
        self.data = data
        self.next = next

n, first, k = map(int, input().split())
nodes = {}
for _ in range(n):
    id, data, next = map(int, input().split())
    nodes[id] = Node(id, data, next)

# Construct the linked list
linked_list = []
current = nodes[first]
while current is not None:
    linked_list.append(current)
    if current.next != -1:
        current = nodes[current.next]
    else:
        current = None

# Find the kth node from the end
kth_node = linked_list[-k]

print(kth_node.id, kth_node.data, kth_node.next)
```

这段代码首先读取输入，然后创建一个Node类的实例来存储每个节点的信息。然后，它遍历链表，将链表中的节点按照顺序添加到一个新的列表中。然后，它找到新列表中的倒数第k个元素，并输出其id、data和next。



### 3.9 回文链表 中等

现有n个结点（编号为从`1`到`n`），以及链表的第一个结点的编号，判断该链表是否是回文的，即链表的结点按数据域正序和逆序是相同的。

**输入**

第一行两个整数n、first（1<=first<=n<=100），分别表示结点的个数、链表第一个结点的编号；

接下来行，每行给出一个结点的信息：

```text
id data next
```

其中整数id（1 <=id <= n）表示结点的编号（每个id只出现一次，顺序不定），整数data（1 <= data <= 1000）表示结点的数据域，整数next（1 <= next <= n或 next = -1）表示当前结点指向的结点编号（其中next = -1表示`NULL`）。

**输出**

如果链表是回文的，那么输出`Yes`，否则输出`No`。

样例1

输入

```
5 2
1 12 3
2 10 4
3 10 -1
4 12 5
5 23 1
```

输出

```
Yes
```

解释

第一个结点编号为`2`，因此整条链表为：`2(10)->4(12)->5(23)->1(12)->3(10)->NULL`。

显然链表是回文的。

样例2

输入

```
2 1
1 23 2
2 10 -1
```

输出

```
No
```

解释

第一个结点编号为`1`，因此整条链表为：`1(23)->2(10)->NULL`。

显然链表不是回文的。



为了判断链表是否是回文的，我们可以首先遍历整个链表，将所有节点的数据域存储在一个列表中。然后，我们可以检查这个列表是否等于它的逆序。

以下是实现这个过程的Python代码：

```python
class Node:
    def __init__(self, id, data, next):
        self.id = id
        self.data = data
        self.next = next

n, first = map(int, input().split())
nodes = {}
for _ in range(n):
    id, data, next = map(int, input().split())
    nodes[id] = Node(id, data, next)

# Construct the linked list
linked_list = []
current = nodes[first]
while current is not None:
    linked_list.append(current.data)
    if current.next != -1:
        current = nodes[current.next]
    else:
        current = None

# Check if the linked list is a palindrome
if linked_list == linked_list[::-1]:
    print("Yes")
else:
    print("No")
```

这段代码首先读取输入，然后创建一个Node类的实例来存储每个节点的信息。然后，它遍历链表，将链表中的节点的数据域按照顺序添加到一个新的列表中。然后，它检查这个列表是否等于它的逆序，如果是，则输出"Yes"，否则输出"No"。









# 搜索专题（15题）

## 1 深度优先搜索（DFS）5题

设想我们现在以第一视角身处一个巨大的迷宫当中，没有上帝视角，没有通信设施，更没有热血动漫里的奇迹，有的只是四周长得一样的墙壁。于是，我们只能自己想办法走出去。如果迷失了内心，随便乱走，那么很可能被四周完全相同的景色绕晕在其中，这时只能放弃所谓的侥幸，而去采取下面这种看上去很盲目但实际上会很有效的方法。

以当前所在位置为起点，沿着一条路向前走，当碰到岔道口时，选择其中一个岔路前进如果选择的这个岔路前方是一条死路，就退回到这个岔道口，选择另一个岔路前进。如果岔路中存在新的岔道口，那么仍然按上面的方法枚举新岔道口的每一条岔路。这样，只要迷宫存在出口，那么这个方法一定能够找到它。可能有读者会问，如果在第一个岔道口处选择了一条没有出路的分支，而这个分支比较深，并且路上多次出现新的岔道口，那么当发现这个分支是个死分支之后，如何退回到最初的这个岔道口?其实方法很简单，只要让右手始终贴着右边的墙壁一路往前走，那么自动会执行上面这个走法，并且最终一定能找到出口。图 8-1 即为使用这个方法走一个简单迷宫的示例。

<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/image-20231126163735204.png" alt="image-20231126163735204" style="zoom:50%;" />



从图 8-1 可知，从起点开始前进，当碰到岔道口时，总是选择其中一条岔路前进(例如图中总是先选择最右手边的岔路)，在岔路上如果又遇到新的岔道口，仍然选择新岔道口的其中一条岔路前进，直到碰到死胡同才回退到最近的岔道口选择另一条岔路。也就是说，当碰到岔道口时，总是以“**深度**”作为前进的关键词，不碰到死胡同就不回头，因此把这种搜索的方式称为**深度优先搜索**(Depth First Search，**DFS**)。
从迷宫的例子还应该注意到，深度优先搜索会走遍所有路径，并且每次走到死胡同就代表一条完整路径的形成。这就是说，**深度优先搜索是一种枚举所有完整路径以遍历所有情况的搜索方法**。



深度优先搜索 (DFS)可以使用栈来实现。但是实现起来却并不轻松，有没有既容易理解又容易实现的方法呢?有的——递归。现在从 DFS 的角度来看当初求解 Fibonacci 数列的过程。
回顾一下 Fibonacci数列的定义: $F(0)=1,F(1)=1,F(n)=F(n-1)+F(n-2)(n≥2)$。可以从这个定义中挖掘到，每当将 F(n)分为两部分 F(n-1)与 F(n-2)时，就可以把 F(n)看作迷宫的岔道口，由它可以到达两个新的关键结点 F(n-1)与 F(n-2)。而之后计算 F(n-1)时，又可以把 F(n-1)当作在岔道口 F(n)之下的岔道口。
既然有岔道口，那么一定有死胡同。很容易想象，当访问到 F(0)和 F(1)时，就无法再向下递归下去，因此 F(0)和 F(1)就是死胡同。这样说来，==递归中的递归式就是岔道口，而递归边界就是死胡同==，这样就可以把如何用递归实现深度优先搜索的过程理解得很清楚。为了使上面的过程更清晰，可以直接来分析递归图 (见图 4-3)：可以在递归图中看到，只要n > 1，F(n)就有两个分支，即把 F(n)当作岔道口；而当n为1或0时，F(1)与F(0)就是迷宫的死胡同，在此处程序就需要返回结果。这样当遍历完所有路径（从顶端的 F(4)到底层的所有 F(1)与 F(0)）后，就可以得到 F(4)的值。

<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/image-20231126164549437.png" alt="image-20231126164549437" style="zoom: 50%;" />

因此，使用递归可以很好地实现深度优先搜索。这个说法并不是说深度优先搜索就是递归，只能说递归是深度优先搜索的一种实现方式，因为使用非递归也是可以实现 DFS 的思想的，但是一般情况下会比递归麻烦。不过，使用递归时，系统会调用一个叫系统栈的东西来存放递归中每一层的状态，因此使用递归来实现 DFS 的本质其实还是栈。



### sy313: 迷宫可行路径数 简单

https://sunnywhy.com/sfbj/8/1/313

现有一个 n*m 大小的迷宫，其中`1`表示不可通过的墙壁，`0`表示平地。每次移动只能向上下左右移动一格（不允许移动到曾经经过的位置），且只能移动到平地上。求从迷宫左上角到右下角的所有可行路径的条数。

**输入**

第一行两个整数 $n、m \hspace{1em} (2 \le n \le5, 2 \le m \le 5)$，分别表示迷宫的行数和列数；

接下来 n 行，每行 m 个整数（值为`0`或`1`），表示迷宫。

**输出**

一个整数，表示可行路径的条数。

样例1

输入

```
3 3
0 0 0
0 1 0
0 0 0
```

输出

```
2
```

解释

假设左上角坐标是(1,1)，行数增加的方向为增长的方向，列数增加的方向为增长的方向。

可以得到从左上角到右下角有两条路径：

1. (1,1)=>(1,2)=>(1,3)=>(2,3)=>(3,3)
2. (1,1)=>(2,1)=>(3,1)=>(3,2)=>(3,3)



**加保护圈，原地修改**

```python
dx = [-1, 0, 1, 0]
dy = [ 0, 1, 0, -1]

def dfs(maze, x, y):
    global cnt
    
    for i in range(4):
        nx = x + dx[i]
        ny = y + dy[i]
            
        if maze[nx][ny] == 'e':
            cnt += 1
            continue
            
        if maze[nx][ny] == 0:
            maze[x][y] = 1
            dfs(maze, nx, ny)
            maze[x][y] = 0
    
    return
            
n, m = map(int, input().split())
maze = []
maze.append( [-1 for x in range(m+2)] )
for _ in range(n):
    maze.append([-1] + [int(_) for _ in input().split()] + [-1])
maze.append( [-1 for x in range(m+2)] )

maze[1][1] = 's'
maze[n][m] = 'e'

cnt = 0
dfs(maze, 1, 1)
print(cnt)
```



**辅助visited空间**

```python
# gpt translated version of the C++ code
MAXN = 5
n, m = map(int, input().split())
maze = []
for _ in range(n):
    row = list(map(int, input().split()))
    maze.append(row)

visited = [[False for _ in range(m)] for _ in range(n)]
counter = 0

MAXD = 4
dx = [0, 0, 1, -1]
dy = [1, -1, 0, 0]

def is_valid(x, y):
    return 0 <= x < n and 0 <= y < m and maze[x][y] == 0 and not visited[x][y]

def DFS(x, y):
    global counter
    if x == n - 1 and y == m - 1:
        counter += 1
        return
    visited[x][y] = True
    for i in range(MAXD):
        nextX = x + dx[i]
        nextY = y + dy[i]
        if is_valid(nextX, nextY):
            DFS(nextX, nextY)
    visited[x][y] = False

DFS(0, 0)
print(counter)

```





### sy314: 指定步数的迷宫问题 中等

https://sunnywhy.com/sfbj/8/1/314

现有一个 n*m 大小的迷宫，其中`1`表示不可通过的墙壁，`0`表示平地。每次移动只能向上下左右移动一格（不允许移动到曾经经过的位置），且只能移动到平地上。现从迷宫左上角出发，问能否在恰好第步时到达右下角。

**输入**

第一行三个整数$n、m、k \hspace{1em} (2 \le n \le5, 2 \le m \le 5, 2 \le k \le n*m)$，分别表示迷宫的行数、列数、移动的步数；

接下来行，每行个整数（值为`0`或`1`），表示迷宫。

**输出**

如果可行，那么输出`Yes`，否则输出`No`。

样例1

输入

```
3 3 4
0 1 0
0 0 0
0 1 0
```

输出

```
Yes
```

解释

假设左上角坐标是(1,1)，行数增加的方向为增长的方向，列数增加的方向为增长的方向。

可以得到从左上角到右下角的步数为`4`的路径为：(1,1)=>(2,1)=>(2,2)=>(2,3)=>(3,3)。

样例2

输入

```
3 3 6
0 1 0
0 0 0
0 1 0
```

输出

```
No
```

解释

由于不能移动到曾经经过的位置，因此无法在恰好第`6`步时到达右下角。



**加保护圈，原地修改**

```python
dx = [-1, 0, 1, 0]
dy = [ 0, 1, 0, -1]

canReach = False
def dfs(maze, x, y, step):
    global canReach
    if canReach:
        return
    
    for i in range(4):
        nx = x + dx[i]
        ny = y + dy[i]
        if maze[nx][ny] == 'e':
            if step==k-1:
                canReach = True
                return
            
            continue
            
        if maze[nx][ny] == 0:
            if step < k:
                maze[x][y] = -1
                dfs(maze, nx, ny, step+1)
                maze[x][y] = 0
    

n, m, k = map(int, input().split())
maze = []
maze.append( [-1 for x in range(m+2)] )
for _ in range(n):
    maze.append([-1] + [int(_) for _ in input().split()] + [-1])
maze.append( [-1 for x in range(m+2)] )

maze[1][1] = 's'
maze[n][m] = 'e'

dfs(maze, 1, 1, 0)
print("Yes" if canReach else "No")
```



**辅助visited空间**

```python
# gpt translated version of the C++ code
MAXN = 5
n, m, k = map(int, input().split())
maze = []
for _ in range(n):
    row = list(map(int, input().split()))
    maze.append(row)

visited = [[False for _ in range(m)] for _ in range(n)]
canReach = False

MAXD = 4
dx = [0, 0, 1, -1]
dy = [1, -1, 0, 0]

def is_valid(x, y):
    return 0 <= x < n and 0 <= y < m and maze[x][y] == 0 and not visited[x][y]

def DFS(x, y, step):
    global canReach
    if canReach:
        return
    if x == n - 1 and y == m - 1:
        if step == k:
            canReach = True
        return
    visited[x][y] = True
    for i in range(MAXD):
        nextX = x + dx[i]
        nextY = y + dy[i]
        if step < k and is_valid(nextX, nextY):
            DFS(nextX, nextY, step + 1)
    visited[x][y] = False

DFS(0, 0, 0)
print("Yes" if canReach else "No")

```





### sy315: 矩阵最大权值 中等

https://sunnywhy.com/sfbj/8/1/315

现有一个 n*m 大小的矩阵，矩阵中的每个元素表示该位置的权值。现需要从矩阵左上角出发到达右下角，每次移动只能向上下左右移动一格（不允许移动到曾经经过的位置）。求最后到达右下角时路径上所有位置的权值之和的最大值。

**输入**

第一行两个整数 $n、m \hspace{1em} (2 \le n \le5, 2 \le m \le 5)$，分别表示矩阵的行数和列数；

接下来 n 行，每行 m 个整数（$-100 \le 整数 \le 100$），表示矩阵每个位置的权值。

**输出**

一个整数，表示权值之和的最大值。

样例1

输入

```
2 2
1 2
3 4
```

输出

```
8
```

解释

从左上角到右下角的最大权值之和为。



**加保护圈，原地修改**

```python
dx = [-1, 0, 1, 0]
dy = [ 0, 1, 0, -1]

maxValue = float("-inf")
def dfs(maze, x, y, nowValue):
    global maxValue
    if x==n and y==m:
        if nowValue > maxValue:
            maxValue = nowValue
        
        return
  
    for i in range(4):
        nx = x + dx[i]
        ny = y + dy[i]
  
        if maze[nx][ny] != -9999:
            tmp = maze[x][y]
            maze[x][y] = -9999
            nextValue = nowValue + maze[nx][ny]
            dfs(maze, nx, ny, nextValue)
            maze[x][y] = tmp
    

n, m = map(int, input().split())
maze = []
maze.append( [-9999 for x in range(m+2)] )
for _ in range(n):
    maze.append([-9999] + [int(_) for _ in input().split()] + [-9999])
maze.append( [-9999 for x in range(m+2)] )


dfs(maze, 1, 1, maze[1][1])
print(maxValue)
```



**辅助visited空间**

```python
# gpt translated version of the C++ code
MAXN = 5
INF = float('inf')
n, m = map(int, input().split())
maze = []
for _ in range(n):
    row = list(map(int, input().split()))
    maze.append(row)

visited = [[False for _ in range(m)] for _ in range(n)]
maxValue = -INF

MAXD = 4
dx = [0, 0, 1, -1]
dy = [1, -1, 0, 0]

def is_valid(x, y):
    return 0 <= x < n and 0 <= y < m and not visited[x][y]

def DFS(x, y, nowValue):
    global maxValue
    if x == n - 1 and y == m - 1:
        if nowValue > maxValue:
            maxValue = nowValue
        return
    visited[x][y] = True
    for i in range(MAXD):
        nextX = x + dx[i]
        nextY = y + dy[i]
        if is_valid(nextX, nextY):
            nextValue = nowValue + maze[nextX][nextY]
            DFS(nextX, nextY, nextValue)
    visited[x][y] = False

DFS(0, 0, maze[0][0])
print(maxValue)

```





### sy316: 矩阵最大权值路径 中等

https://sunnywhy.com/sfbj/8/1/316

现有一个 n*m 大小的矩阵，矩阵中的每个元素表示该位置的权值。现需要从矩阵左上角出发到达右下角，每次移动只能向上下左右移动一格（不允许移动到曾经经过的位置）。假设左上角坐标是(1,1)，行数增加的方向为增长的方向，列数增加的方向为增长的方向。求最后到达右下角时路径上所有位置的权值之和最大的路径。

**输入**

第一行两个整数 $n、m \hspace{1em} (2 \le n \le5, 2 \le m \le 5)$，分别表示矩阵的行数和列数；

接下来 n 行，每行 m 个整数（$-100 \le 整数 \le 100$），表示矩阵每个位置的权值。

**输出**

从左上角的坐标开始，输出若干行（每行两个整数，表示一个坐标），直到右下角的坐标。

数据保证权值之和最大的路径存在且唯一。

样例1

输入

```
2 2
1 2
3 4
```

输出

```
1 1
2 1
2 2
```

解释

显然当路径是(1,1)=>(2,1)=>(2,2)时，权值之和最大，即 1+3+4 = 8。



**辅助visited空间**

```python
# gpt translated version of the C++ code
MAXN = 5
INF = float('inf')
n, m = map(int, input().split())
maze = []
for _ in range(n):
    row = list(map(int, input().split()))
    maze.append(row)

visited = [[False for _ in range(m)] for _ in range(n)]
maxValue = -INF
tempPath, optPath = [], []

MAXD = 4
dx = [0, 0, 1, -1]
dy = [1, -1, 0, 0]

def is_valid(x, y):
    return 0 <= x < n and 0 <= y < m and not visited[x][y]

def DFS(x, y, nowValue):
    global maxValue, tempPath, optPath
    if x == n - 1 and y == m - 1:
        if nowValue > maxValue:
            maxValue = nowValue
            optPath = list(tempPath)
        return
    visited[x][y] = True
    for i in range(MAXD):
        nextX = x + dx[i]
        nextY = y + dy[i]
        if is_valid(nextX, nextY):
            nextValue = nowValue + maze[nextX][nextY]
            tempPath.append((nextX, nextY))
            DFS(nextX, nextY, nextValue)
            tempPath.pop()
    visited[x][y] = False

tempPath.append((0, 0))
DFS(0, 0, maze[0][0])
for pos in optPath:
    print(pos[0] + 1, pos[1] + 1)
```





### sy317: 迷宫最大权值 中等

https://sunnywhy.com/sfbj/8/1/317

现有一个大小的迷宫，其中`1`表示不可通过的墙壁，`0`表示平地。现需要从迷宫左上角出发到达右下角，每次移动只能向上下左右移动一格（不允许移动到曾经经过的位置），且只能移动到平地上。假设迷宫中每个位置都有权值，求最后到达右下角时路径上所有位置的权值之和的最大值。

**输入**

第一行两个整数$n、m \hspace{1em} (2 \le n \le5, 2 \le m \le 5)$，分别表示矩阵的行数和列数；

接下来 n 行，每行个 m 整数（值为`0`或`1`），表示迷宫。

再接下来行，每行个整数（$-100 \le 整数 \le 100$），表示迷宫每个位置的权值。

**输出**

一个整数，表示权值之和的最大值。

样例1

输入

```
3 3
0 0 0
0 1 0
0 0 0
1 2 3
4 5 6
7 8 9
```

输出

```
29
```

解释：从左上角到右下角的最大权值之和为 1+4+7+8+9 = 29。



**加保护圈，原地修改**

```python
dx = [-1, 0, 1, 0]
dy = [ 0, 1, 0, -1]

maxValue = float("-inf")
def dfs(maze, x, y, nowValue):
    global maxValue
    if x==n and y==m:
        if nowValue > maxValue:
            maxValue = nowValue
        
        return
  
    for i in range(4):
        nx = x + dx[i]
        ny = y + dy[i]
  
        if maze[nx][ny] == 0:
            maze[nx][ny] = -1
            tmp = w[x][y]
            w[x][y] = -9999
            nextValue = nowValue + w[nx][ny]
            dfs(maze, nx, ny, nextValue)
            maze[nx][ny] = 0
            w[x][y] = tmp
    

n, m = map(int, input().split())
maze = []
maze.append( [-1 for x in range(m+2)] )
for _ in range(n):
    maze.append([-1] + [int(_) for _ in input().split()] + [-1])
maze.append( [-1 for x in range(m+2)] )

w = []
w.append( [-9999 for x in range(m+2)] )
for _ in range(n):
    w.append([-9999] + [int(_) for _ in input().split()] + [-9999])
w.append( [-9999 for x in range(m+2)] )


dfs(maze, 1, 1, w[1][1])
print(maxValue)
```



**辅助visited空间**

```python
# gpt translated version of the C++ code
MAXN = 5
INF = float('inf')
n, m = map(int, input().split())
maze = [list(map(int, input().split())) for _ in range(n)]
w = [list(map(int, input().split())) for _ in range(n)]
visited = [[False] * m for _ in range(n)]
maxValue = -INF

MAXD = 4
dx = [0, 0, 1, -1]
dy = [1, -1, 0, 0]

def is_valid(x, y):
    return 0 <= x < n and 0 <= y < m and not maze[x][y] and not visited[x][y]

def dfs(x, y, nowValue):
    global maxValue
    if x == n - 1 and y == m - 1:
        if nowValue > maxValue:
            maxValue = nowValue
        return
    visited[x][y] = True
    for i in range(MAXD):
        nextX = x + dx[i]
        nextY = y + dy[i]
        if is_valid(nextX, nextY):
            nextValue = nowValue + w[nextX][nextY]
            dfs(nextX, nextY, nextValue)
    visited[x][y] = False

dfs(0, 0, w[0][0])
print(maxValue)

```







## 2 广度优先搜索（BFS）10题

前面介绍了深度优先搜索，可知 DFS 是以深度作为第一关键词的，即当碰到岔道口时总是先选择其中的一条岔路前进,而不管其他岔路,直到碰到死胡同时才返回岔道口并选择其他岔路。接下来将介绍的**广度优先搜索** (Breadth FirstSearch,**BFS**)则是以广度为第一关键词，当碰到岔道口时,总是先依次访问从该岔道口能直接到达的所有结点,然后再按这些结点被访问的顺序去依次访问它们能直接到达的所有结点，以此类推,直到所有结点都被访问为止。这就跟平静的水面中投入一颗小石子一样,水花总是以石子落水处为中心,并以同心圆的方式向外扩散至整个水面(见图 8-2),从这点来看和 DFS 那种沿着一条线前进的思路是完全不同的。

<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/202311262216546.png" alt="image-20231126221551540" style="zoom:50%;" />

广度优先搜索 (BFS)一般由队列实现,且总是按层次的顺序进行遍历，其基本写法如下(可作模板用):

```python
from collections import deque
  
def bfs(s, e):
    vis = set()
    vis.add(s)
      
    q = deque()
    q.append((0, s))

    while q:
        now, top = q.popleft() # 取出队首元素
        if top == e:
            return now # 返回需要的结果，如：步长、路径等信息

        # 将 top 的下一层结点中未曾入队的结点全部入队q，并加入集合vis设置为已入队
  
```



下面是对该模板中每一个步骤的说明,请结合代码一起看: 

① 定义队列 q，并将起点(0, s)入队，0表示步长目前是0。
② 写一个 while 循环，循环条件是队列q非空。
③ 在 while 循环中，先取出队首元素 top。
④ 将top 的下一层结点中所有**未曾入队**的结点入队，并标记它们的层号为 now 的层号加1，并加入集合vis设置为已入队。
⑤ 返回 ② 继续循环。



再强调一点,在BFS 中设置的 inq 数组的含义是判断结点是否已入过队，而不是**结点是否已被访问**。区别在于:如果设置成是否已被访问，有可能在某个结点正在队列中(但还未访问)时由于其他结点可以到达它而将这个结点再次入队，导致很多结点反复入队，计算量大大增加。因此BFS 中让每个结点只入队一次，故需要设置 inq 数组的含义为**结点是否已入过队**而非结点是否已被访问。



### sy318: 数字操作（一维BFS）

https://sunnywhy.com/sfbj/8/2/318

从整数`1`开始，每轮操作可以选择将上轮结果加`1`或乘`2`。问至少需要多少轮操作才能达到指定整数。

**输入**

一个整数 $n \hspace{1em} (2 \le n \le 10^5)$，表示需要达到的整数。

**输出**

输出一个整数，表示至少需要的操作轮数。

样例1

输入

```
7
```

输出

```
4
```

解释

第`1`轮：1 + 1 = 2

第`2`轮：2 + 1 =3

第`3`轮：3 * 2 = 6

第`4`轮：6 + 1 = 7

因此至少需要操作`4`轮。



#### 数学思维

```python
'''
2023TA-陈威宇，思路：是n的二进制表示 里面 1的个数+1的个数+0的个数-2。
如果我们将 n 的二进制表示的每一位数从左到右依次编号为 0、1、2、...，那么：

1 的个数表示需要进行加 1 的操作次数；
0 的个数表示需要进行乘 2 的操作次数；
len(l) - 2 表示操作的总次数减去初始状态的操作次数 1，即剩余的操作次数；
sum(l) + len(l) - 2 表示所有操作次数之和。
'''
n = int(input())
s = bin(n)
l = [int(i) for i in s[2:]]
print(sum(l) + len(l) - 2)
```



#### 计算机思维

##### Python

```python
from collections import deque

def bfs(n):

    vis = set()
    vis.add(1)
    q = deque()
    q.append((1, 0))
    while q:
        front, step = q.popleft()
        if front == n:
            return step

        if front * 2 <= n and front * 2 not in vis:
            vis.add(front *2)
            q.append((front * 2, step+1))
        if front + 1 <= n and front + 1 not in vis:
            vis.add(front + 1)
            q.append((front + 1, step+1))


n = int(input())
print(bfs(n))

```



```python
# gpt translated version of the C++ code
from collections import deque

MAXN = 100000
in_queue = [False] * (MAXN + 1)

def get_step(n):
    step = 0
    q = deque()
    q.append(1)
    while True:
        cnt = len(q)
        for _ in range(cnt):
            front = q.popleft()
            if front == n:
                return step
            in_queue[front] = True
            if front * 2 <= n and not in_queue[front * 2]:
                q.append(front * 2)
            if front + 1 <= n and not in_queue[front + 1]:
                q.append(front + 1)
        step += 1

if __name__ == "__main__":
    n = int(input())
    print(get_step(n))
```





### sy319: 矩阵中的块

https://sunnywhy.com/sfbj/8/2/319

现有一个 n*m 的矩阵，矩阵中的元素为`0`或`1`。然后进行如下定义：

1. 位置(x,y)与其上下左右四个位置 $(x,y + 1)、(x,y - 1)、(x + 1,y)、(x-1,y)$ 是相邻的；
2. 如果位置 (x1,y1) 与位置 (x2,y2) 相邻，且位置 (x2,y2) 与位置 (x3,y3) 相邻，那么称位置(x1,y1)与位置(x3,y3)也相邻；
3. 称个数尽可能多的相邻的`1`构成一个“块”。

求给定的矩阵中“块”的个数。

**输入**

第一行两个整数 n、m（$2 \le n \le 100, 2 \le m \le 100$），分别表示矩阵的行数和列数；

接下来 n 行，每行 m 个`0`或`1`（用空格隔开），表示矩阵中的所有元素。

**输出**

输出一个整数，表示矩阵中“块”的个数。

样例1

输入

```
6 7
0 1 1 1 0 0 1
0 0 1 0 0 0 0
0 0 0 0 1 0 0
0 0 0 1 1 1 0
1 1 1 0 1 0 0
1 1 1 1 0 0 0
```

输出

```
4
```

解释

矩阵中的`1`共有`4`块，如下图所示。

![矩阵中的块_样例.png](https://raw.githubusercontent.com/GMyhf/img/main/img/202311262246785.png)



#### 加保护圈，inq_set集合判断是否入过队

```python
from collections import deque

# Constants
MAXD = 4
dx = [0, 0, 1, -1]
dy = [1, -1, 0, 0]

def bfs(x, y):
    q = deque([(x, y)])
    inq_set.add((x,y))
    while q:
        front = q.popleft()
        for i in range(MAXD):
            next_x = front[0] + dx[i]
            next_y = front[1] + dy[i]
            if matrix[next_x][next_y] == 1 and (next_x,next_y) not in inq_set:
                inq_set.add((next_x, next_y))
                q.append((next_x, next_y))

# Input
n, m = map(int, input().split())
matrix=[[-1]*(m+2)]+[[-1]+list(map(int,input().split()))+[-1] for i in range(n)]+[[-1]*(m+2)]
inq_set = set()

# Main process
counter = 0
for i in range(1,n+1):
    for j in range(1,m+1):
        if matrix[i][j] == 1 and (i,j) not in inq_set:
            bfs(i, j)
            counter += 1

# Output
print(counter)
```



#### inq 数组，结点是否已入过队

```python
# gpt translated version of the C++ code
from collections import deque

# Constants
MAXN = 100
MAXD = 4
dx = [0, 0, 1, -1]
dy = [1, -1, 0, 0]

# Functions
def can_visit(x, y):
    return 0 <= x < n and 0 <= y < m and matrix[x][y] == 1 and not in_queue[x][y]

def bfs(x, y):
    q = deque([(x, y)])
    in_queue[x][y] = True
    while q:
        front = q.popleft()
        for i in range(MAXD):
            next_x = front[0] + dx[i]
            next_y = front[1] + dy[i]
            if can_visit(next_x, next_y):
                in_queue[next_x][next_y] = True
                q.append((next_x, next_y))

# Input
n, m = map(int, input().split())
matrix = [list(map(int, input().split())) for _ in range(n)]
in_queue = [[False] * MAXN for _ in range(MAXN)]

# Main process
counter = 0
for i in range(n):
    for j in range(m):
        if matrix[i][j] == 1 and not in_queue[i][j]:
            bfs(i, j)
            counter += 1

# Output
print(counter)

```





### sy320: 迷宫问题

https://sunnywhy.com/sfbj/8/2/320

现有一个 n*m 大小的迷宫，其中`1`表示不可通过的墙壁，`0`表示平地。每次移动只能向上下左右移动一格，且只能移动到平地上。求从迷宫左上角到右下角的最小步数。

**输入**

第一行两个整数 $n、m \hspace{1em} (2 \le n \le 100, 2 \le m \le 100)$，分别表示迷宫的行数和列数；

接下来 n 行，每行 m 个整数（值为`0`或`1`），表示迷宫。

**输出**

输出一个整数，表示最小步数。如果无法到达，那么输出`-1`。

样例1

输入

```
3 3
0 1 0
0 0 0
0 1 0
```

输出

```
4
```

解释: 假设左上角坐标是(1,1)，行数增加的方向为增长的方向，列数增加的方向为增长的方向。

可以得到从左上角到右下角的前进路线：(1,1)=>(2,1)=>(2,2)=>(2,3)=>(3,3)。

因此最少需要`4`步。

样例2

输入

```
3 3
0 1 0
0 1 0
0 1 0
```

输出

```
-1
```

解释: 显然从左上角无法到达右下角。



#### 加保护圈，inq_set集合判断是否入过队

```python
from collections import deque

# 声明方向变化的数组，代表上下左右移动
dx = [0, 0, 1, -1]
dy = [1, -1, 0, 0]

def bfs(x, y):
    q = deque()
    q.append((x, y))
    inq_set.add((x, y))
    step = 0
    while q:
        for _ in range(len(q)):
            cur_x, cur_y = q.popleft()
            if cur_x == n and cur_y == m:
                return step
            for direction in range(4):
                next_x = cur_x + dx[direction]
                next_y = cur_y + dy[direction]
                if maze[next_x][next_y] == 0 and (next_x,next_y) not in inq_set:
                    inq_set.add((next_x, next_y))
                    q.append((next_x, next_y))
        step += 1
    return -1

if __name__ == '__main__':

    n, m = map(int, input().split())
    maze = [[-1] * (m + 2)] + [[-1] + list(map(int, input().split())) + [-1] for i in range(n)] + [[-1] * (m + 2)]
    inq_set = set()

    step = bfs(1, 1)
    print(step)

```



#### inq 数组，结点是否已入过队

```python
# gpt translated version of the C++ code
from collections import deque

# 声明方向变化的数组，代表上下左右移动
dx = [0, 0, 1, -1]
dy = [1, -1, 0, 0]

# 检查是否可以访问位置 (x, y)
def can_visit(x, y):
    return 0 <= x < n and 0 <= y < m and maze[x][y] == 0 and not in_queue[x][y]

# BFS函数 实现广度优先搜索
def bfs(x, y):
    q = deque()
    q.append((x, y))
    in_queue[x][y] = True
    step = 0
    while q:
        for _ in range(len(q)):
            cur_x, cur_y = q.popleft()
            if cur_x == n - 1 and cur_y == m - 1:
                return step
            for direction in range(4):
                next_x = cur_x + dx[direction]
                next_y = cur_y + dy[direction]
                if can_visit(next_x, next_y):
                    in_queue[next_x][next_y] = True
                    q.append((next_x, next_y))
        step += 1
    return -1

# 主函数
if __name__ == '__main__':
    # 读取 n 和 m
    n, m = map(int, input().split())
    maze = []
    in_queue = [[False] * m for _ in range(n)]

    # 填充迷宫和访问状态数组
    for i in range(n):
        maze.append(list(map(int, input().split())))

    # 执行BFS并输出步数
    step = bfs(0, 0)
    print(step)

```





### sy321: 迷宫最短路径

https://sunnywhy.com/sfbj/8/2/321

现有一个 n*m 大小的迷宫，其中`1`表示不可通过的墙壁，`0`表示平地。每次移动只能向上下左右移动一格，且只能移动到平地上。假设左上角坐标是(1,1)，行数增加的方向为增长的方向，列数增加的方向为增长的方向，求从迷宫左上角到右下角的最少步数的路径。

**输入**

第一行两个整数$n、m \hspace{1em} (2 \le n \le 100, 2 \le m \le 100)$，分别表示迷宫的行数和列数；

接下来 n 行，每行 m 个整数（值为`0`或`1`），表示迷宫。

**输出**

从左上角的坐标开始，输出若干行（每行两个整数，表示一个坐标），直到右下角的坐标。

数据保证最少步数的路径存在且唯一。

样例1

输入

```
3 3
0 1 0
0 0 0
0 1 0
```

输出

```
1 1
2 1
2 2
2 3
3 3
```

解释

假设左上角坐标是(1,)，行数增加的方向为增长的方向，列数增加的方向为增长的方向。

可以得到从左上角到右下角的最少步数的路径为：(1,1)=>(2,1)=>(2,2)=>(2,3)=>(3,3)。



#### inq 数组，结点是否已入过队

```python
# gpt translated version of the C++ code
from queue import Queue

MAXN = 100
MAXD = 4
dx = [0, 0, 1, -1]
dy = [1, -1, 0, 0]

def canVisit(x, y):
    return x >= 0 and x < n and y >= 0 and y < m and maze[x][y] == 0 and not inQueue[x][y]

def BFS(x, y):
    q = Queue()
    q.put((x, y))
    inQueue[x][y] = True
    while not q.empty():
        front = q.get()
        if front[0] == n - 1 and front[1] == m - 1:
            return
        for i in range(MAXD):
            nextX = front[0] + dx[i]
            nextY = front[1] + dy[i]
            if canVisit(nextX, nextY):
                pre[nextX][nextY] = (front[0], front[1])
                inQueue[nextX][nextY] = True
                q.put((nextX, nextY))

def printPath(p):
    prePosition = pre[p[0]][p[1]]
    if prePosition == (-1, -1):
        print(p[0] + 1, p[1] + 1)
        return
    printPath(prePosition)
    print(p[0] + 1, p[1] + 1)

n, m = map(int, input().split())
maze = []
for _ in range(n):
    row = list(map(int, input().split()))
    maze.append(row)

inQueue = [[False] * m for _ in range(n)]
pre = [[(-1, -1)] * m for _ in range(n)]

BFS(0, 0)
printPath((n - 1, m - 1))
```





### sy322: 跨步迷宫

https://sunnywhy.com/sfbj/8/2/322

现有一个n*m大小的迷宫，其中`1`表示不可通过的墙壁，`0`表示平地。每次移动只能向上下左右移动一格或两格（两格为同向），且只能移动到平地上（不允许跨越墙壁）。求从迷宫左上角到右下角的最小步数（假设移动两格时算作一步）。

**输入**

第一行两个整数 $n、m \hspace{1em} (2 \le n \le 100, 2 \le m \le 100)$，分别表示迷宫的行数和列数；

接下来n行，每行m个整数（值为`0`或`1`），表示迷宫。

**输出**

输出一个整数，表示最小步数。如果无法到达，那么输出`-1`。

样例1

输入

```
3 3
0 1 0
0 0 0
0 1 0
```

输出

```
3
```

解释

假设左上角坐标是，行数增加的方向为增长的方向，列数增加的方向为增长的方向。

可以得到从左上角到右下角的前进路线：=>=>=>。

因此最少需要`3`步。

样例2

输入

```
3 3
0 1 0
0 1 0
0 1 0
```

输出

```
-1
```

解释

显然从左上角无法到达右下角。



```python
from queue import Queue

MAXN = 100
MAXD = 8

dx = [0, 0, 0, 0, 1, -1, 2, -2]
dy = [1, -1, 2, -2, 0, 0, 0, 0]

def canVisit(x, y):
    return x >= 0 and x < n and y >= 0 and y < m and maze[x][y] == 0 and not inQueue[x][y]

def BFS(x, y):
    q = Queue()
    q.put((x, y))
    inQueue[x][y] = True
    step = 0
    while not q.empty():
        cnt = q.qsize()
        while cnt > 0:
            front = q.get()
            cnt -= 1
            if front[0] == n - 1 and front[1] == m - 1:
                return step
            for i in range(MAXD):
                nextX = front[0] + dx[i]
                nextY = front[1] + dy[i]
                nextHalfX = front[0] + dx[i] // 2
                nextHalfY = front[1] + dy[i] // 2
                if canVisit(nextX, nextY) and maze[nextHalfX][nextHalfY] == 0:
                    inQueue[nextX][nextY] = True
                    q.put((nextX, nextY))
        step += 1
    return -1

n, m = map(int, input().split())
maze = []
inQueue = [[False] * m for _ in range(n)]
for _ in range(n):
    maze.append(list(map(int, input().split())))

step = BFS(0, 0)
print(step)
```



### sy323: 字符迷宫

https://sunnywhy.com/sfbj/8/2/323

现有一个n*m大小的迷宫，其中`*`表示不可通过的墙壁，`.`表示平地。每次移动只能向上下左右移动一格，且只能移动到平地上。求从起点`S`到终点`T`的最小步数。

**输入**

第一行两个整数 $n、m \hspace{1em} (2 \le n \le 100, 2 \le m \le 100)$，分别表示迷宫的行数和列数；

接下来n行，每行一个长度为m的字符串，表示迷宫。

**输出**

输出一个整数，表示最小步数。如果无法从`S`到达`T`，那么输出`-1`。

样例1

输入

```
5 5
.....
.*.*.
.*S*.
.***.
...T*
```

输出

```
11
```

解释

假设左上角坐标是，行数增加的方向为增长的方向，列数增加的方向为增长的方向。

起点的坐标为，终点的坐标为。

可以得到从`S`到`T`的前进路线：=>=>=>=>=>=>=>=>=>=>=>。

样例2

输入

复制

```
5 5
.....
.*.*.
.*S*.
.***.
..*T*
```

输出

```
-1
```

解释

显然终点`T`被墙壁包围，无法到达。





```python
from queue import Queue

MAXN = 100
MAXD = 4

dx = [0, 0, 1, -1]
dy = [1, -1, 0, 0]

def canVisit(x, y):
    return x >= 0 and x < n and y >= 0 and y < m and maze[x][y] == 0 and not inQueue[x][y]

def BFS(start, target):
    q = Queue()
    q.put(start)
    inQueue[start[0]][start[1]] = True
    step = 0
    while not q.empty():
        cnt = q.qsize()
        while cnt > 0:
            front = q.get()
            cnt -= 1
            if front == target:
                return step
            for i in range(MAXD):
                nextX = front[0] + dx[i]
                nextY = front[1] + dy[i]
                if canVisit(nextX, nextY):
                    inQueue[nextX][nextY] = True
                    q.put((nextX, nextY))
        step += 1
    return -1

n, m = map(int, input().split())
maze = []
inQueue = [[False] * m for _ in range(n)]
start, target = None, None

for i in range(n):
    row = input()
    maze_row = []
    for j in range(m):
        if row[j] == '.':
            maze_row.append(0)
        elif row[j] == '*':
            maze_row.append(1)
        elif row[j] == 'S':
            start = (i, j)
            maze_row.append(0)
        elif row[j] == 'T':
            target = (i, j)
            maze_row.append(0)
    maze.append(maze_row)

step = BFS(start, target)
print(step)
```



### sy324: 多终点迷宫问题

https://sunnywhy.com/sfbj/8/2/324

现有一个 n*m 大小的迷宫，其中`1`表示不可通过的墙壁，`0`表示平地。每次移动只能向上下左右移动一格，且只能移动到平地上。求从迷宫左上角到迷宫中每个位置的最小步数。

**输入**

第一行两个整数  $n、m \hspace{1em} (2 \le n \le 100, 2 \le m \le 100)$，分别表示迷宫的行数和列数；

接下来n行，每行m个整数（值为`0`或`1`），表示迷宫。

**输出**

输出n行m列个整数，表示从左上角到迷宫中每个位置需要的最小步数。如果无法到达，那么输出`-1`。注意，整数之间用空格隔开，行末不允许有多余的空格。

样例1

输入

```
3 3
0 0 0
1 0 0
0 1 0
```

输出

```
0 1 2
-1 2 3
-1 -1 4
```

解释

假设左上角坐标是，行数增加的方向为增长的方向，列数增加的方向为增长的方向。

可以得到从左上角到所有点的前进路线：=>=>或=>=>。

左下角的三个位置无法到达。



```python
from queue import Queue
import sys

INF = sys.maxsize
MAXN = 100
MAXD = 4

dx = [0, 0, 1, -1]
dy = [1, -1, 0, 0]

def canVisit(x, y):
    return x >= 0 and x < n and y >= 0 and y < m and maze[x][y] == 0 and not inQueue[x][y]

def BFS(x, y):
    minStep = [[-1] * m for _ in range(n)]
    q = Queue()
    q.put((x, y))
    inQueue[x][y] = True
    minStep[x][y] = 0
    step = 0
    while not q.empty():
        cnt = q.qsize()
        while cnt > 0:
            front = q.get()
            cnt -= 1
            for i in range(MAXD):
                nextX = front[0] + dx[i]
                nextY = front[1] + dy[i]
                if canVisit(nextX, nextY):
                    inQueue[nextX][nextY] = True
                    minStep[nextX][nextY] = step + 1
                    q.put((nextX, nextY))
        step += 1
    return minStep

n, m = map(int, input().split())
maze = []
inQueue = [[False] * m for _ in range(n)]

for _ in range(n):
    maze.append(list(map(int, input().split())))

minStep = BFS(0, 0)
for i in range(n):
    #for j in range(m):
    print(' '.join(map(str, minStep[i])))
#        print(minStep[i][j], end='')
#        if j < m - 1:
#            print(' ', end='')
#    print()
```



### sy325: 迷宫问题-传送点

https://sunnywhy.com/sfbj/8/2/325

现有一个n*m大小的迷宫，其中`1`表示不可通过的墙壁，`0`表示平地，`2`表示传送点。每次移动只能向上下左右移动一格，且只能移动到平地或传送点上。当位于传送点时，可以选择传送到另一个`2`处（传送不计入步数），也可以选择不传送。求从迷宫左上角到右下角的最小步数。

**输入**

第一行两个整数$n、m \hspace{1em} (2 \le n \le 100, 2 \le m \le 100)$，分别表示迷宫的行数和列数；

接下来n行，每行m个整数（值为`0`或`1`或`2`），表示迷宫。数据保证有且只有两个`2`，且传送点不会在起始点出现。

**输出**

输出一个整数，表示最小步数。如果无法到达，那么输出`-1`。

样例1

输入

复制

```
3 3
0 1 2
0 1 0
2 1 0
```

输出

```
4
```

解释

假设左上角坐标是，行数增加的方向为增长的方向，列数增加的方向为增长的方向。

可以得到从左上角到右下角的前进路线：=>=>=>=>=>，其中=>属于传送，不计入步数。

因此最少需要`4`步。

样例2

输入

```
3 3
0 1 0
2 1 0
2 1 0
```

输出

```
-1
```

解释

显然从左上角无法到达右下角。



将 transVector 中的第一个位置映射到第二个位置，并将第二个位置映射到第一个位置。这样，就建立了传送门的双向映射关系。

在 BFS 函数中，当遇到传送门时，通过映射表 transMap 找到传送门的另一侧位置，并将其加入队列，以便继续进行搜索。

```python
from queue import Queue

MAXN = 100
MAXD = 4

dx = [0, 0, 1, -1]
dy = [1, -1, 0, 0]

def canVisit(x, y):
    return x >= 0 and x < n and y >= 0 and y < m and (maze[x][y] == 0 or maze[x][y] == 2) and not inQueue[x][y]

def BFS(x, y):
    q = Queue()
    q.put((x, y))
    inQueue[x][y] = True
    step = 0
    while not q.empty():
        cnt = q.qsize()
        while cnt > 0:
            front = q.get()
            cnt -= 1
            if front[0] == n - 1 and front[1] == m - 1:
                return step
            for i in range(MAXD):
                nextX = front[0] + dx[i]
                nextY = front[1] + dy[i]
                if canVisit(nextX, nextY):
                    inQueue[nextX][nextY] = True
                    q.put((nextX, nextY))
                    if maze[nextX][nextY] == 2:
                        transPosition = transMap[(nextX, nextY)]
                        inQueue[transPosition[0]][transPosition[1]] = True
                        q.put(transPosition)
        step += 1
    return -1

n, m = map(int, input().split())
maze = []
inQueue = [[False] * m for _ in range(n)]
transMap = {}
transVector = []
for i in range(n):
    row = list(map(int, input().split()))
    maze.append(row)

    if 2 in row:
        #transVector.append( (i, j) for j, val in enumerate(row) if val == 2)
        for j, val in enumerate(row):
            if val == 2:
                transVector.append((i,j))

        if len(transVector) == 2:
            transMap[transVector[0]] = transVector[1]
            transMap[transVector[1]] = transVector[0]

    #print(transMap)
step = BFS(0, 0)
print(step)
```



### sy326: 中国象棋-马-无障碍

 https://sunnywhy.com/sfbj/8/2/326

现有一个n*m大小的棋盘，在棋盘的第行第列的位置放置了一个棋子，其他位置都未放置棋子。棋子的走位参照中国象棋的“马”。求该棋子到棋盘上每个位置的最小步数。

注：中国象棋中“马”的走位为“日”字形，如下图所示。

![image-20231213160152455](https://raw.githubusercontent.com/GMyhf/img/main/img/image-20231213160152455.png)

**输入**

四个整数$n、m、x、y \hspace{1em} (2 \le n \le 100, 2 \le m \le 100, 1 \le x \le n, 1\le y \le m)$，分别表示棋盘的行数和列数、棋子的所在位置。

**输出**

输出行列个整数，表示从棋子到棋盘上每个位置需要的最小步数。如果无法到达，那么输出`-1`。注意，整数之间用空格隔开，行末不允许有多余的空格。

样例1

输入

```
3 3 2 1
```

输出

```
3 2 1
0 -1 4
3 2 1
```

解释

共`3`行`3`列，“马”在第`2`行第`1`列的位置，由此可得“马”能够前进的路线如下图所示。

<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/image-20231213160421486.png" alt="image-20231213160421486" style="zoom:67%;" />





```python
from collections import deque

MAXN = 100
MAXD = 8

dx = [-2, -1, 1, 2, -2, -1, 1, 2]
dy = [1, 2, 2, 1, -1, -2, -2, -1]

def canVisit(x, y):
    return 0 <= x < n and 0 <= y < m and not inQueue[x][y]

def BFS(x, y):
    minStep = [[-1] * m for _ in range(n)]
    queue = deque()
    queue.append((x, y))
    inQueue[x][y] = True
    minStep[x][y] = 0
    step = 0
    while queue:
        cnt = len(queue)
        while cnt > 0:
            front = queue.popleft()
            cnt -= 1
            for i in range(MAXD):
                nextX = front[0] + dx[i]
                nextY = front[1] + dy[i]
                if canVisit(nextX, nextY):
                    inQueue[nextX][nextY] = True
                    minStep[nextX][nextY] = step + 1
                    queue.append((nextX, nextY))
        step += 1
    return minStep


n, m, x, y = map(int, input().split())
inQueue = [[False] * m for _ in range(n)]
minStep = BFS(x - 1, y - 1)
for row in minStep:
    print(' '.join(map(str, row)))
```



### sy327: 中国象棋-马-有障碍

https://sunnywhy.com/sfbj/8/2/327

现有一个大小的棋盘，在棋盘的第行第列的位置放置了一个棋子，其他位置中的一部分放置了障碍棋子。棋子的走位参照中国象棋的“马”（障碍棋子将成为“马脚”）。求该棋子到棋盘上每个位置的最小步数。

注`1`：中国象棋中“马”的走位为“日”字形，如下图所示。

![中国象棋-马-有障碍_题目描述1.png](https://raw.githubusercontent.com/GMyhf/img/main/img/405270a4-8a80-4837-891a-d0d05cc5577c.png)

注`2`：与“马”**直接相邻**的棋子会成为“马脚”，“马”不能往以“马”=>“马脚”为**长边**的方向前进，如下图所示。

![中国象棋-马-有障碍_题目描述2.png](https://raw.githubusercontent.com/GMyhf/img/main/img/0b79f8a0-7b3e-4675-899c-b44e86ee5e40.png)

**输入**

第一行四个整数$n、m、x、y \hspace{1em} (2 \le n \le 100, 2 \le m \le 100, 1 \le x \le n, 1\le y \le m)$，分别表示棋盘的行数和列数、棋子的所在位置；

第二行一个整数$k（1 \le k \le 10）$，表示障碍棋子的个数；

接下来k行，每行两个整数$x_i、y_i（1 \le x_i \le n, 1 \le y_i \le m）$，表示第i个障碍棋子的所在位置。数据保证不存在相同位置的障碍棋子。

**输出**

输出n行m列个整数，表示从棋子到棋盘上每个位置需要的最小步数。如果无法到达，那么输出`-1`。注意，整数之间用空格隔开，行末不允许有多余的空格。

样例1

输入

复制

```
3 3 2 1
1
1 2
```

输出

复制

```
3 -1 1
0 -1 -1
-1 2 1
```

解释

共`3`行`3`列，“马”在第`2`行第`1`列的位置，障碍棋子在第`1`行第`2`列的位置，由此可得“马”能够前进的路线如下图所示。

<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/f005a3c6-b042-471b-b10f-26daf7ff97fb.png" alt="中国象棋-马-有障碍_样例.png" style="zoom:67%;" />



```python
from collections import deque

MAXD = 8
dx = [-2, -1, 1, 2, -2, -1, 1, 2]
dy = [1, 2, 2, 1, -1, -2, -2, -1]


def canVisit(x, y):
    return x >= 0 and x < n and y >= 0 and y < m and not isBlock.get((x, y), False) and not inQueue[x][y]


def BFS(x, y):
    minStep = [[-1] * m for _ in range(n)]
    queue = deque()
    queue.append((x, y))
    inQueue[x][y] = True
    minStep[x][y] = 0
    step = 0
    while queue:
        cnt = len(queue)
        for _ in range(cnt):
            front = queue.popleft()
            for i in range(MAXD):
                nextX = front[0] + dx[i]
                nextY = front[1] + dy[i]
                if dx[i] == -1 and dy[i] == -1: #如果dx=-1，-1//2=-1，期望得到0
                    footX, footY = front[0], front[1]
                elif dx[i] == -1 and dy[i] != -1:
                    footX, footY = front[0], front[1] + dy[i] // 2
                elif dx[i] != -1 and dy[i] == -1:
                    footX, footY = front[0] + dx[i] // 2, front[1]
                else:
                    footX, footY = front[0] + dx[i] // 2, front[1] + dy[i] // 2

                if canVisit(nextX, nextY) and not isBlock.get((footX, footY), False):
                    inQueue[nextX][nextY] = True
                    minStep[nextX][nextY] = step + 1
                    queue.append((nextX, nextY))


        step += 1
    return minStep

n, m, x, y = map(int, input().split())
inQueue = [[False] * m for _ in range(n)]
isBlock = {}

k = int(input())
for _ in range(k):
    blockX, blockY = map(int, input().split())
    isBlock[(blockX - 1, blockY - 1)] = True

minStep = BFS(x - 1, y - 1)

for row in minStep:
    print(' '.join(map(str, row)))
```





# 树专题（46题）

## 1 树与二叉树 1题

### 1.1 树的判定

https://sunnywhy.com/sfbj/9/1

现有一个由个结点连接而成的**连通**结构，已知这个结构中存在的边数，问这个连通结构是否是一棵树。

**输入**

两个整数$n、m（1 \le n \le 100, 0 \le m \le 100）$，分别表示结点数和边数。

**输出**

如果是一棵树，那么输出`Yes`，否则输出`No`。

样例1

输入

```
2 1
```

输出

```
Yes
```

解释

两个结点，一条边，显然是一棵树。

样例2

输入

```
2 0
```

输出

```
No
```

解释

两个结点，没有边，显然不是树。



```python
def is_tree(nodes, edges):
    if nodes - 1 == edges:
        return 'Yes'
    else:
        return 'No'

if __name__ == "__main__":
    n, m = map(int, input().split())
    print(is_tree(n, m))
```



## 2 二叉树的遍历 16题

### 2.1 二叉树的先序遍历

https://sunnywhy.com/sfbj/9/2

现有一棵个结点的二叉树（结点编号为从`0`到`n-1`，根结点为`0`号结点），求这棵二叉树的先序遍历序列。

**输入**

第一行一个整数 $n (1 \le n \le 50)$，表示二叉树的结点个数；

接下来行，每行一个结点，按顺序给出编号从`0`到`n-1`的结点的左子结点编号和右子结点编号，中间用空格隔开。如果不存在对应的子结点，那么用`-1`表示。

**输出**

输出个整数，表示先序遍历序列，中间用空格隔开，行末不允许有多余的空格。

样例1

输入

```
6
2 5
-1 -1
1 4
-1 -1
-1 -1
-1 3
```

输出

```
0 2 1 4 5 3
```

解释

对应的二叉树如下图所示，先序序列为`0 2 1 4 5 3`。

![二叉树的先序遍历.png](https://raw.githubusercontent.com/GMyhf/img/main/img/b875230c-1a81-4e44-8512-0a014b092745.png)



```python
from collections import deque

class Node():
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def traversal(self, mode):
        result = []
        if mode == "preorder":
            result.append(self.val)
            if self.left is not None:
                result.extend(self.left.traversal(mode))
            if self.right is not None:
                result.extend(self.right.traversal(mode))
            return result
        elif mode == "postorder":
            if self.left is not None:
                result.extend(self.left.traversal(mode))
            if self.right is not None:
                result.extend(self.right.traversal(mode))
            result.append(self.val)
            return result
        elif mode == "inorder":
            if self.left is not None:
                result.extend(self.left.traversal(mode))
            result.append(self.val)
            if self.right is not None:
                result.extend(self.right.traversal(mode))
            return result
        elif mode == "levelorder":
            queue = deque([self])
            while queue:
                node = queue.popleft()
                result.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            return result
          
	def tree_height(self):
    if self is None:
        return -1  # 根据定义，空树高度为-1
    return max(tree_height(self.left), tree_height(self.right)) + 1


n = int(input())
nodes = [Node(i) for i in range(n)]
for i in range(n):
    left, right = map(int, input().split())
    nodes[i].left = nodes[left] if left != -1 else None
    nodes[i].right = nodes[right] if right != -1 else None

# "preorder", "postorder", "inorder, "levelorder"
mode = "levelorder"
pt = nodes[0]
result = pt.traversal(mode)

print(*result)
```





### 2.2 二叉树的中序遍历



mode = "preorder"



### 2.3 二叉树的后序遍历



mode = "postorder"



### 2.4 二叉树的层次遍历



mode = "levelorder"

```python

```





### 2.5 二叉树的高度

层级 Level：从根节点开始到达一个节点的路径，所包含的边的数量，称为这个节点的层级。根节点的层级为 0。

高度 Height：树中所有节点的最大层级称为树的高度。因此空树的高度是-1。



```python
from collections import deque

class Node():
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def traversal(self, mode):
        result = []
        if mode == "preorder":
            result.append(self.val)
            if self.left is not None:
                result.extend(self.left.traversal(mode))
            if self.right is not None:
                result.extend(self.right.traversal(mode))
            return result
        elif mode == "postorder":
            if self.left is not None:
                result.extend(self.left.traversal(mode))
            if self.right is not None:
                result.extend(self.right.traversal(mode))
            result.append(self.val)
            return result
        elif mode == "inorder":
            if self.left is not None:
                result.extend(self.left.traversal(mode))
            result.append(self.val)
            if self.right is not None:
                result.extend(self.right.traversal(mode))
            return result
        elif mode == "levelorder":
            queue = deque([self])
            while queue:
                node = queue.popleft()
                result.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            return result

    def height(self):
        if self.left is None and self.right is None:
            return 0
        left_height = self.left.height() if self.left else 0
        right_height = self.right.height() if self.right else 0
        return max(left_height, right_height) + 1


n = int(input())
nodes = [Node(i) for i in range(n)]
has_parent = [False] * n  # 用来标记节点是否有父节点

for i in range(n):
    left, right = map(int, input().split())
    if left != -1:
        nodes[i].left = nodes[left]
        has_parent[left] = True
    if right != -1:
        nodes[i].right = nodes[right]
        has_parent[right] = True

# 寻找根节点，也就是没有父节点的节点
root_index = has_parent.index(False)
root = nodes[root_index]

# "preorder", "postorder", "inorder, "levelorder"
# mode = "levelorder"
# result = root.traversal(mode)
# print(*result)

"""
6
2 5
-1 -1
1 4
-1 -1
-1 -1
-1 3
"""
print(root.height())
# 2
```



### 2.6 二叉树的结点层号

https://sunnywhy.com/sfbj/9/2/334

现有一棵个结点的二叉树（结点编号为从`0`到`n-1`，根结点为`0`号结点），求这棵二叉树所有结点的层号（假设根结点的层号为`1`）。

**输入**

第一行一个整数 n (1<=n<=50)，表示二叉树的结点个数；

接下来行，每行一个结点，按顺序给出编号从`0`到`n-1`的结点的左子结点编号和右子结点编号，中间用空格隔开。如果不存在对应的子结点，那么用`-1`表示。

**输出**

输出个整数，分别表示编号从`0`到`n-1`的结点的层号，中间用空格隔开，行末不允许有多余的空格。

样例1

输入

```
6
2 5
-1 -1
1 4
-1 -1
-1 -1
-1 3
```

输出

```
1 3 2 3 3 2
```

解释

对应的二叉树如下图所示，层号为`1`的结点编号为`0`，层号为`2`的结点编号为`2`、`5`，层号为`3`的结点编号为`1`、`4`、`3`。

![二叉树的先序遍历.png](https://raw.githubusercontent.com/GMyhf/img/main/img/b875230c-1a81-4e44-8512-0a014b092745.png)



```python
from collections import deque

def node_levels(n, nodes):
    levels = [0] * n
    queue = deque([(0, 1)])  # (node, level)

    while queue:
        node, level = queue.popleft()
        levels[node] = level
        left, right = nodes[node]
        if left != -1:
            queue.append((left, level + 1))
        if right != -1:
            queue.append((right, level + 1))

    return levels

n = int(input())
nodes = [[left, right] for left, right in [map(int, input().split()) for _ in range(n)]]

print(*node_levels(n, nodes))
```



### 2.7 翻转二叉树

https://sunnywhy.com/sfbj/9/2/335

现有一棵个结点的二叉树（结点编号为从`0`到`n-1`，根结点为`0`号结点），将这棵二叉树中每个结点的左右子树交换，输出新的二叉树的先序序列和中序序列。

**输入**

第一行一个整数 n (1<=n<=50)，表示二叉树的结点个数；

接下来行，每行一个结点，按顺序给出编号从`0`到`n-1`的结点的左子结点编号和右子结点编号，中间用空格隔开。如果不存在对应的子结点，那么用`-1`表示。

**输出**

输出两行，第一行为先序序列，第二行为中序序列。整数之间用空格隔开，行末不允许有多余的空格。

样例1

输入

```
6
2 5
-1 -1
1 4
-1 -1
-1 -1
-1 3
```

输出

```
0 5 3 2 4 1
3 5 0 4 2 1
```

解释

对应的二叉树和翻转后的二叉树如下图所示。

![翻转二叉树.png](https://raw.githubusercontent.com/GMyhf/img/main/img/93380dc9-4690-45ac-8b42-d8347bc14fc4.png)



```python
class Node():
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def flip_tree(node):
    if node is None:
        return
    node.left, node.right = node.right, node.left
    flip_tree(node.left)
    flip_tree(node.right)

def preorder_traversal(node):
    if node is None:
        return []
    return [node.val] + preorder_traversal(node.left) + preorder_traversal(node.right)

def inorder_traversal(node):
    if node is None:
        return []
    return inorder_traversal(node.left) + [node.val] + inorder_traversal(node.right)

n = int(input())
nodes = [Node(i) for i in range(n)]
for i in range(n):
    left, right = map(int, input().split())
    nodes[i].left = nodes[left] if left != -1 else None
    nodes[i].right = nodes[right] if right != -1 else None

flip_tree(nodes[0])

print(*preorder_traversal(nodes[0]))
print(*inorder_traversal(nodes[0]))
```





### 2.8 先序中序还原二叉树

https://sunnywhy.com/sfbj/9/2/336

现有一棵个结点的二叉树（结点编号为从`0`到`n-1`），已知其先序序列和中序序列，求后序序列。

**输入**

第一行一个整数 n (1<=n<=50)，表示二叉树的结点个数；

第二行为个整数，表示二叉树的先序序列；

第三行为个整数，表示二叉树的中序序列。

**输出**

输出个整数，表示二叉树的后序序列，中间用空格隔开，行末不允许有多余的空格。

样例1

输入

```
6
0 2 1 4 5 3
1 2 4 0 5 3
```

输出

```
1 4 2 3 5 0
```

解释

对应的二叉树如下图所示，其后序序列为`1 4 2 3 5 0`。

![二叉树的先序遍历.png](https://raw.githubusercontent.com/GMyhf/img/main/img/aaaa2905-d60b-4ca6-b445-d7c2600df176.png)

```python
class Node():
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def build_tree(preorder, inorder):
    if inorder:
        index = inorder.index(preorder.pop(0))
        node = Node(inorder[index])
        node.left = build_tree(preorder, inorder[0:index])
        node.right = build_tree(preorder, inorder[index+1:])
        return node

def postorder_traversal(node):
    if node is None:
        return []
    return postorder_traversal(node.left) + postorder_traversal(node.right) + [node.val]

n = int(input())
preorder = list(map(int, input().split()))
inorder = list(map(int, input().split()))

root = build_tree(preorder, inorder)
print(*postorder_traversal(root))
```





### 2.9 后序中序还原二叉树

https://sunnywhy.com/sfbj/9/2/337

现有一棵个结点的二叉树（结点编号为从`0`到`n-1`），已知其后序序列和中序序列，求先序序列。

**输入**

第一行一个整数 n (1<=n<=50)，表示二叉树的结点个数；

第二行为个整数，表示二叉树的后序序列；

第三行为个整数，表示二叉树的中序序列。

**输出**

输出个整数，表示二叉树的先序序列，中间用空格隔开，行末不允许有多余的空格。

样例1

输入

```
6
1 4 2 3 5 0
1 2 4 0 5 3
```

输出

```
0 2 1 4 5 3
```

解释

对应的二叉树如下图所示，其先序序列为`0 2 1 4 5 3`。

![二叉树的先序遍历.png](https://raw.githubusercontent.com/GMyhf/img/main/img/3a704577-f914-4fa8-812c-13b79ac9d104.png)

```python
class Node():
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def build_tree(postorder, inorder):
    if inorder:
        index = inorder.index(postorder.pop())
        node = Node(inorder[index])
        node.right = build_tree(postorder, inorder[index+1:])
        node.left = build_tree(postorder, inorder[0:index])
        return node

def preorder_traversal(node):
    if node is None:
        return []
    return [node.val] + preorder_traversal(node.left) + preorder_traversal(node.right)

n = int(input())
postorder = list(map(int, input().split()))
inorder = list(map(int, input().split()))

root = build_tree(postorder, inorder)
print(*preorder_traversal(root))
```



### 2.10 层序中序还原二叉树

https://sunnywhy.com/sfbj/9/2/338

现有一棵个结点的二叉树（结点编号为从`0`到`n-1`），已知其层序序列和中序序列，求先序序列。

**输入**

第一行一个整数n (1<=n<=50)，表示二叉树的结点个数；

第二行为个整数，表示二叉树的层序序列；

第三行为个整数，表示二叉树的中序序列。

**输出**

输出个整数，表示二叉树的先序序列，中间用空格隔开，行末不允许有多余的空格。

样例1

输入

```
6
0 2 5 1 4 3
1 2 4 0 5 3
```

输出

```
0 2 1 4 5 3
```

解释

对应的二叉树如下图所示，其先序序列为`0 2 1 4 5 3`。

![二叉树的先序遍历.png](https://raw.githubusercontent.com/GMyhf/img/main/img/7d88fa4c-28ef-4aca-84d4-5f16bc147517.png)



```python
class Node():
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def build_tree(levelorder, inorder):
    if inorder:
        for i in range(0, len(levelorder)):
            if levelorder[i] in inorder:
                node = Node(levelorder[i])
                io_index = inorder.index(levelorder[i])
                break
        node.left = build_tree(levelorder, inorder[0:io_index])
        node.right = build_tree(levelorder, inorder[io_index+1:])
        return node

def preorder_traversal(node):
    if node is None:
        return []
    return [node.val] + preorder_traversal(node.left) + preorder_traversal(node.right)

n = int(input())
levelorder = list(map(int, input().split()))
inorder = list(map(int, input().split()))

root = build_tree(levelorder, inorder)
print(*preorder_traversal(root))
```



### 2.11 二叉树的最近公共祖先

https://sunnywhy.com/sfbj/9/2/339

现有一棵个结点的二叉树（结点编号为从`0`到`n-1`，根结点为`0`号结点），求两个指定编号结点的最近公共祖先。

注：二叉树上两个结点A、B的最近公共祖先是指：二叉树上存在的一个结点，使得既是的祖先，又是的祖先，并且需要离根结点尽可能远（即层号尽可能大）。

**输入**

第一行三个整数$n、k_1、k_2 (1 \le n \le 50, 0 \le k_1 \le n-1, 0 \le k_2 \le n-1)$，分别表示二叉树的结点个数、需要求最近公共祖先的两个结点的编号；

接下来行，每行一个结点，按顺序给出编号从`0`到`n-1`的结点的左子结点编号和右子结点编号，中间用空格隔开。如果不存在对应的子结点，那么用`-1`表示。

**输出**

输出一个整数，表示最近公共祖先的编号。

样例1

输入

```
6 1 4
2 5
-1 -1
1 4
-1 -1
-1 -1
-1 3
```

输出

```
2
```

解释

对应的二叉树如下图所示，结点`1`和结点`4`的公共祖先有结点`2`和结点`0`，其中结点`2`是最近公共祖先。

![二叉树的先序遍历.png](https://raw.githubusercontent.com/GMyhf/img/main/img/b875230c-1a81-4e44-8512-0a014b092745.png)



```python
class Node():
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def find_path(root, path, k):
    if root is None:
        return False
    path.append(root.val)
    if root.val == k:
        return True
    if ((root.left != None and find_path(root.left, path, k)) or
            (root.right!= None and find_path(root.right, path, k))):
        return True
    path.pop()
    return False

def find_LCA(root, n1, n2):
    path1 = []
    path2 = []
    if (not find_path(root, path1, n1) or not find_path(root, path2, n2)):
        return -1
    i = 0
    while(i < len(path1) and i < len(path2)):
        if path1[i] != path2[i]:
            break
        i += 1
    return path1[i-1]

n, n1, n2 = map(int, input().split())
nodes = [Node(i) for i in range(n)]
for i in range(n):
    left, right = map(int, input().split())
    nodes[i].left = nodes[left] if left != -1 else None
    nodes[i].right = nodes[right] if right != -1 else None

print(find_LCA(nodes[0], n1, n2))
```



### 2.12 二叉树的路径和

https://sunnywhy.com/sfbj/9/2/340

现有一棵个结点的二叉树（结点编号为从`0`到`n-1`，根结点为`0`号结点），每个结点有各自的权值。

1. 结点的路径和是指，从根结点到该结点的路径上所有结点的权值之和；
2. 二叉树的路径和是指，二叉树所有叶结点的路径和之和。

求这棵二叉树的路径和。

**输入**

第一行一个整数n (1<=n<=50)，表示二叉树的结点个数；

第二行个整数，分别给出编号从`0`到`n-1`的个结点的权值（）；

接下来行，每行一个结点，按顺序给出编号从`0`到`n-1`的结点的左子结点编号和右子结点编号，中间用空格隔开。如果不存在对应的子结点，那么用`-1`表示。

**输出**

输出一个整数，表示二叉树的路径和。

样例1

输入

```
6
3 2 1 5 1 2
2 5
-1 -1
1 4
-1 -1
-1 -1
-1 3
```

输出

```
21
```

解释

对应的二叉树如下图所示，其中黑色数字为结点编号，编号右下角的灰色数字为结点权值。由此可得叶结点`1`的路径和为，叶结点`4`的路径和为，叶结点`3`的路径和为，因此二叉树的路径和为。

![二叉树的路径和.png](https://raw.githubusercontent.com/GMyhf/img/main/img/061c3f04-4557-4ab1-aec3-5c563c7e1e5d.png)



```python
class Node():
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def path_sum(node, current_sum=0):
    if node is None:
        return 0
    current_sum += node.val
    if node.left is None and node.right is None:
        return current_sum
    return path_sum(node.left, current_sum) + path_sum(node.right, current_sum)

n = int(input())
values = list(map(int, input().split()))
nodes = [Node(values[i]) for i in range(n)]
for i in range(n):
    left, right = map(int, input().split())
    nodes[i].left = nodes[left] if left != -1 else None
    nodes[i].right = nodes[right] if right != -1 else None

print(path_sum(nodes[0]))
```



### 2.13 二叉树的带权路径长度

https://sunnywhy.com/sfbj/9/2/341

现有一棵个结点的二叉树（结点编号为从`0`到`n-1`，根结点为`0`号结点），每个结点有各自的权值。

1. 结点的路径长度是指，从根结点到该结点的边数；
2. 结点的带权路径长度是指，结点权值乘以结点的路径长度；
3. 二叉树的带权路径长度是指，二叉树所有叶结点的带权路径长度之和。

求这棵二叉树的带权路径长度。

**输入**

第一行一个整数 n (1<=n<=50)，表示二叉树的结点个数；

第二行个整数，分别给出编号从`0`到`n-1`的个结点的权值（）；

接下来行，每行一个结点，按顺序给出编号从`0`到`n-1`的结点的左子结点编号和右子结点编号，中间用空格隔开。如果不存在对应的子结点，那么用`-1`表示。

**输出**

输出一个整数，表示二叉树的带权路径长度。

样例1

输入

```
5
2 3 1 2 1
2 3
-1 -1
1 4
-1 -1
-1 -1
```

输出

```
10
```

解释

对应的二叉树如下图所示，其中黑色数字为结点编号，编号右下角的格式为`结点权值*结点路径长度=结点带权路径长度`。由此可得二叉树的带权路径长度为。

![二叉树的带权路径长度.png](https://raw.githubusercontent.com/GMyhf/img/main/img/835e6c0a-d265-4d6b-b484-c2261915cc22.png)



```python
class TreeNode:
    def __init__(self, value=0):
        self.value = value
        self.left = None
        self.right = None

def build_tree(weights, edges):
    # 根据边构建二叉树，并返回根节点
    nodes = [TreeNode(w) for w in weights]
    for i, (left, right) in enumerate(edges):
        if left != -1:
            nodes[i].left = nodes[left]
        if right != -1:
            nodes[i].right = nodes[right]
    return nodes[0] if nodes else None

def weighted_path_length(node, depth=0):
    # 计算带权路径长度
    if not node:
        return 0
    # 如果是叶子节点，返回其带权路径长度
    if not node.left and not node.right:
        return node.value * depth
    # 否则递归计算左右子树的带权路径长度
    return weighted_path_length(node.left, depth + 1) + weighted_path_length(node.right, depth + 1)

# 输入处理
n = int(input())  # 节点个数
weights = list(map(int, input().split()))  # 各节点权值
edges = [list(map(int, input().split())) for _ in range(n)]  # 节点边

# 构建二叉树
root = build_tree(weights, edges)

# 计算并输出带权路径长度
print(weighted_path_length(root))

```



### 2.14 二叉树的左视图序列

https://sunnywhy.com/sfbj/9/2/342

现有一棵个结点的二叉树（结点编号为从`0`到`n-1`，根结点为`0`号结点），从二叉树的左侧看去，同一层的多个结点只能看到这层中最左边的结点，这些能看到的结点从上到下组成的序列称为左视图序列。求这棵二叉树的左视图序列。

**输入**

第一行一个整数 n (1<=n<=50)，表示二叉树的结点个数；

接下来行，每行一个结点，按顺序给出编号从`0`到`n-1`的结点的左子结点编号和右子结点编号，中间用空格隔开。如果不存在对应的子结点，那么用`-1`表示。

**输出**

输出若干个整数，表示左视图序列，中间用空格隔开，行末不允许有多余的空格。

样例1

输入

```
6
2 5
-1 -1
1 4
-1 -1
-1 -1
-1 3
```

输出

```
0 2 1
```

解释

对应的二叉树如下图所示，从左侧看去，第一层可以看到结点`0`，第二层可以看到结点`2`，第三层可以看到结点`1`，因此左视图序列是`0 2 1`。

![二叉树的先序遍历.png](https://raw.githubusercontent.com/GMyhf/img/main/img/b875230c-1a81-4e44-8512-0a014b092745.png)



```python
class Node():
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def left_view(root):
    if root is None:
        return []
    queue = [root]
    view = [root.val]
    while queue:
        level = []
        for node in queue:
            if node.left:
                level.append(node.left)
            if node.right:
                level.append(node.right)
        if level:
            view.append(level[0].val)
        queue = level
    return view

n = int(input())
nodes = [Node(i) for i in range(n)]
for i in range(n):
    left, right = map(int, input().split())
    nodes[i].left = nodes[left] if left != -1 else None
    nodes[i].right = nodes[right] if right != -1 else None

print(*left_view(nodes[0]))
```



### 2.15 满二叉树的判定

https://sunnywhy.com/sfbj/9/2/343

现有一棵个结点的二叉树（结点编号为从`0`到`n-1`，根结点为`0`号结点），判断这个二叉树是否是满二叉树。

注：如果一棵二叉树每一层的结点数都达到了当层能达到的最大结点数（即如果二叉树的层数为，且结点总数为 $2^k-1$），那么称这棵二叉树为满二叉树。

**输入**

第一行一个整数n (1<=n<=64)，表示二叉树的结点个数；

接下来行，每行一个结点，按顺序给出编号从`0`到`n-1`的结点的左子结点编号和右子结点编号，中间用空格隔开。如果不存在对应的子结点，那么用`-1`表示。

**输出**

如果是满二叉树，那么输出`Yes`，否则输出`No`。

样例1

输入

```
7
2 5
-1 -1
1 4
-1 -1
-1 -1
6 3
-1 -1
```

输出

```
Yes
```

解释

对应的二叉树如下图所示，是满二叉树。

![满二叉树的判定.png](https://raw.githubusercontent.com/GMyhf/img/main/img/f5ce47ce-813f-47bc-badb-1bdf7e2b95e2.png)



```python
class Node():
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def is_full(node):
    if node is None:
        return True
    if (node.left is None and node.right is None) or (node.left is not None and node.right is not None):
        return is_full(node.left) and is_full(node.right)
    return False

n = int(input())
nodes = [Node(i) for i in range(n)]
for i in range(n):
    left, right = map(int, input().split())
    nodes[i].left = nodes[left] if left != -1 else None
    nodes[i].right = nodes[right] if right != -1 else None

print("Yes" if is_full(nodes[0]) else "No")
```



### 2.16 完全二叉树的判定

https://sunnywhy.com/sfbj/9/2/344

现有一棵个结点的二叉树（结点编号为从`0`到`n-1`，根结点为`0`号结点），判断这个二叉树是否是完全二叉树。

注：如果一棵二叉树除了最下面一层之外，其余层的结点个数都达到了当层能达到的最大结点数，且最下面一层只从左至右连续存在若干结点，而这些连续结点右边不存在别的结点，那么就称这棵二叉树为完全二叉树。

**输入**

第一行一个整数n (1<=n<=64)，表示二叉树的结点个数；

接下来行，每行一个结点，按顺序给出编号从`0`到`n-1`的结点的左子结点编号和右子结点编号，中间用空格隔开。如果不存在对应的子结点，那么用`-1`表示。

**输出**

如果是完全二叉树，那么输出`Yes`，否则输出`No`。

样例1

输入

```
6
2 5
-1 -1
1 4
-1 -1
-1 -1
3 -1
```

输出

```
Yes
```

解释

对应的二叉树如下图所示，是完全二叉树。

![完全二叉树的判定.png](https://raw.githubusercontent.com/GMyhf/img/main/img/b51c5d62-9753-46dd-a57c-60658c32847a.png)



```python
class Node():
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def is_complete(root):
    if root is None:
        return True
    queue = [root]
    flag = False
    while queue:
        node = queue.pop(0)
        if node.left:
            if flag:  # If we have seen a node with a missing right or left child
                return False
            queue.append(node.left)
        else:
            flag = True
        if node.right:
            if flag:  # If we have seen a node with a missing right or left child
                return False
            queue.append(node.right)
        else:
            flag = True
    return True

n = int(input())
nodes = [Node(i) for i in range(n)]
for i in range(n):
    left, right = map(int, input().split())
    nodes[i].left = nodes[left] if left != -1 else None
    nodes[i].right = nodes[right] if right != -1 else None

print("Yes" if is_complete(nodes[0]) else "No")
```





## 3 树的遍历 7题

### 3.1 树的先根遍历

https://sunnywhy.com/sfbj/9/3

现有一棵个结点的树（结点编号为从`0`到`n-1`，根结点为`0`号结点），求这棵树的先根遍历序列。

**输入**

第一行一个整数 $n (1 \le n \le 50)$，表示树的结点个数；

接下来行，每行一个结点的子结点信息，格式如下：

```text
k child_1 child_2 ... child_k
```

其中k表示该结点的子结点个数，child1、child2、...、childk表示子结点的编号。

**输出**

输出个整数，表示先根遍历序列，中间用空格隔开，行末不允许有多余的空格。

样例1

输入

```
7
3 2 4 5
0
2 1 6
0
0
1 3
0
```

输出

```
0 2 1 6 4 5 3
```

解释

对应的树如下图所示，先根遍历序列为`0 2 1 6 4 5 3`。

![树的先根遍历.png](https://raw.githubusercontent.com/GMyhf/img/main/img/6259b6fa-bd37-4ade-9e7f-6eefc1e1af74.png)





```python
class Node():
    def __init__(self, val, children=None):
        self.val = val
        self.children = children if children is not None else []

def pre_order(node):
    if node is None:
        return []
    result = [node.val]
    for child in node.children:
        result.extend(pre_order(child))
    return result

n = int(input())
nodes = [Node(i) for i in range(n)]
for i in range(n):
    children = list(map(int, input().split()))[1:]
    nodes[i].children = [nodes[child] for child in children]

print(*pre_order(nodes[0]))
```



### 3.2 树的后根遍历

https://sunnywhy.com/sfbj/9/3/346

现有一棵个结点的树（结点编号为从`0`到`n-1`，根结点为`0`号结点），求这棵树的后根遍历序列。

**输入**

第一行一个整数 $n (1 \le n \le 50)$， 表示树的结点个数；

接下来行，每行一个结点的子结点信息，格式如下：

```text
k child_1 child_2 ... child_k
```

其中k表示该结点的子结点个数，child1、child2、...、childk表示子结点的编号。

**输出**

输出个整数，表示后根遍历序列，中间用空格隔开，行末不允许有多余的空格。

样例1

输入

```
7
3 2 4 5
0
2 1 6
0
0
1 3
0
```

输出

```
1 6 2 4 3 5 0
```

解释

对应的树如下图所示，后根遍历序列为`1 6 2 4 3 5 0`。

![树的先根遍历.png](https://raw.githubusercontent.com/GMyhf/img/main/img/6259b6fa-bd37-4ade-9e7f-6eefc1e1af74.png)



```python
class Node():
    def __init__(self, val, children=None):
        self.val = val
        self.children = children if children is not None else []

def post_order(node):
    if node is None:
        return []
    result = []
    for child in node.children:
        result.extend(post_order(child))
    result.append(node.val)
    return result

n = int(input())
nodes = [Node(i) for i in range(n)]
for i in range(n):
    children = list(map(int, input().split()))[1:]
    nodes[i].children = [nodes[child] for child in children]

print(*post_order(nodes[0]))
```



### 3.3 树的层序遍历

https://sunnywhy.com/sfbj/9/3/347

现有一棵个结点的树（结点编号为从`0`到`n-1`，根结点为`0`号结点），求这棵树的层序遍历序列。

**输入**

第一行一个整数 $n (1 \le n \le 50)$，表示树的结点个数；

接下来行，每行一个结点的子结点信息，格式如下：

```text
k child_1 child_2 ... child_k
```

其中k表示该结点的子结点个数，child1、child2、...、childk表示子结点的编号。

**输出**

输出个整数，表示层序遍历序列，中间用空格隔开，行末不允许有多余的空格。

样例1

输入

```
7
3 2 4 5
0
2 1 6
0
0
1 3
0
```

输出

```
0 2 4 5 1 6 3
```

解释

对应的树如下图所示，层序遍历序列为`0 2 4 5 1 6 3`。

![树的先根遍历.png](https://raw.githubusercontent.com/GMyhf/img/main/img/6259b6fa-bd37-4ade-9e7f-6eefc1e1af74.png)



```python
class Node():
    def __init__(self, val, children=None):
        self.val = val
        self.children = children if children is not None else []

def level_order(root):
    if root is None:
        return []
    queue = [root]
    traversal = []
    while queue:
        node = queue.pop(0)
        traversal.append(node.val)
        queue.extend(node.children)
    return traversal

n = int(input())
nodes = [Node(i) for i in range(n)]
for i in range(n):
    children = list(map(int, input().split()))[1:]
    nodes[i].children = [nodes[child] for child in children]

print(*level_order(nodes[0]))
```



### 3.4 树的高度

https://sunnywhy.com/sfbj/9/3/348

现有一棵个结点的树（结点编号为从`0`到`n-1`，根结点为`0`号结点），求这棵树的高度。

**输入**

第一行一个整数 $n (1 \le n \le 50)$，表示树的结点个数；

接下来行，每行一个结点的子结点信息，格式如下：

```text
k child_1 child_2 ... child_k
```

其中k表示该结点的子结点个数，child1、child2、...、childk表示子结点的编号。

**输出**

输出一个整数，表示树的高度。

样例1

输入

```
7
3 2 4 5
0
2 1 6
0
0
1 3
0
```

输出

```
3
```

解释

对应的树如下图所示，高度为`3`。

![树的先根遍历.png](https://raw.githubusercontent.com/GMyhf/img/main/img/6259b6fa-bd37-4ade-9e7f-6eefc1e1af74.png)



```python
class Node():
    def __init__(self, val, children=None):
        self.val = val
        self.children = children if children is not None else []

def height(node):
    if node is None:
        return 0
    if not node.children:
        return 1
    return max(height(child) for child in node.children) + 1

n = int(input())
nodes = [Node(i) for i in range(n)]
for i in range(n):
    children = list(map(int, input().split()))[1:]
    nodes[i].children = [nodes[child] for child in children]

print(height(nodes[0]))
```



### 3.5 树的结点层号

https://sunnywhy.com/sfbj/9/3/349

现有一棵个结点的树（结点编号为从`0`到`n-1`，根结点为`0`号结点），求这棵树的所有结点的层号（假设根结点的层号为`1`）。

**输入**

第一行一个整数 $n (1 \le n \le 50)$，表示树的结点个数；

接下来行，每行一个结点的子结点信息，格式如下：

```text
k child_1 child_2 ... child_k
```

其中k表示该结点的子结点个数，child1、child2、...、childk表示子结点的编号。

**输出**

输出个整数，分别表示编号从`0`到`n-1`的结点的层号，中间用空格隔开，行末不允许有多余的空格。

样例1

输入

```
7
3 2 4 5
0
2 1 6
0
0
1 3
0
```

输出

```
1 3 2 3 2 2 3
```

解释

对应的树如下图所示，层号为`1`的结点编号为`0`，层号为`2`的结点编号为`2`、`4`、`5`，层号为`3`的结点编号为`1`、`6`、`3`。

![树的先根遍历.png](https://raw.githubusercontent.com/GMyhf/img/main/img/6259b6fa-bd37-4ade-9e7f-6eefc1e1af74.png)



```python
n = int(input().strip())
tree = [[] for _ in range(n)]
levels = [0 for _ in range(n)]

for i in range(n):
    line = list(map(int, input().strip().split()))
    k = line[0]
    for j in range(1, k + 1):
        tree[i].append(line[j])

q = [(0, 1)]
while q:
    node, level = q.pop(0)
    levels[node] = level
    for child in tree[node]:
        q.append((child, level + 1))

print(' '.join(map(str, levels)))
```





```python
class Tree:
    def __init__(self, n):
        self.n = n
        self.tree = [[] for _ in range(n)]
        self.levels = [0 for _ in range(n)]

    def add_node(self, node, children):
        self.tree[node] = children

    def bfs(self):
        q = [(0, 1)]
        while q:
            node, level = q.pop(0)
            self.levels[node] = level
            for child in self.tree[node]:
                q.append((child, level + 1))

    def print_levels(self):
        print(' '.join(map(str, self.levels)))


n = int(input().strip())
tree = Tree(n)

for i in range(n):
    line = list(map(int, input().strip().split()))
    k = line[0]
    children = line[1:k+1]
    tree.add_node(i, children)

tree.bfs()
tree.print_levels()
```



### 3.6 树的路径和

https://sunnywhy.com/sfbj/9/3/350

现有一棵个结点的树（结点编号为从`0`到`n-1`，根结点为`0`号结点），每个结点有各自的权值。

1. 结点的路径和是指，从根结点到该结点的路径上所有结点的权值之和；
2. 树的路径和是指，树所有叶结点的路径和之和。

求这棵树的路径和。

**输入**

第一行一个整数 $n (1 \le n \le 50)$，表示树的结点个数；

第二行个整数，分别给出编号从`0`到`n-1`的个结点的权值$w (1 \le w \le 100)$，；

接下来行，每行一个结点的子结点信息，格式如下：

```text
k child_1 child_2 ... child_k
```

其中k表示该结点的子结点个数，child1、child2、...、childk表示子结点的编号。

**输出**

输出一个整数，表示树的路径和。

样例1

输入

```
7
3 5 1 1 2 4 2
3 2 4 5
0
2 1 6
0
0
1 3
0
```

输出

```
28
```

解释

对应的树如下图所示，其中黑色数字为结点编号，编号右下角的灰色数字为结点权值。由此可得叶结点`1`的路径和为，叶结点`6`的路径和为，叶结点`4`的路径和为，叶结点`3`的路径和为，因此二叉树的路径和为。

![树的路径和.png](https://raw.githubusercontent.com/GMyhf/img/main/img/202403202120669.png)





```python
n = int(input().strip())
weights = list(map(int, input().strip().split()))
tree = [[] for _ in range(n)]

for i in range(n):
    line = list(map(int, input().strip().split()))
    k = line[0]
    for j in range(1, k + 1):
        tree[i].append(line[j])

def dfs(node, path_sum):
    path_sum += weights[node]
    if not tree[node]:  # if the node is a leaf node
        return path_sum
    return sum(dfs(child, path_sum) for child in tree[node])

result = dfs(0, 0)
print(result)
```



### 3.7 树的带权路径长度

https://sunnywhy.com/sfbj/9/3/351

现有一棵个结点的树（结点编号为从`0`到`n-1`，根结点为`0`号结点），每个结点有各自的权值。

1. 结点的路径长度是指，从根结点到该结点的边数；
2. 结点的带权路径长度是指，结点权值乘以结点的路径长度；
3. 树的带权路径长度是指，树的所有叶结点的带权路径长度之和。

求这棵树的带权路径长度。

**输入**

第一行一个整数 $n (1 \le n \le 50)$，表示树的结点个数；

第二行个整数，分别给出编号从`0`到`n-1`的个结点的权值（）；

接下来行，每行一个结点的子结点信息，格式如下：

```text
k child_1 child_2 ... child_k
```

其中k表示该结点的子结点个数，child1、child2、...、childk表示子结点的编号。

**输出**

输出一个整数，表示树的带权路径长度。

样例1

输入

```
7
3 5 1 1 2 4 2
3 2 4 5
0
2 1 6
0
0
1 3
0
```

输出

```
18
```

解释

对应的树如下图所示，其中黑色数字为结点编号，编号右下角的格式为`结点权值*结点路径长度=结点带权路径长度`。由此可得树的带权路径长度为。

![树的带权路径长度.png](https://cdn.sunnywhy.com/202203/2d112f11-2165-4105-ab75-3c0782aa4572.png)



```python
class TreeNode:
    def __init__(self, weight=0):
        self.weight = weight
        self.children = []

def build_tree(weights, edges):
    nodes = [TreeNode(weight=w) for w in weights]
    for i, children in enumerate(edges):
        for child in children:
            nodes[i].children.append(nodes[child])
    return nodes[0]  # 返回根节点

def dfs(node, depth):
    # 如果当前节点是叶子节点，则返回其带权路径长度
    if not node.children:
        return node.weight * depth
    # 否则，递归遍历其子节点
    total_weight_path_length = 0
    for child in node.children:
        total_weight_path_length += dfs(child, depth + 1)
    return total_weight_path_length

def weighted_path_length(n, weights, edges):
    # 构建树
    root = build_tree(weights, edges)
    # 从根节点开始深度优先搜索
    return dfs(root, 0)

# 输入处理
n = int(input().strip())
weights = list(map(int, input().strip().split()))
edges = []
for _ in range(n):
    line = list(map(int, input().strip().split()))
    if line[0] != 0:  # 忽略没有子节点的情况
        edges.append(line[1:])
    else:
        edges.append([])

# 计算带权路径长度
print(weighted_path_length(n, weights, edges))

```









## 4 二叉查找树（BST）5题

### 4.1 二叉查找树的建立

https://sunnywhy.com/sfbj/9/4

将n个互不相同的正整数先后插入到一棵空的二叉查找树中，求最后生成的二叉查找树的先序序列。

**输入**

第一行一个整数 $n (1 \le n \le 50)$，表示二叉查找树的结点个数；

第二行 n 个整数 $a_i (1 \le a_i \le 100)$，表示插入序列。

**输出**

输出个整数，表示先序遍历序列，中间用空格隔开，行末不允许有多余的空格。

样例1

输入

```
6
5 2 3 6 1 8
```

输出

```
5 2 1 3 6 8
```

解释

插入的过程如下图所示。

<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/202403202151255.png" alt="二叉查找树的建立.png" style="zoom:67%;" />







```python
class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

class BST:
    def __init__(self):
        self.root = None

    def insert(self, value):
        if self.root is None:
            self.root = Node(value)
        else:
            self._insert(value, self.root)

    def _insert(self, value, node):
        if value < node.value:
            if node.left is None:
                node.left = Node(value)
            else:
                self._insert(value, node.left)
        else:
            if node.right is None:
                node.right = Node(value)
            else:
                self._insert(value, node.right)

    def preorder(self):
        return self._preorder(self.root)

    def _preorder(self, node):
        if node is None:
            return []
        return [node.value] + self._preorder(node.left) + self._preorder(node.right)

n = int(input().strip())
values = list(map(int, input().strip().split()))
bst = BST()
for value in values:
    bst.insert(value)
print(' '.join(map(str, bst.preorder())))
```



### 4.2 二叉查找树的判定

https://sunnywhy.com/sfbj/9/4/353

现有一棵二叉树的中序遍历序列，问这棵二叉树是否是二叉查找树。

二叉查找树的定义：在二叉树定义的基础上，满足左子结点的数据域小于或等于根结点的数据域，右子结点的数据域大于根结点的数据域。

**输入**

第一行一个整数 $n (1 \le n \le 50)$，表示二叉查找树的结点个数；

第二行 n 个整数 $a_i (1 \le a_i \le 100)$，表示中序遍历序列。数据保证序列元素互不相同。

**输出**

如果是二叉查找树，那么输出`Yes`，否则输出`No`。

样例1

输入

```
3
1 2 3
```

输出

```
Yes
```

解释

对应的二叉树如下所示，是二叉查找树。

![二叉查找树的判定.png](https://raw.githubusercontent.com/GMyhf/img/main/img/202403202221291.png)

样例2

输入

```
3
2 1 3
```

输出

```
No
```

解释

对应的二叉树如下所示，不是二叉查找树。

![二叉查找树的判定_2.png](https://raw.githubusercontent.com/GMyhf/img/main/img/202403202222819.png)





```python
n = int(input().strip())
sequence = list(map(int, input().strip().split()))

if sequence == sorted(sequence):
    print("Yes")
else:
    print("No")
```



### 4.3 还原二叉查找树

https://sunnywhy.com/sfbj/9/4/354

现有一棵二叉查找树的先序遍历序列，还原这棵二叉查找树，并输出它的后序序列。

**输入**

第一行一个整数 $n (1 \le n \le 50)$，表示二叉查找树的结点个数；

第二行 n 个整数 $a_i (1 \le a_i \le 100)$，表示先序遍历序列。数据保证序列元素互不相同。

**输出**

输出个整数，表示后序遍历序列，中间用空格隔开，行末不允许有多余的空格。

样例1

输入

```
6
5 2 1 3 6 8
```

输出

```
1 3 2 8 6 5
```

解释

对应的二叉查找树如下所示，后序序列为`1 3 2 8 6 5`。

![还原二叉查找树.png](https://raw.githubusercontent.com/GMyhf/img/main/img/202403202231311.png)





```python
class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

class BST:
    def __init__(self, preorder):
        if not preorder:
            self.root = None
        else:
            self.root = self.build(preorder)

    def build(self, preorder):
        if not preorder:
            return None
        root = Node(preorder[0])
        i = 1
        while i < len(preorder) and preorder[i] < root.value:
            i += 1
        root.left = self.build(preorder[1:i])
        root.right = self.build(preorder[i:])
        return root

    def postorder(self):
        return self._postorder(self.root)

    def _postorder(self, node):
        if node is None:
            return []
        return self._postorder(node.left) + self._postorder(node.right) + [node.value]

n = int(input().strip())
preorder = list(map(int, input().strip().split()))
bst = BST(preorder)
print(' '.join(map(str, bst.postorder())))
```



### 4.4 相同的二叉查找树

https://sunnywhy.com/sfbj/9/4/355

将第一组个互不相同的正整数先后插入到一棵空的二叉查找树中，得到二叉查找树；再将第二组个互不相同的正整数先后插入到一棵空的二叉查找树中，得到二叉查找树。判断和是否是同一棵二叉查找树。

**输入**

第一行一个整数 $n (1 \le n \le 50)$，表示二叉查找树的结点个数；

第二行个 n 个整数 $a_i (1 \le a_i \le 100)$，表示第一组插入序列；

第三行个 n 个整数 $b_i (1 \le b_i \le 100)$，表示第二组插入序列。

**输出**

如果是同一棵二叉查找树，那么输出`Yes`，否则输出`No`。

样例1

输入

```
6
5 2 3 6 1 8
5 6 8 2 1 3
```

输出

```
Yes
```

解释

两种插入方式均可以得到下面这棵二叉查找树。

![还原二叉查找树.png](https://raw.githubusercontent.com/GMyhf/img/main/img/202403202231311.png)

样例2

输入

```
6
5 2 3 6 1 8
5 6 8 3 1 2
```

输出

```
No
```

解释

两种插入方式分别得到下图的两种二叉查找树。

![相同的二叉查找树_2.png](https://raw.githubusercontent.com/GMyhf/img/main/img/202403202341265.png)



先定义了`TreeNode`类用于表示二叉树的节点，然后定义了`insert_into_bst`函数用于将一个新值插入到二叉查找树中。`build_bst_from_sequence`函数接收一个序列，依次调用`insert_into_bst`来构建出一棵二叉查找树。`is_same_tree`函数用于比较两棵二叉树是否结构相同（即形状相同且对应位置的节点值相等）。

```python
class TreeNode:
    def __init__(self, val=0):
        self.val = val
        self.left = None
        self.right = None

def insert_into_bst(root, val):
    if not root:
        return TreeNode(val)
    if val < root.val:
        root.left = insert_into_bst(root.left, val)
    else:
        root.right = insert_into_bst(root.right, val)
    return root

def build_bst_from_sequence(sequence):
    root = None
    for val in sequence:
        root = insert_into_bst(root, val)
    return root

def is_same_tree(p, q):
    if not p and not q:
        return True
    if not p or not q:
        return False
    if p.val != q.val:
        return False
    return is_same_tree(p.left, q.left) and is_same_tree(p.right, q.right)

# 输入处理
n = int(input().strip())
seq1 = list(map(int, input().strip().split()))
seq2 = list(map(int, input().strip().split()))

# 构建二叉查找树
tree1 = build_bst_from_sequence(seq1)
tree2 = build_bst_from_sequence(seq2)

# 判断是否为同一棵树
if is_same_tree(tree1, tree2):
    print("Yes")
else:
    print("No")

```



### 4.5 填充二叉查找树

https://sunnywhy.com/sfbj/9/4/356

现有一棵个结点的二叉树（结点编号为从`0`到`n-1`，根结点为`0`号结点），将个互不相同的正整数填入这棵二叉树结点的数据域中，使其成为二叉查找树。求填充后二叉查找树的先序序列。

**输入**

第一行一个整数 $n (1 \le n \le 50)$，表示二叉树的结点个数；

第二行 n 个整数，表示需要填入二叉树中的数 $val_i$, 其中填入数字的范围为 $1 \le val_i \le 100$。

接下来 n 行，每行一个结点，按顺序给出编号为从`0`到`n-1`的个结点的左子结点编号和右子结点编号，中间用空格隔开。如果不存在对应的子结点，那么用`-1`表示。

**输出**

输出 n 个整数，表示先序遍历序列，中间用空格隔开，行末不允许有多余的空格。

样例1

输入

```
6
1 2 3 5 6 8
2 5
-1 -1
1 4
-1 -1
-1 -1
-1 3
```

输出

```
5 2 1 3 6 8
```

解释

下左图为输入的二叉树，填入`6`个整数后变为下右图的二叉查找树。

![填充二叉查找树.png](https://raw.githubusercontent.com/GMyhf/img/main/img/202403202353104.png)





To solve this problem, you can follow these steps:

1. Read the input values and the structure of the binary tree.
2. Sort the input values in ascending order.
3. Perform an inorder traversal of the binary tree and fill the nodes with the sorted values. This will make the binary tree a binary search tree (BST) because the inorder traversal of a BST is a sorted sequence.
4. Perform a preorder traversal of the BST and print the result.

Here is the Python code that implements this plan:

```python
class Node:
    def __init__(self, value=None):
        self.value = value
        self.left = None
        self.right = None

class BST:
    def __init__(self, n):
        self.nodes = [Node() for _ in range(n)]
        self.index = 0

    def insert_values(self, values):
        self.values = iter(sorted(values))
        self._insert_values(self.nodes[0])

    def _insert_values(self, node):
        if node is not None:
            self._insert_values(node.left)
            node.value = next(self.values)
            self._insert_values(node.right)

    def preorder(self):
        return self._preorder(self.nodes[0])

    def _preorder(self, node):
        if node is None:
            return []
        return [node.value] + self._preorder(node.left) + self._preorder(node.right)

n = int(input().strip())
values = list(map(int, input().strip().split()))
bst = BST(n)
for i in range(n):
    left, right = map(int, input().strip().split())
    if left != -1:
        bst.nodes[i].left = bst.nodes[left]
    if right != -1:
        bst.nodes[i].right = bst.nodes[right]
bst.insert_values(values)
print(' '.join(map(str, bst.preorder())))
```

This code reads the input values and the structure of the binary tree from the input, sorts the input values, fills the nodes of the binary tree with the sorted values, performs a preorder traversal of the BST, and then prints the result.





## 5 平衡二叉树（AVL树）3题

### 5.1 二叉查找树的平衡因子

https://sunnywhy.com/sfbj/9/5

对一棵二叉查找树来说，我们把结点的左子树和右子树的高度之差称为结点的平衡因子。

现将 n 个互不相同的正整数先后插入到一棵空的二叉查找树中，然后按中序遍历的顺序，给出二叉查找树中各结点的平衡因子。

**输入**

第一行一个整数 $n (1 \le n \le 50)$，表示二叉查找树的结点个数；

第二行个整数 $a_i (1 \le a_i \le 100)$，表示插入序列。

**输出**

输出 n 个整数，表示 n 个结点的平衡因子，中间用空格隔开，行末不允许有多余的空格。

样例1

输入

```
7
5 2 3 6 1 8 7
```

输出

```
0 0 0 -1 -2 0 1
```

解释

生成的二叉查找树和每个结点的平衡因子计算过程如图所示。

![二叉查找树的平衡因子.png](https://raw.githubusercontent.com/GMyhf/img/main/img/202403210006174.png)



To solve this problem, you can follow these steps:

1. Read the input sequence.
2. Insert the values into a binary search tree (BST).
3. Perform an inorder traversal of the BST and calculate the balance factor for each node. The balance factor of a node is the height of its left subtree minus the height of its right subtree.
4. Print the balance factors in the order they were visited during the inorder traversal.

Here is the Python code that implements this plan:

```python
class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        self.height = 1

class BST:
    def __init__(self):
        self.root = None

    def insert(self, value):
        if self.root is None:
            self.root = Node(value)
        else:
            self._insert(value, self.root)

    def _insert(self, value, node):
        if value < node.value:
            if node.left is None:
                node.left = Node(value)
            else:
                self._insert(value, node.left)
        else:
            if node.right is None:
                node.right = Node(value)
            else:
                self._insert(value, node.right)
        node.height = 1 + max(self._get_height(node.left), self._get_height(node.right))

    def _get_height(self, node):
        if node is None:
            return 0
        return node.height

    def balance_factors(self):
        return self._balance_factors(self.root)

    def _balance_factors(self, node):
        if node is None:
            return []
        balance_factor = self._get_height(node.left) - self._get_height(node.right)
        return self._balance_factors(node.left) + [balance_factor] + self._balance_factors(node.right)

n = int(input().strip())
sequence = list(map(int, input().strip().split()))

bst = BST()
for value in sequence:
    bst.insert(value)

print(' '.join(map(str, bst.balance_factors())))
```

This code reads the sequence from the input, inserts its values into a BST, calculates the balance factors of the nodes during an inorder traversal of the BST, and then prints the results.



### 5.2 平衡二叉树的判定

https://sunnywhy.com/sfbj/9/5/358

将 n 个互不相同的正整数先后插入到一棵空的二叉查找树中，判断最后生成的二叉查找树是否是平衡二叉树（AVL树）。

**输入**

第一行一个整数 $n (1 \le n \le 50)$，表示二叉查找树的结点个数；

第二行个 n 整数 $a_i (1 \le a_i \le 100)$，表示插入序列。数据保证序列元素互不相同。

**输出**

如果是平衡二叉树，那么输出`Yes`，否则输出`No`。

样例1

输入

```
5
5 2 3 6 1
```

输出

```
Yes
```

解释

对应的二叉查找树如下所示，是平衡二叉树。

![平衡二叉树的判定_样例1.png](https://raw.githubusercontent.com/GMyhf/img/main/img/202403210034791.png)

样例2

输入

```
4
5 2 3 1
```

输出

```
No
```

解释

对应的二叉查找树如下所示，不是平衡二叉树。

![平衡二叉树的判定_样例2.png](https://raw.githubusercontent.com/GMyhf/img/main/img/202403210034500.png)



To solve this problem, you can follow these steps:

1. Read the input sequence.
2. Insert the values into a binary search tree (BST).
3. Check if the BST is balanced. A BST is balanced if the absolute difference between the heights of the left and right subtrees of every node is at most 1.
4. Print "Yes" if the BST is balanced, otherwise print "No".

Here is the Python code that implements this plan:

```python
class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        self.height = 1

class BST:
    def __init__(self):
        self.root = None

    def insert(self, value):
        if self.root is None:
            self.root = Node(value)
        else:
            self._insert(value, self.root)

    def _insert(self, value, node):
        if value < node.value:
            if node.left is None:
                node.left = Node(value)
            else:
                self._insert(value, node.left)
        else:
            if node.right is None:
                node.right = Node(value)
            else:
                self._insert(value, node.right)
        node.height = 1 + max(self._get_height(node.left), self._get_height(node.right))

    def _get_height(self, node):
        if node is None:
            return 0
        return node.height

    def is_balanced(self):
        return self._is_balanced(self.root)

    def _is_balanced(self, node):
        if node is None:
            return True
        balance_factor = abs(self._get_height(node.left) - self._get_height(node.right))
        return balance_factor <= 1 and self._is_balanced(node.left) and self._is_balanced(node.right)

n = int(input().strip())
sequence = list(map(int, input().strip().split()))

bst = BST()
for value in sequence:
    bst.insert(value)

if bst.is_balanced():
    print("Yes")
else:
    print("No")
```

This code reads the sequence from the input, inserts its values into a BST, checks if the BST is balanced, and then prints the result.





### 5.3 平衡二叉树的建立

https://sunnywhy.com/sfbj/9/5/359

将 n 个互不相同的正整数先后插入到一棵空的AVL树中，求最后生成的AVL树的先序序列。

**输入**

第一行一个整数 $n (1 \le n \le 50)$，表示AVL树的结点个数；

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

<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/202403210041932.png" alt="平衡二叉树的建立.png" style="zoom:67%;" />



To solve this problem, you can follow these steps:

1. Read the input sequence.
2. Insert the values into an AVL tree. An AVL tree is a self-balancing binary search tree, and the heights of the two child subtrees of any node differ by at most one.
3. Perform a preorder traversal of the AVL tree and print the result.

Here is the Python code that implements this plan:

```python
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







## 6 并查集 5题

### 6.1 学校的班级个数（1）

https://sunnywhy.com/sfbj/9/6/360

现有一个学校，学校中有若干个班级，每个班级中有若干个学生，每个学生只会存在于一个班级中。如果学生`A`和学生`B`处于一个班级，学生`B`和学生`C`处于一个班级，那么我们称学生`A`和学生`C`也处于一个班级。

现已知学校中共 n 个学生（编号为从`1`到`n`），并给出 m 组学生关系（指定两个学生处于一个班级），问总共有多少个班级。

**输入**

第一行两个整数 $m、n (1 \le n \le 100, 1 \le m \le 100)$，分别表示学生个数、学生关系个数；

接下来 m 行，每行两个整数 a 和 b $ (1 \le a \le n, 1 \le b \le n, a \ne b)$，表示编号为`a`的学生和编号为`b`的学生处于一个班级。

**输出**

输出一个整数，表示班级个数。

样例1

输入

```
5 3
4 2
1 3
2 5
```

输出

```
2
```

解释

编号`2`、`4`、`5`的学生在同一个班级，编号`1`、`3`的学生在同一个班级，因此共有两个班级。



To solve this problem, you can use the concept of disjoint sets (also known as union-find). Each student is initially in their own set (or class). When a relationship between two students is given, you unite their sets. The number of disjoint sets at the end will be the number of classes.

Here is a step-by-step plan:

1. Initialize an array `parent` of size `n+1` where `parent[i]` is initially `i`. This means each student is in their own class.
2. For each student relationship, unite their sets using the union operation.
3. Count the number of unique parents. This will be the number of classes.

Here is the Python code that implements this plan:

```python
def find(x):
    if parent[x] != x: # 如果不是根结点，继续循环
        parent[x] = find(parent[x])
    return parent[x]

def union(x, y):
    parent[find(x)] = find(y)

n, m = map(int, input().split())
parent = list(range(n + 1))	# parent[i] == i，则说明元素i是该集合的根结点

for _ in range(m):
    a, b = map(int, input().split())
    union(a, b)

classes = set(find(x) for x in range(1, n + 1))
print(len(classes))
```

This code reads the number of students and relationships from the input, initializes each student to be in their own class, unites the classes of each pair of students that have a relationship, and then counts and prints the number of unique classes.





### 6.2 学校的班级人数（2）

https://sunnywhy.com/sfbj/9/6/361

现有一个学校，学校中有若干个班级，每个班级中有若干个学生，每个学生只会存在于一个班级中。如果学生`A`和学生`B`处于一个班级，学生`B`和学生`C`处于一个班级，那么我们称学生`A`和学生`C`也处于一个班级。

现已知学校中共 n 个学生（编号为从`1`到`n`），并给出 m 组学生关系（指定两个学生处于一个班级），问总共有多少个班级，并按降序给出每个班级的人数。

**输入**

第一行两个整数 $m、n (1 \le n \le 100, 1 \le m \le 100)$，分别表示学生个数、学生关系个数；

接下来 m 行，每行两个整数 a 和 b $ (1 \le a \le n, 1 \le b \le n, a \ne b)$，表示编号为`a`的学生和编号为`b`的学生处于一个班级。

**输出**

第一行输出一个整数，表示班级个数；

第二行若干个整数，按降序给出每个班级的人数。整数之间用空格隔开，行末不允许有多余的空格。

样例1

输入

```
5 3
4 2
1 3
2 5
```

输出

```
2
3 2
```

解释

编号`2`、`4`、`5`的学生在同一个班级，编号`1`、`3`的学生在同一个班级，因此共有两个班级，人数分别是`3`和`2`。





To solve this problem, you can use the concept of disjoint sets (also known as union-find). Each student is initially in their own set (or class). When a relationship between two students is given, you unite their sets. The number of disjoint sets at the end will be the number of classes. You can also maintain a count of the number of students in each class.

Here is a step-by-step plan:

1. Initialize an array `parent` of size `n+1` where `parent[i]` is initially `i`. This means each student is in their own class.
2. Initialize an array `size` of size `n+1` where `size[i]` is initially `1`. This means each class initially has one student.
3. For each student relationship, unite their sets using the union operation and update the size of the new set.
4. Count the number of unique parents. This will be the number of classes.
5. Print the sizes of the classes in descending order.

Here is the Python code that implements this plan:

```python
def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(x, y):
    root_x = find(x)
    root_y = find(y)
    if root_x != root_y:
        parent[root_x] = root_y
        size[root_y] += size[root_x]

n, m = map(int, input().split())
parent = list(range(n + 1))
size = [1] * (n + 1)

for _ in range(m):
    a, b = map(int, input().split())
    union(a, b)

#classes = [size[find(x)] for x in range(1, n + 1) if x == parent[x]]
classes = [size[x] for x in range(1, n + 1) if x == parent[x]]
print(len(classes))
print(' '.join(map(str, sorted(classes, reverse=True))))
```

This code reads the number of students and relationships from the input, initializes each student to be in their own class, unites the classes of each pair of students that have a relationship, counts and prints the number of unique classes, and prints the sizes of the classes in descending order.





### 6.3 是否相同班级

https://sunnywhy.com/sfbj/9/6/362

现有一个学校，学校中有若干个班级，每个班级中有若干个学生，每个学生只会存在于一个班级中。如果学生`A`和学生`B`处于一个班级，学生`B`和学生`C`处于一个班级，那么我们称学生`A`和学生`C`也处于一个班级。

现已知学校中共 n 个学生（编号为从`1`到`n`），并给出 m 组学生关系（指定两个学生处于一个班级）。然后给出 k 个查询，每个查询询问两个学生是否在同一个班级。

**输入**

第一行两个整数 $n、m (1 \le n \le 10^5, 1 \le m \le 10^5)$，分别表示学生个数、学生关系个数；

接下来 m 行，每行两个整数 a 和 b $ (1 \le a \le n, 1 \le b \le n, a \ne b)$，表示编号为`a`的学生和编号为`b`的学生处于一个班级。

然后一个整数 $k (1 \le k \le 10^3)$，表示查询个数；

接下来 k 行，每行两个整数 a 和 b $ (1 \le a \le n, 1 \le b \le n)$，表示询问编号为`a`的学生和编号为`b`的学生是否在同一个班级。

**输出**

每个查询输出一行，如果在同一个班级，那么输出`Yes`，否则输出`No`。

样例1

输入

```
5 3
4 2
1 3
2 5
2
4 5
1 2
```

输出

```
Yes
No
```

解释

编号`2`、`4`、`5`的学生在同一个班级，编号`1`、`3`的学生在同一个班级，因此编号`4`和`5`的学生在同一个班级，编号`1`和`2`的学生不在同一个班级。





To solve this problem, you can use the concept of disjoint sets (also known as union-find). Each student is initially in their own set (or class). When a relationship between two students is given, you unite their sets. For each query, you check if the two students are in the same set.

Here is a step-by-step plan:

1. Initialize an array `parent` of size `n+1` where `parent[i]` is initially `i`. This means each student is in their own class.
2. For each student relationship, unite their sets using the union operation.
3. For each query, check if the two students are in the same set using the find operation.

Here is the Python code that implements this plan:

```python
def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(x, y):
    parent[find(x)] = find(y)

n, m = map(int, input().split())
parent = list(range(n + 1))

for _ in range(m):
    a, b = map(int, input().split())
    union(a, b)

k = int(input())
for _ in range(k):
    a, b = map(int, input().split())
    if find(a) == find(b):
        print('Yes')
    else:
        print('No')
```

This code reads the number of students and relationships from the input, initializes each student to be in their own class, unites the classes of each pair of students that have a relationship, and then for each query, checks and prints whether the two students are in the same class.



### 6.4 迷宫连通性

https://sunnywhy.com/sfbj/9/6/363

现有一个迷宫，迷宫中有 n 个房间（编号为从`1`到`n`），房间与房间之间可能连通。如果房间`A`和房间`B`连通，房间`B`和房间`C`连通，那么我们称房间`A`和房间`C`也连通。给定 m 组连通关系（指定两个房间连通），问迷宫中的所有房间是否连通。

**输入**

第一行两个整数$n、m (1 \le n \le 100, 1 \le m \le 100)$，分别表示房间个数、连通关系个数；

接下来行，每行两个整数 a 和 b $ (1 \le a \le n, 1 \le b \le n)$，表示编号为`a`的房间和编号为`b`的房间是连通的。

**输出**

如果所有房间连通，那么输出`Yes`，否则输出`No`。

样例1

输入

```
5 4
4 2
1 3
2 5
1 5
```

输出

```
Yes
```

解释

所有房间都连通，因此输出`Yes`。

样例2

输入

```
5 3
4 2
1 3
2 5
```

输出

```
No
```

解释

编号`2`、`4`、`5`的房间互相连通，编号`1`、`3`的房间互相连通，因此没有全部互相连通，输出`No`。



To solve this problem, you can use the concept of disjoint sets (also known as union-find). Each room is initially in its own set. When a connection between two rooms is given, you unite their sets. If at the end there is only one set, then all rooms are connected.

Here is a step-by-step plan:

1. Initialize an array `parent` of size `n+1` where `parent[i]` is initially `i`. This means each room is in its own set.
2. For each connection, unite their sets using the union operation.
3. Check if all rooms are in the same set.

Here is the Python code that implements this plan:

```python
def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(x, y):
    parent[find(x)] = find(y)

n, m = map(int, input().split())
parent = list(range(n + 1))

for _ in range(m):
    a, b = map(int, input().split())
    union(a, b)

sets = set(find(x) for x in range(1, n + 1))
if len(sets) == 1:
    print('Yes')
else:
    print('No')
```

This code reads the number of rooms and connections from the input, initializes each room to be in its own set, unites the sets of each pair of rooms that have a connection, and then checks and prints whether all rooms are in the same set.





### 6.5 班级最高分

https://sunnywhy.com/sfbj/9/6/364

现有一个学校，学校中有若干个班级，每个班级中有若干个学生，每个学生只会存在于一个班级中。如果学生`A`和学生`B`处于一个班级，学生`B`和学生`C`处于一个班级，那么我们称学生`A`和学生`C`也处于一个班级。

现已知学校中共 n 个学生（编号为从`1`到`n`），每个学生有一个考试分数，再给出 m 组学生关系（指定两个学生处于一个班级），问总共有多少个班级，并按降序给出每个班级的最高考试分数。

**输入**

第一行两个整数 $n、m (1 \le n \le 100, 1 \le m \le 100)$，分别表示学生个数、学生关系个数；

第二行为用空格隔开的 n 个整数（$0 \le 每个整数 \le 100$），表示个学生的考试分数；

接下来 m 行，每行两个整数 a 和 b $ (1 \le a \le n, 1 \le b \le n)$，表示编号为`a`的学生和编号为`b`的学生处于一个班级。

**输出**

第一行输出一个整数，表示班级个数；

第二行若干个整数，按降序给出每个班级的最高考试分数。整数之间用空格隔开，行末不允许有多余的空格。

样例1

输入



```
5 3
88 90 86 92 95
4 2
1 3
2 5
```

输出

```
2
95 88
```

解释

编号`2`、`4`、`5`的学生在同一个班级，编号`1`、`3`的学生在同一个班级，因此共有两个班级，最高分数分别是编号`1`的学生的`88`分、编号`5`的学生的`95`分。



To solve this problem, you can use the concept of disjoint sets (also known as union-find). Each student is initially in their own set (or class). When a relationship between two students is given, you unite their sets. The number of disjoint sets at the end will be the number of classes. You can also maintain a list of scores for each class and find the maximum score for each class.

Here is a step-by-step plan:

1. Initialize an array `parent` of size `n+1` where `parent[i]` is initially `i`. This means each student is in their own class.
2. Initialize an array `scores` of size `n+1` to store the scores of each student.
3. For each student relationship, unite their sets using the union operation.
4. For each class, find the maximum score.
5. Count the number of unique parents. This will be the number of classes.
6. Print the maximum scores of the classes in descending order.

Here is the Python code that implements this plan:

```python
def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(x, y):
    root_x = find(x)
    root_y = find(y)
    if root_x != root_y:
        parent[root_x] = root_y
        scores[root_y] = max(scores[root_y], scores[root_x])

n, m = map(int, input().split())
parent = list(range(n + 1))
scores = list(map(int, input().split()))
scores.insert(0, 0)  # to make the scores 1-indexed

for _ in range(m):
    a, b = map(int, input().split())
    union(a, b)

classes_scores = [scores[find(x)] for x in range(1, n + 1) if x == parent[x]]
print(len(classes_scores))
print(' '.join(map(str, sorted(classes_scores, reverse=True))))
```

This code reads the number of students and relationships from the input, initializes each student to be in their own class, unites the classes of each pair of students that have a relationship, finds the maximum score for each class, counts and prints the number of unique classes, and prints the maximum scores of the classes in descending order.





## 7 堆 6题

### 7.1 向下调整构建大顶堆

https://sunnywhy.com/sfbj/9/7

现有个不同的正整数，将它们按层序生成完全二叉树，然后使用**向下调整**的方式构建一个完整的大顶堆。最后按层序输出堆中的所有元素。

**输入**

第一行一个整数$n (1 \le n \le 10^3)$，表示正整数的个数；

第二行 n 个整数$a_i (1 \le a_i \le 10^4) $​，表示正整数序列。

**输出**

输出 n 个整数，表示堆的层序序列，中间用空格隔开，行末不允许有多余的空格。

样例1

输入

```
6
3 2 6 5 8 7
```

输出

```
8 5 7 3 2 6
```

解释

调整前的完全二叉树和调整后的堆如下图所示。

<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/202403210116556.png" alt="向下调整构建大顶堆.png" style="zoom:67%;" />



解法1:

```python
class BinHeap:
    def __init__(self):
        self.heapList = [0]
        self.currentSize = 0

    def percUp(self, i):
        while i // 2 > 0:
            if self.heapList[i] < self.heapList[i // 2]:
                tmp = self.heapList[i // 2]
                self.heapList[i // 2] = self.heapList[i]
                self.heapList[i] = tmp
            i = i // 2

    def insert(self, k):
        self.heapList.append(k)
        self.currentSize = self.currentSize + 1
        self.percUp(self.currentSize)

    def percDown(self, i):
        while (i * 2) <= self.currentSize:
            mc = self.minChild(i)
            if self.heapList[i] > self.heapList[mc]:
                tmp = self.heapList[i]
                self.heapList[i] = self.heapList[mc]
                self.heapList[mc] = tmp
            i = mc

    def minChild(self, i):
        if i * 2 + 1 > self.currentSize:
            return i * 2
        else:
            if self.heapList[i * 2] < self.heapList[i * 2 + 1]:
                return i * 2
            else:
                return i * 2 + 1

    def delMin(self):
        retval = self.heapList[1]
        self.heapList[1] = self.heapList[self.currentSize]
        self.currentSize = self.currentSize - 1
        self.heapList.pop()
        self.percDown(1)
        return retval

    def buildHeap(self, alist):
        i = len(alist) // 2
        self.currentSize = len(alist)
        self.heapList = [0] + alist[:]
        while (i > 0):
            #print(f'i = {i}, {self.heapList}')
            self.percDown(i)
            i = i - 1
        #print(f'i = {i}, {self.heapList}')


n = int(input().strip())
heap = list(map(int, input().strip().split())) # [9, 5, 6, 2, 3]
heap = [-x for x in heap]

bh = BinHeap()
bh.buildHeap(heap)
ans = [-x for x in bh.heapList[1:]]
print(*ans)
```



解法2:

To solve this problem, you can use the concept of a binary heap. A binary heap is a complete binary tree where each node is either greater than or equal to its children (in a max heap) or less than or equal to its children (in a min heap). In this case, you are asked to create a max heap.

Here is a step-by-step plan:

1. Initialize an array `heap` of size `n` to store the elements of the heap.
2. For each element in the input, insert it into the `heap` array.
3. For each non-leaf node in the `heap` array (starting from the last non-leaf node and moving to the root), perform a downward adjustment to ensure the max heap property is maintained.
4. Print the elements of the `heap` array in order.

Here is the Python code that implements this plan:

```python
def down_adjust(parent_index, length):
    temp = heap[parent_index]
    child_index = 2 * parent_index + 1
    while child_index < length:
        if child_index + 1 < length and heap[child_index + 1] > heap[child_index]:
            child_index += 1
        if temp > heap[child_index]:
            break
        heap[parent_index] = heap[child_index]
        parent_index = child_index
        child_index = 2 * child_index + 1
    heap[parent_index] = temp

n = int(input().strip())
heap = list(map(int, input().strip().split()))

for i in range((n - 2) // 2, -1, -1):
    down_adjust(i, n)

print(' '.join(map(str, heap)))
```

This code reads the number of elements and the elements themselves from the input, inserts each element into the `heap` array, performs a downward adjustment for each non-leaf node in the `heap` array to ensure the max heap property is maintained, and then prints the elements of the `heap` array in order.





### 7.2 向上调整构建大顶堆

https://sunnywhy.com/sfbj/9/7/366

现有 n 个不同的正整数，将它们按层序生成完全二叉树，然后使用**向上调整**的方式构建一个完整的大顶堆。最后按层序输出堆中的所有元素。

**输入**

第一行一个整数 $n (1 \le n \le 10^3)$，表示正整数的个数；

第二行 n 个整数$a_i (1 \le a_i \le 10^4) $​，表示正整数序列。

**输出**

输出 n 个整数，表示堆的层序序列，中间用空格隔开，行末不允许有多余的空格。

样例1

输入

```
6
3 2 6 5 8 7
```

输出

```
8 6 7 2 5 3
```

解释

调整前的完全二叉树和调整后的堆如下图所示。

<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/202403210120258.png" alt="向上调整构建大顶堆.png" style="zoom: 67%;" />





To solve this problem, you can use the concept of a binary heap. A binary heap is a complete binary tree where each node is either greater than or equal to its children (in a max heap) or less than or equal to its children (in a min heap). In this case, you are asked to create a max heap.

Here is a step-by-step plan:

1. Initialize an array `heap` of size `n` to store the elements of the heap.
2. For each element in the input, insert it into the `heap` array.
3. For each inserted node in the `heap` array (starting from the last inserted node and moving to the root), perform an upward adjustment to ensure the max heap property is maintained.
4. Print the elements of the `heap` array in order.

Here is the Python code that implements this plan:

```python
def up_adjust(child_index):
    temp = heap[child_index]
    parent_index = (child_index - 1) // 2
    while child_index > 0 and temp > heap[parent_index]:
        heap[child_index] = heap[parent_index]
        child_index = parent_index
        parent_index = (parent_index - 1) // 2
    heap[child_index] = temp

n = int(input().strip())
heap = list(map(int, input().strip().split()))

for i in range(1, n):
    up_adjust(i)

print(' '.join(map(str, heap)))
```

This code reads the number of elements and the elements themselves from the input, inserts each element into the `heap` array, performs an upward adjustment for each inserted node in the `heap` array to ensure the max heap property is maintained, and then prints the elements of the `heap` array in order.







### 7.3 删除堆顶元素

https://sunnywhy.com/sfbj/9/7/367

现有 n 个不同的正整数，将它们按层序生成完全二叉树，然后使用**向下调整**的方式构建一个完整的大顶堆。然后删除堆顶元素，并将层序最后一个元素置于堆顶，进行一次向下调整，以形成新的堆。最后按层序输出新堆中的所有元素。

**输入**

第一行一个整数 $n (1 \le n \le 10^3)$，表示正整数的个数；

第二行 n 个整数 $a_i (1 \le a_i \le 10^4) $​，表示正整数序列。

**输出**

输出 n - 1 个整数，表示堆的层序序列，中间用空格隔开，行末不允许有多余的空格。

样例1

输入

```
6
3 2 6 5 8 7
```

输出

```
7 5 6 3 2
```

解释

操作过程如下图所示。

<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/202403210124838.png" alt="删除堆顶元素.png" style="zoom:67%;" />



解法1:

```python
class BinHeap:
    def __init__(self):
        self.heapList = [0]
        self.currentSize = 0

    def percUp(self, i):
        while i // 2 > 0:
            if self.heapList[i] < self.heapList[i // 2]:
                tmp = self.heapList[i // 2]
                self.heapList[i // 2] = self.heapList[i]
                self.heapList[i] = tmp
            i = i // 2

    def insert(self, k):
        self.heapList.append(k)
        self.currentSize = self.currentSize + 1
        self.percUp(self.currentSize)

    def percDown(self, i):
        while (i * 2) <= self.currentSize:
            mc = self.minChild(i)
            if self.heapList[i] > self.heapList[mc]:
                tmp = self.heapList[i]
                self.heapList[i] = self.heapList[mc]
                self.heapList[mc] = tmp
            i = mc

    def minChild(self, i):
        if i * 2 + 1 > self.currentSize:
            return i * 2
        else:
            if self.heapList[i * 2] < self.heapList[i * 2 + 1]:
                return i * 2
            else:
                return i * 2 + 1

    def delMin(self):
        retval = self.heapList[1]
        self.heapList[1] = self.heapList[self.currentSize]
        self.currentSize = self.currentSize - 1
        self.heapList.pop()
        self.percDown(1)
        return retval

    def buildHeap(self, alist):
        i = len(alist) // 2
        self.currentSize = len(alist)
        self.heapList = [0] + alist[:]
        while (i > 0):
            #print(f'i = {i}, {self.heapList}')
            self.percDown(i)
            i = i - 1
        #print(f'i = {i}, {self.heapList}')


n = int(input().strip())
heap = list(map(int, input().strip().split())) # [9, 5, 6, 2, 3]
heap = [-x for x in heap]

bh = BinHeap()
bh.buildHeap(heap)

bh.delMin()

ans = [-x for x in bh.heapList[1:]]
print(*ans)
```



解法2:

To solve this problem, you can use the concept of a binary heap. A binary heap is a complete binary tree where each node is either greater than or equal to its children (in a max heap) or less than or equal to its children (in a min heap). In this case, you are asked to create a max heap, delete the root, and then adjust the heap.

Here is a step-by-step plan:

1. Initialize an array `heap` of size `n` to store the elements of the heap.
2. For each element in the input, insert it into the `heap` array.
3. For each non-leaf node in the `heap` array (starting from the last non-leaf node and moving to the root), perform a downward adjustment to ensure the max heap property is maintained.
4. Delete the root of the heap and replace it with the last element in the heap.
5. Perform a downward adjustment on the new root to maintain the max heap property.
6. Print the elements of the `heap` array in order.

Here is the Python code that implements this plan:

```python
def down_adjust(parent_index, length):
    temp = heap[parent_index]
    child_index = 2 * parent_index + 1
    while child_index < length:
        if child_index + 1 < length and heap[child_index + 1] > heap[child_index]:
            child_index += 1
        if temp > heap[child_index]:
            break
        heap[parent_index] = heap[child_index]
        parent_index = child_index
        child_index = 2 * child_index + 1
    heap[parent_index] = temp

n = int(input().strip())
heap = list(map(int, input().strip().split()))

for i in range((n - 2) // 2, -1, -1):
    down_adjust(i, n)

heap[0] = heap[n - 1]
down_adjust(0, n - 1)

print(' '.join(map(str, heap[:-1])))
```

This code reads the number of elements and the elements themselves from the input, inserts each element into the `heap` array, performs a downward adjustment for each non-leaf node in the `heap` array to ensure the max heap property is maintained, deletes the root of the heap and replaces it with the last element in the heap, performs a downward adjustment on the new root to maintain the max heap property, and then prints the elements of the `heap` array in order.





### 7.4 堆排序

https://sunnywhy.com/sfbj/9/7/368

输入 n 个正整数，使用堆排序算法将它们按从小到大的顺序进行排序。

**输入**

第一行一个整数 $n (1 \le n \le 50)$，表示正整数的个数；

第二行为用空格隔开的 n 个正整数（每个正整数均不超过`100`）。

**输出**

输出一行，表示排序后的 n 个正整数。整数间用一个空格隔开，行末不允许有多余的空格。

样例1

输入

```
5
2 8 5 1 3
```

输出

```
1 2 3 5 8
```

解释

从小到大排序后可以得到序列`1 2 3 5 8`



解法1:

```python
class BinHeap:
    def __init__(self):
        self.heapList = [0]
        self.currentSize = 0

    def percUp(self, i):
        while i // 2 > 0:
            if self.heapList[i] < self.heapList[i // 2]:
                tmp = self.heapList[i // 2]
                self.heapList[i // 2] = self.heapList[i]
                self.heapList[i] = tmp
            i = i // 2

    def insert(self, k):
        self.heapList.append(k)
        self.currentSize = self.currentSize + 1
        self.percUp(self.currentSize)

    def percDown(self, i):
        while (i * 2) <= self.currentSize:
            mc = self.minChild(i)
            if self.heapList[i] > self.heapList[mc]:
                tmp = self.heapList[i]
                self.heapList[i] = self.heapList[mc]
                self.heapList[mc] = tmp
            i = mc

    def minChild(self, i):
        if i * 2 + 1 > self.currentSize:
            return i * 2
        else:
            if self.heapList[i * 2] < self.heapList[i * 2 + 1]:
                return i * 2
            else:
                return i * 2 + 1

    def delMin(self):
        retval = self.heapList[1]
        self.heapList[1] = self.heapList[self.currentSize]
        self.currentSize = self.currentSize - 1
        self.heapList.pop()
        self.percDown(1)
        return retval

    def buildHeap(self, alist):
        i = len(alist) // 2
        self.currentSize = len(alist)
        self.heapList = [0] + alist[:]
        while (i > 0):
            #print(f'i = {i}, {self.heapList}')
            self.percDown(i)
            i = i - 1
        #print(f'i = {i}, {self.heapList}')


n = int(input().strip())
heap = list(map(int, input().strip().split())) # [9, 5, 6, 2, 3]

bh = BinHeap()
bh.buildHeap(heap)

ans = []
while bh.currentSize > 0:
    ans.append(bh.delMin())
print(*ans)
```



解法2:

To solve this problem, you can use the heap sort algorithm. Heap sort is a comparison-based sorting algorithm that uses a binary heap data structure. It works by building a max heap from the input data, then iteratively removing the maximum element from the heap and inserting it at the end of the sorted section of the array.

Here is a step-by-step plan:

1. Initialize an array `heap` of size `n` to store the elements of the heap.
2. For each element in the input, insert it into the `heap` array.
3. For each non-leaf node in the `heap` array (starting from the last non-leaf node and moving to the root), perform a downward adjustment to ensure the max heap property is maintained.
4. Repeatedly swap the first element in the heap with the last element in the unsorted section of the array, then perform a downward adjustment on the new root to maintain the max heap property.
5. Print the elements of the `heap` array in order.

Here is the Python code that implements this plan:

```python
def down_adjust(parent_index, length):
    temp = heap[parent_index]
    child_index = 2 * parent_index + 1
    while child_index < length:
        if child_index + 1 < length and heap[child_index + 1] > heap[child_index]:
            child_index += 1
        if temp > heap[child_index]:
            break
        heap[parent_index] = heap[child_index]
        parent_index = child_index
        child_index = 2 * child_index + 1
    heap[parent_index] = temp

n = int(input().strip())
heap = list(map(int, input().strip().split()))

for i in range((n - 2) // 2, -1, -1):
    down_adjust(i, n)

for i in range(n - 1, 0, -1):
    heap[i], heap[0] = heap[0], heap[i]
    down_adjust(0, i)

print(' '.join(map(str, heap)))
```

This code reads the number of elements and the elements themselves from the input, inserts each element into the `heap` array, performs a downward adjustment for each non-leaf node in the `heap` array to ensure the max heap property is maintained, repeatedly swaps the first element in the heap with the last element in the unsorted section of the array and performs a downward adjustment on the new root, and then prints the elements of the `heap` array in order.





### 7.5 数据流第K大元素

https://sunnywhy.com/sfbj/9/7/369

现有一个初始为空的序列 S，对其执行 n 个操作，每个操作是以下两种操作之一：

1. 往序列 S 中加入一个正整数；
2. 输出当前序列 S 中第大的数。

其中，第大是指将序列从大到小排序后的第 k 个数。

**输入**

第一行两个整数$n、k (1 \le n \le 10^4, 2 \le k \le 100)$，分别表示操作个数、需要输出第几大的数；

接下来 n 行，每行一个操作信息：使用"`Push x`"来表示往序列中加入正整数 $x (1 \le x \le 10^9)$，使用"`Print`"来表示需要输出当前序列中第大的数。

**输出**

每次执行`Print`操作时，输出一行，表示当前序列中第大的数。如果不存在第大的数，那么输出`-1`。

样例1

输入

```
7 2
Push 1
Print
Push 3
Print
Push 7
Push 6
Print
```

输出

```
-1
1
6
```

解释

第一个`Print`时序列中元素为`1`，不存在第`2`大的元素，因此输出`-1`；

第二个`Print`时序列中元素为`1 3`，因此第`2`大的元素为`1`；

第三个`Print`时序列中元素为`1 3 7 6`，因此第`2`大的元素为`6`。





To solve this problem, you can use a priority queue data structure. A priority queue can efficiently insert elements and retrieve the maximum element. In Python, you can use the `heapq` module to implement a priority queue. However, Python's `heapq` module only provides a min-heap, so you need to insert the negative of the numbers to simulate a max-heap.

Here is a step-by-step plan:

1. Initialize an empty list `heap` to store the elements of the heap.
2. For each operation:
   - If the operation is "Push x", insert `-x` into the `heap`.
   - If the operation is "Print", if the size of the `heap` is less than `k`, print `-1`. Otherwise, create a copy of the `heap`, pop `k` elements from the copy, and print the negative of the last popped element.

Here is the Python code that implements this plan:

```python
import heapq

n, k = map(int, input().split())
heap = []

for _ in range(n):
    operation = input().split()
    if operation[0] == "Push":
        heapq.heappush(heap, -int(operation[1]))
    else:  # operation[0] == "Print"
        if len(heap) < k:
            print(-1)
        else:
            temp_heap = heap.copy()
            for _ in range(k):
                result = heapq.heappop(temp_heap)
            print(-result)
```

This code reads the number of operations and the value of `k` from the input, then for each operation, if the operation is "Push x", it inserts `-x` into the `heap`, and if the operation is "Print", it checks if the size of the `heap` is less than `k`, if so, it prints `-1`, otherwise, it creates a copy of the `heap`, pops `k` elements from the copy, and prints the negative of the last popped element.





### 7.6 数据流中位数

https://sunnywhy.com/sfbj/9/7/370

现有一个初始为空的序列 S，对其执行 n 个操作，每个操作是以下两种操作之一：

1. 往序列 S 中加入一个正整数 x ；
2. 输出当前序列 S 的中位数。

注：序列的中位数是指，将这个序列从小到大排序后最中间的那个元素；如果最中间有两个元素，那么取这两个元素的平均数作为序列的中位数。

**输入**

第一行一个整数 $n (2 \le n \le 10^4)$，表示操作个数；

接下来行，每行一个操作信息：使用"`Push x`"来表示往序列中加入正整数$x (1 \le x \le 10^5)$，使用"`Print`"来表示需要输出当前序列的中位数。

数据保证不会在序列为空时进行`Print`操作。

**输出**

每次执行`Print`操作时，输出一行，表示当前序列的中位数。结果保留一位小数。

样例1

输入

```
6
Push 3
Push 7
Push 6
Print
Push 1
Print
```

输出

```
6.0
4.5
```

解释

第一个`Print`时序列中元素为`3 7 6`，因此中位数是`6`；

第二个`Print`时序列中元素为`3 7 6 1`，因此中位数是。



To solve this problem, you can use two heaps: a max heap to store the smaller half of the numbers, and a min heap to store the larger half. The median is then either the maximum element in the max heap (when the total number of elements is odd) or the average of the maximum element in the max heap and the minimum element in the min heap (when the total number of elements is even).

Here is a step-by-step plan:

1. Initialize an empty max heap `left` and an empty min heap `right`.
2. For each operation:
   - If the operation is "Push x", insert `x` into the appropriate heap. If the size of the heaps differ by more than 1 after the insertion, balance the heaps by moving the top element from the heap with more elements to the heap with fewer elements.
   - If the operation is "Print", print the median. The median is the top element of `left` if the total number of elements is odd, or the average of the top elements of `left` and `right` if the total number of elements is even.

Here is the Python code that implements this plan:

```python
import heapq

n = int(input().strip())
left, right = [], []

for _ in range(n):
    operation = input().split()
    if operation[0] == "Push":
        x = int(operation[1])
        if not left or x <= -left[0]:
            heapq.heappush(left, -x)
        else:
            heapq.heappush(right, x)
        if len(left) < len(right):
            heapq.heappush(left, -heapq.heappop(right))
        elif len(left) > len(right) + 1:
            heapq.heappush(right, -heapq.heappop(left))
    else:  # operation[0] == "Print"
        if len(left) > len(right):
            print(f"{-left[0]:.1f}")
        else:
            print(f"{(-left[0] + right[0]) / 2:.1f}")
```

This code reads the number of operations from the input, then for each operation, if the operation is "Push x", it inserts `x` into the appropriate heap and balances the heaps if necessary, and if the operation is "Print", it prints the median.





## 8 哈夫曼树 3题

### 8.1 合并果子

https://sunnywhy.com/sfbj/9/8

有 n 堆果子，每堆果子的质量已知，现在需要把这些果子合并成一堆，但是每次只能把两堆果子合并到一起，同时会消耗与两堆果子质量之和等值的体力。显然，在进行 n - 1 次合并之后，就只剩下一堆了。为了尽可能节省体力，需要使耗费的总体力最小。求需要耗费的最小总体力。

**输入**

第一行一个整数$n (1 \le n \le 100)$，表示果子的堆数；

第二行为用空格隔开的 n 个正整数（每个正整数均不超过`100`），表示每堆果子的质量。

**输出**

输出一个整数，表示需要耗费的最小总体力。

样例1

输入

```
3
1 2 9
```

输出

```
15
```

解释

先将质量为`1`的果堆和质量为`2`的果堆合并，得到质量为`3`的果堆，同时消耗体力值`3`；

接着将质量为`3`的果堆和质量为`9`的果堆合并，得到质量为`12`的果堆，同时消耗体力值`12`；

因此共消耗体力值。



To solve this problem, you can use a priority queue data structure. A priority queue can efficiently insert elements and retrieve the minimum element. In Python, you can use the `heapq` module to implement a priority queue.

Here is a step-by-step plan:

1. Initialize an empty min heap `heap`.
2. For each pile of fruits, insert its weight into the `heap`.
3. While there is more than one pile of fruits, remove the two piles with the smallest weights from the `heap`, add their weights together, add the result to the total energy consumption, and insert the result back into the `heap`.
4. Print the total energy consumption.

Here is the Python code that implements this plan:

```python
import heapq

n = int(input().strip())
heap = list(map(int, input().strip().split()))
heapq.heapify(heap)

energy = 0
while len(heap) > 1:
    a = heapq.heappop(heap)
    b = heapq.heappop(heap)
    energy += a + b
    heapq.heappush(heap, a + b)

print(energy)
```

This code reads the number of piles of fruits and the weights of the piles from the input, inserts each weight into the `heap`, while there is more than one pile of fruits, removes the two piles with the smallest weights from the `heap`, adds their weights together, adds the result to the total energy consumption, and inserts the result back into the `heap`, and then prints the total energy consumption.



### 8.2 树的最小带权路径长度

https://sunnywhy.com/sfbj/9/8/372

对一棵 n 个结点的树（结点编号为从`0`到`n-1`，根结点为`0`号结点，每个结点有各自的权值）来说：

1. 结点的路径长度是指，从根结点到该结点的边数；
2. 结点的带权路径长度是指，结点权值乘以结点的路径长度；
3. 树的带权路径长度是指，树的所有叶结点的带权路径长度之和。

现有 n 个不同的正整数，需要寻找一棵树，使得树的所有叶子结点的权值恰好为这 n 个数，并且使得这棵树的带权路径长度是所有可能的树中最小的。求最小带权路径长度。

**输入**

第一行一个整数 $n (1 \le n \le 50)$，表示正整数的个数；

第二行为用空格隔开的 n 个正整数（每个正整数均不超过`100`），含义如题意所示。

**输出**

输出一个整数，表示最小带权路径长度。

样例1

输入

```
3
1 2 9
```

输出

```
15
```

解释

对应最小带权路径长度的树如下图所示，其中黑色数字为结点编号，编号右下角的格式为`结点权值*结点路径长度=结点带权路径长度`。由此可得树的带权路径长度为，是所有可能的树中最小的。

![树的最小带权路径长度.png](https://raw.githubusercontent.com/GMyhf/img/main/img/202403210139224.png)



To solve this problem, you can use a priority queue data structure. A priority queue can efficiently insert elements and retrieve the minimum element. In Python, you can use the `heapq` module to implement a priority queue.

Here is a step-by-step plan:

1. Initialize an empty min heap `heap`.
2. For each weight, insert it into the `heap`.
3. While there is more than one weight in the `heap`, remove the two weights with the smallest values from the `heap`, add them together, add the result to the total weighted path length, and insert the result back into the `heap`.
4. Print the total weighted path length.

Here is the Python code that implements this plan:

```python
import heapq

n = int(input().strip())
heap = list(map(int, input().strip().split()))
heapq.heapify(heap)

weighted_path_length = 0
while len(heap) > 1:
    a = heapq.heappop(heap)
    b = heapq.heappop(heap)
    weighted_path_length += a + b
    heapq.heappush(heap, a + b)

print(weighted_path_length)
```

This code reads the number of weights from the input, inserts each weight into the `heap`, while there is more than one weight in the `heap`, removes the two weights with the smallest values from the `heap`, adds them together, adds the result to the total weighted path length, and inserts the result back into the `heap`, and then prints the total weighted path length.



### 8.3 最小前缀编码长度

https://sunnywhy.com/sfbj/9/8/373

现需要将一个字符串 s 使用**前缀编码**的方式编码为 01 串，使得解码时不会产生混淆。求编码出的 01 串的最小长度。

**输入**

一个仅由大写字母组成、长度不超过的字符串。

**输出**

输出一个整数，表示最小长度。

样例1

输入

```
ABBC
```

输出

```
6
```

解释

将`A`编码为`00`，`B`编码为`1`，`C`编码为`01`，可以得到`ABBC`的前缀编码串`001101`，此时达到了所有可能情况中的最小长度`6`。



解法1:

使用一种基于哈夫曼编码的方法。哈夫曼编码是一种用于无损数据压缩的最优前缀编码方法。简单来说，它通过创建一棵二叉树，其中每个叶节点代表一个字符，每个节点的路径长度（从根到叶）代表该字符编码的长度，来生成最短的编码。字符出现的频率越高，其在树中的路径就越短，这样可以保证整个编码的总长度最小。

首先需要统计输入字符串中每个字符的出现频率。然后，根据这些频率构建哈夫曼树。构建完成后，遍历这棵树以确定每个字符的编码长度。最后，将所有字符的编码长度乘以其出现次数，累加起来，就得到了编码后的字符串的最小长度。

```python
from collections import Counter
import heapq

class HuffmanNode:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None
    
    # 为了让节点可以在优先队列中被比较，定义比较方法
    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(frequencies):
    priority_queue = [HuffmanNode(char, freq) for char, freq in frequencies.items()]
    heapq.heapify(priority_queue)
    
    while len(priority_queue) > 1:
        left = heapq.heappop(priority_queue)
        right = heapq.heappop(priority_queue)

        merged = HuffmanNode(None, left.freq + right.freq)
        merged.left = left
        merged.right = right

        heapq.heappush(priority_queue, merged)
    
    return priority_queue[0]

def calculate_huffman_code_lengths(node, length=0):
    if node is None:
        return {}
    if node.char is not None:
        return {node.char: length}
    left_lengths = calculate_huffman_code_lengths(node.left, length + 1)
    right_lengths = calculate_huffman_code_lengths(node.right, length + 1)
    left_lengths.update(right_lengths)
    return left_lengths

def min_encoding_length(s):
    frequencies = Counter(s)
    huffman_tree = build_huffman_tree(frequencies)
    code_lengths = calculate_huffman_code_lengths(huffman_tree)
    return sum(frequencies[char] * length for char, length in code_lengths.items())


s = input().strip()
print(min_encoding_length(s)) 
```





解法2:

To solve this problem, you can use Huffman coding, which is a common algorithm for constructing prefix codes. Huffman coding is a greedy algorithm that uses a priority queue where the node with the lowest probability is given the highest priority.

Here is a step-by-step plan:

1. Count the frequency of each character in the string.
2. Create a priority queue (min heap) where each node contains a character and its frequency. The priority of a node is determined by its frequency.
3. While there is more than one node in the queue:
   - Dequeue the two nodes with the highest priority (lowest frequency).
   - Create a new node with these two nodes as children and with frequency equal to the sum of the two nodes' frequencies.
   - Enqueue the new node.
4. The remaining node is the root of the Huffman tree. Traverse the Huffman tree and assign a binary code to each character. The binary code of a character is the path from the root to the character, where going to the left child adds a '0' and going to the right child adds a '1'.
5. Replace each character in the string with its binary code. The length of the resulting string is the minimum length of the prefix encoding.

Here is the Python code that implements this plan:

```python
import heapq
from collections import Counter

def huffman_encoding(s):
    # Count the frequency of each character
    freq = Counter(s)

    # Create a priority queue with single node trees
    heap = [[weight, [char, ""]] for char, weight in freq.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

    huff = sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[-1]), p))

    # Compute the length of the encoded string
    length = sum(freq[char] * len(code) for char, code in huff)

    return length

s = input().strip()
print(huffman_encoding(s))
```

This code reads a string from the input, counts the frequency of each character, creates a priority queue with single node trees, while there is more than one node in the queue, dequeues the two nodes with the highest priority (lowest frequency), creates a new node with these two nodes as children and with frequency equal to the sum of the two nodes' frequencies, enqueues the new node, the remaining node is the root of the Huffman tree, traverses the Huffman tree and assigns a binary code to each character, replaces each character in the string with its binary code, and then prints the length of the resulting string.



# 图算法专题（33题）



## 1 图的定义和相关术语 2题  

### sy374 无向图的度 简单

https://sunnywhy.com/sfbj/10/1/374

现有一个共n个顶点、m条边的无向图（假设顶点编号为从`0`到`n-1`），求每个顶点的度。

**输入**

第一行两个整数n、m（$1 \le n \le 100,0 \le m \le \frac{n(n-1)}2$），分别表示顶点数和边数；

接下来m行，每行两个整数u、v（$0 \le u \le n-1,0 \le v \le n-1, u \ne v$），表示一条边的两个端点的编号。数据保证不会有重边。

**输出**

在一行中输出n个整数，表示编号为从`0`到`n-1`的顶点的度。整数之间用空格隔开，行末不允许有多余的空格。

样例1

输入

```
3 2
0 1
0 2
```

输出

```
2 1 1
```

解释

对应的无向图如下图所示，`0`号顶点的度为`2`，`1`号和`2`号顶点的度为`1`。

![无向图的度.png](https://raw.githubusercontent.com/GMyhf/img/main/img/602b5ae4-6958-4a1a-8d5a-00e9d888e3c1.png)





为了求解每个顶点的度，我们可以创建一个列表来存储每个顶点的度，初始值都为0。然后，对于每条边，我们将边的两个端点的度都加1。

以下是实现这个过程的Python代码：

```python
n, m = map(int, input().split())
degrees = [0] * n
for _ in range(m):
    u, v = map(int, input().split())
    degrees[u] += 1
    degrees[v] += 1

print(' '.join(map(str, degrees)))
```

这段代码首先读取输入，然后创建一个列表来存储每个顶点的度。然后，它遍历每条边，将边的两个端点的度都加1。最后，它输出每个顶点的度。





### sy375: 有向图的度 简单

https://sunnywhy.com/sfbj/10/1/375

现有一个共n个顶点、m条边的有向图（假设顶点编号为从`0`到`n-1`），求每个顶点的入度和出度。

**输入**

第一行两个整数n、m（$1 \le n \le 100,0 \le m \le n(n-1)$），分别表示顶点数和边数；

接下来m行，每行两个整数u、v（$0 \le u \le n-1,0 \le v \le n-1, u \ne v$），表示一条边的两个端点的编号。数据保证不会有重边。

**输出**

输出行，每行为编号从`0`到`n-1`的一个顶点的入度和出度，中间用空格隔开。

样例1

输入

```
3 3
0 1
0 2
2 1
```

输出

```
0 2
2 0
1 1
```

解释

对应的有向图如下图所示。

`0`号顶点有`0`条入边，`2`条出边，因此入度为`0`，出度为`2`；

`1`号顶点有`2`条入边，`0`条出边，因此入度为`2`，出度为`0`；

`2`号顶点有`1`条入边，`1`条出边，因此入度为`1`，出度为`1`。

![有向图的度.png](https://raw.githubusercontent.com/GMyhf/img/main/img/21cec140-e555-4cb5-9ca9-290e373a782f.png)





为了求解每个顶点的入度和出度，我们可以创建两个列表来分别存储每个顶点的入度和出度，初始值都为0。然后，对于每条边，我们将起点的出度加1，终点的入度加1。

以下是实现这个过程的Python代码：

```python
n, m = map(int, input().split())
in_degrees = [0] * n
out_degrees = [0] * n
for _ in range(m):
    u, v = map(int, input().split())
    out_degrees[u] += 1
    in_degrees[v] += 1

for i in range(n):
    print(in_degrees[i], out_degrees[i])
```

这段代码首先读取输入，然后创建两个列表来存储每个顶点的入度和出度。然后，它遍历每条边，将边的起点的出度加1，终点的入度加1。最后，它输出每个顶点的入度和出度。





## 2 图的存储 4题



### sy376: 无向图的邻接矩阵 简单

https://sunnywhy.com/sfbj/10/2/376

现有一个共n个顶点、m条边的无向图（假设顶点编号为从`0`到`n-1`），将其按邻接矩阵的方式存储（存在边的位置填充`1`，不存在边的位置填充`0`），然后输出整个邻接矩阵。

**输入**

第一行两个整数n、m（$1 \le n \le 100,0 \le m \le \frac{n(n-1)}2$），分别表示顶点数和边数；

接下来m行，每行两个整数u、v（$0 \le u \le n-1,0 \le v \le n-1, u \ne v$），表示一条边的两个端点的编号。数据保证不会有重边。

**输出**

输出n行n列，表示邻接矩阵。整数之间用空格隔开，行末不允许有多余的空格。

样例1

输入

```
3 2
0 1
0 2
```

输出

```
0 1 1
1 0 0
1 0 0
```

解释

对应的无向图如下图所示。

`0`号顶点有`2`条出边，分别到达`1`号顶点和`2`号顶点；

`1`号顶点有`1`条出边，到达`0`号顶点；

`2`号顶点有`1`条出边，到达`0`号顶点。

![无向图的度.png](https://raw.githubusercontent.com/GMyhf/img/main/img/602b5ae4-6958-4a1a-8d5a-00e9d888e3c1.png)





为了将无向图按邻接矩阵的方式存储，我们可以创建一个n*n的二维列表，初始值都为0。然后，对于每条边，我们将边的两个端点对应的位置填充为1。

以下是实现这个过程的Python代码：

```python
n, m = map(int, input().split())
adjacency_matrix = [[0]*n for _ in range(n)]
for _ in range(m):
    u, v = map(int, input().split())
    adjacency_matrix[u][v] = 1
    adjacency_matrix[v][u] = 1

for row in adjacency_matrix:
    print(' '.join(map(str, row)))
```

这段代码首先读取输入，然后创建一个n*n的二维列表来存储邻接矩阵。然后，它遍历每条边，将边的两个端点对应的位置填充为1。最后，它输出整个邻接矩阵。





### sy377: 有向图的邻接矩阵 简单

https://sunnywhy.com/sfbj/10/2/377

现有一个共n个顶点、m条边的有向图（假设顶点编号为从`0`到`n-1`），将其按邻接矩阵的方式存储（存在边的位置填充`1`，不存在边的位置填充`0`），然后输出整个邻接矩阵。

**输入**

第一行两个整数n、m（$1 \le n \le 100,0 \le m \le n(n-1)$），分别表示顶点数和边数；

接下来m行，每行两个整数u、v（$0 \le u \le n-1,0 \le v \le n-1, u \ne v$），表示一条边的起点和终点的编号。数据保证不会有重边。

**输出**

输出n行n列，表示邻接矩阵。整数之间用空格隔开，行末不允许有多余的空格。

样例1

输入

```
3 3
0 1
0 2
2 1
```

输出

```
0 1 1
0 0 0
0 1 0
```

解释

对应的有向图如下图所示。

`0`号顶点有`2`条出边，分别到达`1`号顶点和`2`号顶点；

`1`号顶点有`0`条出边；

`2`号顶点有`1`条出边，到达`1`号顶点。

![有向图的度.png](https://raw.githubusercontent.com/GMyhf/img/main/img/21cec140-e555-4cb5-9ca9-290e373a782f.png)





为了将有向图按邻接矩阵的方式存储，我们可以创建一个n*n的二维列表，初始值都为0。然后，对于每条边，我们将边的起点和终点对应的位置填充为1。

以下是实现这个过程的Python代码：

```python
n, m = map(int, input().split())
adjacency_matrix = [[0]*n for _ in range(n)]
for _ in range(m):
    u, v = map(int, input().split())
    adjacency_matrix[u][v] = 1

for row in adjacency_matrix:
    print(' '.join(map(str, row)))
```

这段代码首先读取输入，然后创建一个n*n的二维列表来存储邻接矩阵。然后，它遍历每条边，将边的起点和终点对应的位置填充为1。最后，它输出整个邻接矩阵。



### sy378: 无向图的邻接表 简单

https://sunnywhy.com/sfbj/10/2/378

现有一个共n个顶点、m条边的无向图（假设顶点编号为从`0`到`n-1`），将其按邻接表的方式存储，然后输出整个邻接表。

**输入**

第一行两个整数n、m（$1 \le n \le 100,0 \le m \le \frac{n(n-1)}2$），分别表示顶点数和边数；

接下来m行，每行两个整数u、v（$0 \le u \le n-1,0 \le v \le n-1, u \ne v$），表示一条边的两个端点的编号。数据保证不会有重边。

**输出**

输出行，按顺序给出编号从`0`到`n-1`的顶点的所有出边，每行格式如下：

```text
id(k) v_1 v_2 ... v_k
```

其中表示当前顶点的编号，表示该顶点的出边数量，、、...、表示条出边的终点编号（按边输入的顺序输出）。行末不允许有多余的空格。

样例1

输入

```
3 2
0 1
0 2
```

输出

```
0(2) 1 2
1(1) 0
2(1) 0
```

解释

对应的无向图如下图所示。

`0`号顶点有`2`条出边，分别到达`1`号顶点和`2`号顶点；

`1`号顶点有`1`条出边，到达`0`号顶点；

`2`号顶点有`1`条出边，到达`0`号顶点。

![无向图的度.png](https://raw.githubusercontent.com/GMyhf/img/main/img/602b5ae4-6958-4a1a-8d5a-00e9d888e3c1.png)





为了将无向图按邻接表的方式存储，我们可以创建一个列表，其中每个元素都是一个列表，表示一个顶点的所有邻接顶点。然后，对于每条边，我们将边的两个端点添加到对方的邻接列表中。

以下是实现这个过程的Python代码：

```python
n, m = map(int, input().split())
adjacency_list = [[] for _ in range(n)]
for _ in range(m):
    u, v = map(int, input().split())
    adjacency_list[u].append(v)
    adjacency_list[v].append(u)

for i in range(n):
    num = len(adjacency_list[i])
    if num == 0:
        print(f"{i}({num})")
    else:
        print(f"{i}({num})", ' '.join(map(str, adjacency_list[i])))
```

这段代码首先读取输入，然后创建一个列表来存储邻接表。然后，它遍历每条边，将边的两个端点添加到对方的邻接列表中。最后，它输出整个邻接表。





### sy379: 有向图的邻接表 简单

https://sunnywhy.com/sfbj/10/2/379

现有一个共n个顶点、m条边的有向图（假设顶点编号为从`0`到`n-1`），将其按邻接表的方式存储，然后输出整个邻接表。

**输入**

第一行两个整数n、m（$1 \le n \le 100,0 \le m \le n(n-1)$），分别表示顶点数和边数；

接下来m行，每行两个整数u、v（$0 \le u \le n-1,0 \le v \le n-1, u \ne v$），表示一条边的起点和终点的编号。数据保证不会有重边。

**输出**

输出行，按顺序给出编号从`0`到`n-1`的顶点的所有出边，每行格式如下：

```text
id(k) v_1 v_2 ... v_k
```

其中id表示当前顶点的编号，k表示该顶点的出边数量，v1、v2、...、vk表示k条出边的终点编号（按边输入的顺序输出）。行末不允许有多余的空格。

样例1

输入

```
3 3
0 1
0 2
2 1
```

输出

```
0(2) 1 2
1(0)
2(1) 1
```

解释

对应的有向图如下图所示。

`0`号顶点有`2`条出边，分别到达`1`号顶点和`2`号顶点；

`1`号顶点有`0`条出边；

`2`号顶点有`1`条出边，到达`1`号顶点。

![有向图的度.png](https://raw.githubusercontent.com/GMyhf/img/main/img/21cec140-e555-4cb5-9ca9-290e373a782f.png)





为了将有向图按邻接表的方式存储，我们可以创建一个列表，其中每个元素都是一个列表，表示一个顶点的所有邻接顶点。然后，对于每条边，我们将边的终点添加到起点的邻接列表中。

以下是实现这个过程的Python代码：

```python
n, m = map(int, input().split())
adjacency_list = [[] for _ in range(n)]
for _ in range(m):
    u, v = map(int, input().split())
    adjacency_list[u].append(v)

for i in range(n):
    num = len(adjacency_list[i])
    if num == 0:
        print(f"{i}({num})")
    else:
        print(f"{i}({num})", ' '.join(map(str, adjacency_list[i])))
```

这段代码首先读取输入，然后创建一个列表来存储邻接表。然后，它遍历每条边，将边的终点添加到起点的邻接列表中。最后，它输出整个邻接表。







## 3 图的遍历 6题



### sy380: 无向图的连通块 简单

https://sunnywhy.com/sfbj/10/3/380

现有一个共n个顶点、m条边的无向图（假设顶点编号为从`0`到`n-1`），求图中的连通块个数。

**输入**

第一行两个整数n、m（$1 \le n \le 100,0 \le m \le \frac{n(n-1)}2$），分别表示顶点数和边数；

接下来m行，每行两个整数u、v（$0 \le u \le n-1,0 \le v \le n-1, u \ne v$），表示一条边的两个端点的编号。数据保证不会有重边。

**输出**

输出一个整数，表示图中的连通块个数。

样例1

输入

```
6 5
0 1
0 3
3 5
2 4
1 5
```

输出

```
2
```

解释

对应的无向图如下图所示，共有两个连通块。

![无向图的连通块.png](https://raw.githubusercontent.com/GMyhf/img/main/img/d95ea531-1e5b-44e7-8d9c-8908a4bb1115.png)





要解决这个问题，我们可以使用深度优先搜索（DFS）或广度优先搜索（BFS）来找出图中的所有连通块。我们首先创建一个邻接列表来表示图，然后对每个未访问的节点进行DFS或BFS，每次搜索都会找到一个连通块。搜索过程中，我们需要标记已访问的节点，以防止重复访问。

以下是使用DFS的Python代码：

```python
def dfs(node, visited, adjacency_list):
    visited[node] = True
    for neighbor in adjacency_list[node]:
        if not visited[neighbor]:
            dfs(neighbor, visited, adjacency_list)

n, m = map(int, input().split())
adjacency_list = [[] for _ in range(n)]
for _ in range(m):
    u, v = map(int, input().split())
    adjacency_list[u].append(v)
    adjacency_list[v].append(u)

visited = [False] * n
connected_components = 0
for i in range(n):
    if not visited[i]:
        dfs(i, visited, adjacency_list)
        connected_components += 1

print(connected_components)
```

在这段代码中，我们首先读取顶点数和边数，然后读取每条边的信息并在邻接列表中添加相应的边。然后我们初始化一个布尔数组`visited`来跟踪已访问的节点，并初始化连通块的数量为0。接着，我们遍历每个节点，如果它还未被访问，我们就从它开始进行DFS，并将连通块的数量加1。最后，我们输出连通块的数量。



### sy381: 无向连通图 简单

https://sunnywhy.com/sfbj/10/3/381

现有一个共个顶点、条边的无向图（假设顶点编号为从`0`到`n-1`），判断其是否是连通图。

**输入**

第一行两个整数n、m（$1 \le n \le 100,0 \le m \le \frac{n(n-1)}2$），分别表示顶点数和边数；

接下来m行，每行两个整数u、v（$0 \le u \le n-1,0 \le v \le n-1, u \ne v$），表示一条边的起点和终点的编号。数据保证不会有重边。

**输出**

如果是连通图，那么输出`Yes`，否则输出`No`。

样例1

输入

```
6 5
0 1
0 3
3 5
2 4
1 5
```

输出

```
No
```

解释

对应的无向图如下图所示，不是连通图。

![无向图的连通块.png](https://raw.githubusercontent.com/GMyhf/img/main/img/d95ea531-1e5b-44e7-8d9c-8908a4bb1115.png)

样例2

输入

```
6 6
0 1
0 3
3 5
2 4
1 5
3 4
```

输出

```
Yes
```

解释

对应的无向图如下图所示，是连通图。

![无向连通图_样例2.png](https://raw.githubusercontent.com/GMyhf/img/main/img/7c186721-b047-4cc0-b3d9-329d17c53d82.png)





要判断一个图是否是连通图，我们可以使用深度优先搜索（DFS）或广度优先搜索（BFS）来遍历图。如果在遍历结束后，所有的节点都被访问过，那么这个图就是连通图。否则，就不是连通图。

以下是使用DFS的Python代码：

```python
def dfs(node, visited, adjacency_list):
    visited[node] = True
    for neighbor in adjacency_list[node]:
        if not visited[neighbor]:
            dfs(neighbor, visited, adjacency_list)

n, m = map(int, input().split())
adjacency_list = [[] for _ in range(n)]
for _ in range(m):
    u, v = map(int, input().split())
    adjacency_list[u].append(v)
    adjacency_list[v].append(u)

visited = [False] * n
dfs(0, visited, adjacency_list)

if all(visited):
    print("Yes")
else:
    print("No")
```

在这段代码中，我们首先读取顶点数和边数，然后读取每条边的信息并在邻接列表中添加相应的边。然后我们初始化一个布尔数组`visited`来跟踪已访问的节点，并从第一个节点开始进行DFS。最后，我们检查是否所有的节点都被访问过，如果是，那么输出`Yes`，否则输出`No`。



### sy382: 有向图判环 中等

https://sunnywhy.com/sfbj/10/3/382

现有一个共n个顶点、m条边的有向图（假设顶点编号为从`0`到`n-1`），如果从图中一个顶点出发，沿着图中的有向边前进，最后能回到这个顶点，那么就称其为图中的一个环。判断图中是否有环。

**输入**

第一行两个整数n、m（$1 \le n \le 100,0 \le m \le n(n-1)$），分别表示顶点数和边数；

接下来m行，每行两个整数u、v（$0 \le u \le n-1,0 \le v \le n-1, u \ne v$），表示一条边的起点和终点的编号。数据保证不会有重边。

**输出**

如果图中有环，那么输出`Yes`，否则输出`No`。

样例1

输入

```
4 4
1 0
0 3
3 2
2 1
```

输出

```
Yes
```

解释

对应的有向图如下图所示，存在`0->3->2->1->0`的环。

![有向图判环_样例1.png](https://raw.githubusercontent.com/GMyhf/img/main/img/c2072489-936c-40c8-8e89-7ddd87d76705.png)

样例2

输入

```
4 4
1 0
0 3
2 3
2 1
```

输出

```
No
```

解释

对应的有向图如下图所示，图中不存在环。

![有向图判环_样例2.png](https://raw.githubusercontent.com/GMyhf/img/main/img/cd48cf44-20f5-4c21-aaae-f1df3fcc3ba8.png)





在这个问题中，需要检查给定的有向图是否包含一个环。可以使用深度优先搜索（DFS）来解决这个问题。在DFS中，从一个节点开始，然后访问它的每一个邻居。如果在访问过程中，遇到了一个已经在当前路径中的节点，那么就存在一个环。可以使用一个颜色数组来跟踪每个节点的状态：未访问（0），正在访问（1），已访问（2）。

以下是解决这个问题的Python代码：

```python
def has_cycle(n, edges):
    graph = [[] for _ in range(n)]
    for u, v in edges:
        graph[u].append(v)

    color = [0] * n

    def dfs(node):
        if color[node] == 1:
            return True
        if color[node] == 2:
            return False

        color[node] = 1
        for neighbor in graph[node]:
            if dfs(neighbor):
                return True
        color[node] = 2
        return False

    for i in range(n):
        if dfs(i):
            return "Yes"
    return "No"

# 接收数据
n, m = map(int, input().split())
edges = []
for _ in range(m):
    u, v = map(int, input().split())
    edges.append((u, v))

# 调用函数
print(has_cycle(n, edges))
```

在这个函数中，我们首先构建了一个邻接列表来表示图。然后，我们对每个节点执行深度优先搜索。如果在搜索过程中，我们遇到了一个正在访问的节点，那么就存在一个环。如果我们遍历完所有的节点都没有找到环，那么就返回"No"。



### sy383: 最大权值连通块 中等

https://sunnywhy.com/sfbj/10/3/383

现有一个共个顶点、条边的无向图（假设顶点编号为从`0`到`n-1`），每个顶点有各自的权值。我们把一个连通块中所有顶点的权值之和称为这个连通块的权值。求图中所有连通块的最大权值。

**输入**

第一行两个整数n、m（$1 \le n \le 100,0 \le m \le \frac{n(n-1)}2$​），分别表示顶点数和边数；

第二行个用空格隔开的正整数（每个正整数不超过`100`），表示个顶点的权值。

接下来m行，每行两个整数u、v（$0 \le u \le n-1,0 \le v \le n-1, u \ne v$），表示一条边的起点和终点的编号。数据保证不会有重边。

**输出**

输出一个整数，表示连通块的最大权值。

样例1

输入

```
6 5
2 3 4 1 3 2
0 1
0 3
3 5
2 4
1 5
```

输出

```
8
```

解释

对应的无向图如下图所示，左边连通块的权值为，右边连通块的权值为，因此最大权值为`8`。

![最大权值连通块.png](https://raw.githubusercontent.com/GMyhf/img/main/img/0b6c3a49-118e-4432-bc17-e8d2f72d4410.png)





需要找到给定无向图中所有连通块的最大权值。使用深度优先搜索（DFS）来解决这个问题。在DFS中，从一个节点开始，然后访问它的每一个邻居。可以使用一个visited数组来跟踪每个节点是否已经被访问过。对于每个连通块，可以计算其权值之和，并更新最大权值。

以下是解决这个问题的Python代码：

```python
def max_weight(n, m, weights, edges):
    graph = [[] for _ in range(n)]
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    visited = [False] * n
    max_weight = 0

    def dfs(node):
        visited[node] = True
        total_weight = weights[node]
        for neighbor in graph[node]:
            if not visited[neighbor]:
                total_weight += dfs(neighbor)
        return total_weight

    for i in range(n):
        if not visited[i]:
            max_weight = max(max_weight, dfs(i))

    return max_weight

# 接收数据
n, m = map(int, input().split())
weights = list(map(int, input().split()))
edges = []
for _ in range(m):
    u, v = map(int, input().split())
    edges.append((u, v))

# 调用函数
print(max_weight(n, m, weights, edges))
```

在这段代码中，首先通过`input()`函数接收用户输入的顶点数`n`、边数`m`和每个顶点的权值，然后在一个循环中接收每条边的起点和终点，并将它们添加到`edges`列表中。然后，我们调用`max_weight`函数并打印结果。





### sy384: 无向图的顶点层号

https://sunnywhy.com/sfbj/10/3/384

现有一个共n个顶点、m条边的无向连通图（假设顶点编号为从`0`到`n-1`）。我们称从s号顶点出发到达其他顶点经过的最小边数称为各顶点的层号。求图中所有顶点的层号。

**输入**

第一行三个整数n、m、s（$1 \le n \le 100,0 \le m \le \frac{n(n-1)}2, 0 \le s \le n -1$​），分别表示顶点数、边数、起始顶点编号；

接下来m行，每行两个整数u、v（$0 \le u \le n-1,0 \le v \le n-1, u \ne v$），表示一条边的起点和终点的编号。数据保证不会有重边。

**输出**

输出n个整数，分别为编号从`0`到`n-1`的顶点的层号。整数之间用空格隔开，行末不允许有多余的空格。

样例1

输入

```
6 6 0
0 1
0 3
3 5
2 4
1 5
3 4
```

输出

```
0 1 3 1 2 2
```

解释

对应的无向图和顶点层号如下图所示。

![无向图的顶点层号.png](https://raw.githubusercontent.com/GMyhf/img/main/img/a896296d-23e0-4420-9e70-c4e979a78e89.png)





需要找到从给定的起始顶点到图中所有其他顶点的最短路径长度，这也被称为顶点的层号。可以使用广度优先搜索（BFS）来解决这个问题。在BFS中，从起始节点开始，然后访问它的所有邻居，然后再访问这些邻居的邻居，依此类推。我们可以使用一个队列来跟踪待访问的节点，并使用一个距离数组来记录从起始节点到每个节点的最短距离。

以下是解决这个问题的Python代码：

```python
from collections import deque

def bfs(n, m, s, edges):
    graph = [[] for _ in range(n)]
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    distance = [-1] * n
    distance[s] = 0

    queue = deque([s])
    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if distance[neighbor] == -1:
                distance[neighbor] = distance[node] + 1
                queue.append(neighbor)

    return distance

# 接收数据
n, m, s = map(int, input().split())
edges = []
for _ in range(m):
    u, v = map(int, input().split())
    edges.append((u, v))

# 调用函数
distances = bfs(n, m, s, edges)
print(' '.join(map(str, distances)))
```

在这段代码中，我们首先通过`input()`函数接收用户输入的顶点数`n`、边数`m`和起始顶点`s`，然后在一个循环中接收每条边的起点和终点，并将它们添加到`edges`列表中。然后，我们调用`bfs`函数并打印结果。





### sy385: 受限层号的顶点数 中等

https://sunnywhy.com/sfbj/10/3/385

现有一个共n个顶点、m条边的有向图（假设顶点编号为从`0`到`n-1`）。我们称从s号顶点出发到达其他顶点经过的最小边数称为各顶点的层号。求层号不超过的顶点个数。

**输入**

第一行四个整数n、m、s、k（$1 \le n \le 100,0 \le m \le \frac{n(n-1)}2, 0 \le s \le n -1, 0 \le k \le 100$​），分别表示顶点数、边数、起始顶点编号；

接下来m行，每行两个整数u、v（$0 \le u \le n-1,0 \le v \le n-1, u \ne v$），表示一条边的起点和终点的编号。数据保证不会有重边。

**输出**

输出一个整数，表示层号不超过的顶点个数。

样例1

输入

```
6 6 0 2
0 1
0 3
3 5
4 2
3 4
5 2
```

输出

```
5
```

解释

对应的有向图和顶点层号如下图所示，层号不超过`2`的顶点有`5`个。

![受限层号的顶点数.png](https://raw.githubusercontent.com/GMyhf/img/main/img/4f4d78cc-db0d-4a94-b195-b8fcc14e1d36.png)





需要找到从给定的起始顶点到图中所有其他顶点的最短路径长度（也被称为顶点的层号），并计算层号不超过k的顶点个数。可以使用广度优先搜索（BFS）来解决这个问题。在BFS中，从起始节点开始，然后访问它的所有邻居，然后再访问这些邻居的邻居，依此类推。可以使用一个队列来跟踪待访问的节点，并使用一个距离数组来记录从起始节点到每个节点的最短距离。

以下是解决这个问题的Python代码：

```python
from collections import deque

def bfs(n, m, s, k, edges):
    graph = [[] for _ in range(n)]
    for u, v in edges:
        graph[u].append(v)  # 只按照输入的方向添加边

    distance = [-1] * n
    distance[s] = 0

    queue = deque([s])
    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if distance[neighbor] == -1:
                distance[neighbor] = distance[node] + 1
                queue.append(neighbor)

    return sum(1 for d in distance if d <= k and d != -1)

# 接收数据
n, m, s, k = map(int, input().split())
edges = []
for _ in range(m):
    u, v = map(int, input().split())
    edges.append((u, v))

# 调用函数
count = bfs(n, m, s, k, edges)
print(count)
```

在这段代码中，首先通过`input()`函数接收用户输入的顶点数`n`、边数`m`、起始顶点`s`和层号上限`k`，然后在一个循环中接收每条边的起点和终点，并将它们添加到`edges`列表中。然后，调用`bfs`函数并打印结果。





## 4 最短路径 10题



### sy386: 最短距离 简单

https://sunnywhy.com/sfbj/10/4/386

现有一个共n个顶点（代表城市）、m条边（代表道路）的无向图（假设顶点编号为从`0`到`n-1`），每条边有各自的边权，代表两个城市之间的距离。求从s号城市出发到达t号城市的最短距离。

**输入**

第一行四个整数n、m、s、t（$1 \le n \le 100,0 \le m \le \frac{n(n-1)}2, 0 \le s \le n -1, 0 \le t \le n-1$​），分别表示顶点数、边数、起始编号、终点编号；

接下来m行，每行三个整数u、v、w（$0 \le u \le n-1,0 \le v \le n-1, u \ne v, 1 \le w \le 100$），表示一条边的两个端点的编号及边权距离。数据保证不会有重边。

**输出**

输出一个整数，表示最短距离。如果无法到达，那么输出`-1`。

样例1

输入

```
6 6 0 2
0 1 2
0 2 5
0 3 1
2 3 2
1 2 1
4 5 1
```

输出

```
3
```

解释

对应的无向图如下图所示。

共有`3`条从`0`号顶点到`2`号顶点的路径：

1. `0->3->2`：距离为`3`；
2. `0->2`：距离为`5`；
3. `0->1->2`：距离为`3`。

因此最短距离为`3`。

![最短距离.png](https://raw.githubusercontent.com/GMyhf/img/main/img/1123ea31-976a-43fb-bc9d-11eec6ce0f26.png)

样例2

输入

```
6 6 0 5
0 1 2
0 2 5
0 3 1
2 3 2
1 2 1
4 5 1
```

输出

```
-1
```

解释

和第一个样例相同的图，终点换成了`5`号顶点，显然从`0`号无法到达`5`号。



需要找到从给定的起始城市到目标城市的最短距离。可以使用Dijkstra算法来解决这个问题。Dijkstra算法是一种用于在图中找到最短路径的算法。它从起始节点开始，然后逐步扩展到所有可达的节点，每次选择当前最短的路径进行扩展。

以下是使用 Python 实现 Dijkstra 算法来解决这个问题的示例代码：

```python
import heapq

def dijkstra(n, edges, s, t):
    graph = [[] for _ in range(n)]
    for u, v, w in edges:
        graph[u].append((v, w))
        graph[v].append((u, w))

    pq = [(0, s)]  # (distance, node)
    visited = set()
    distances = [float('inf')] * n
    distances[s] = 0

    while pq:
        dist, node = heapq.heappop(pq)
        if node == t:
            return dist
        if node in visited:
            continue
        visited.add(node)
        for neighbor, weight in graph[node]:
            if neighbor not in visited:
                new_dist = dist + weight
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    heapq.heappush(pq, (new_dist, neighbor))
    return -1

# Read input
n, m, s, t = map(int, input().split())
edges = [list(map(int, input().split())) for _ in range(m)]

# Solve the problem and print the result
result = dijkstra(n, edges, s, t)
print(result)
```

这段代码实现了 Dijkstra 算法来求解从起点到终点的最短路径。首先构建了一个图，然后使用优先队列来选择下一个要探索的节点，并在探索过程中更新最短距离。最后返回从起点到终点的最短距离。

这个版本的Dijkstra算法使用了一个集合`visited`来记录已经访问过的节点，这样可以避免对同一个节点的重复处理。当我们从优先队列中取出一个节点时，如果这个节点已经在`visited`集合中，那么我们就跳过这个节点，处理下一个节点。这样可以提高算法的效率。

此外，这个版本的Dijkstra算法还在找到目标节点`t`时就立即返回结果，而不是等到遍历完所有节点。这是因为Dijkstra算法保证了每次从优先队列中取出的节点就是当前距离最短的节点，所以当我们找到目标节点`t`时，就已经找到了从起始节点`s`到`t`的最短路径，无需再继续搜索。

这个版本的Dijkstra算法的时间复杂度仍然是O((V+E)logV)，其中V是顶点数，E是边数。这是因为每个节点最多会被加入到优先队列中一次（当找到一条更短的路径时），并且每条边都会被处理一次（在遍历节点的邻居时）。优先队列的插入和删除操作的时间复杂度都是O(logV)，所以总的时间复杂度是O((V+E)logV)。



Dijkstra 算法是一种经典的图算法，它综合运用了多种技术，包括邻接表、集合、优先队列（堆）、贪心算法和动态规划的思想。例题：最短距离，https://sunnywhy.com/sfbj/10/4/386

- 邻接表：Dijkstra 算法通常使用邻接表来表示图的结构，这样可以高效地存储图中的节点和边。
- 集合：在算法中需要跟踪已经访问过的节点，以避免重复访问，这一般使用集合（或哈希集合）来实现。
- 优先队列（堆）：Dijkstra 算法中需要选择下一个要探索的节点，通常使用优先队列（堆）来维护当前候选节点的集合，并确保每次都能快速找到距离起点最近的节点。
- 贪心算法：Dijkstra 算法每次选择距离起点最近的节点作为下一个要探索的节点，这是一种贪心策略，即每次做出局部最优的选择，期望最终能达到全局最优。
- 动态规划：Dijkstra 算法通过不断地更新节点的最短距离来逐步得到从起点到各个节点的最短路径，这是一种动态规划的思想，即将原问题拆解成若干子问题，并以最优子结构来解决。

综合运用这些技术，Dijkstra 算法能够高效地求解单源最短路径问题，对于解决许多实际问题具有重要意义。





第2种写法，没有用set记录访问过的结点。

```python
import heapq

def dijkstra(n, s, t, edges):
    graph = [[] for _ in range(n)]
    for u, v, w in edges:
        graph[u].append((v, w))
        graph[v].append((u, w))

    distance = [float('inf')] * n
    distance[s] = 0

    queue = [(0, s)]
    while queue:
        dist, node = heapq.heappop(queue)
        if dist != distance[node]:
            continue
        for neighbor, weight in graph[node]:
            if distance[node] + weight < distance[neighbor]:
                distance[neighbor] = distance[node] + weight
                heapq.heappush(queue, (distance[neighbor], neighbor))

    return distance[t] if distance[t] != float('inf') else -1

# 接收数据
n, m, s, t = map(int, input().split())
edges = []
for _ in range(m):
    u, v, w = map(int, input().split())
    edges.append((u, v, w))

# 调用函数
min_distance = dijkstra(n, s, t, edges)
print(min_distance)
```

第15行的判断`if dist != distance[node]: continue`的作用是跳过已经找到更短路径的节点。

在Dijkstra算法中，我们使用优先队列（在Python中是heapq）来存储待处理的节点，每次从队列中取出当前距离最短的节点进行处理。但是在处理过程中，有可能会多次将同一个节点加入到队列中，因为我们可能会通过不同的路径到达同一个节点，每次到达时都会将其加入到队列中。

因此，当我们从队列中取出一个节点时，需要判断这个节点当前的最短距离是否与队列中存储的距离相同。如果不同，说明这个节点在队列中等待处理的时候，已经有了一条更短的路径，所以我们可以跳过这个节点，处理下一个节点。



### sy387: 最短距离-多终点 简单

https://sunnywhy.com/sfbj/10/4/387

现有一个共n个顶点（代表城市）、m条边（代表道路）的无向图（假设顶点编号为从`0`到`n-1`），每条边有各自的边权，代表两个城市之间的距离。求从s号城市出发到达其他每个城市的最短距离。

**输入**

第一行三个整数n、m、s（$1 \le n \le 100,0 \le m \le \frac{n(n-1)}2, 0 \le s \le n -1$​），分别表示顶点数、边数、起始编号；

接下来m行，每行三个整数u、v、w（$0 \le u \le n-1,0 \le v \le n-1, u \ne v, 1 \le w \le 100$），表示一条边的两个端点的编号及边权距离。数据保证不会有重边。

**输出**

在一行中输出个整数，依次表示到达编号从`0`到`n-1`的顶点的最短距离。如果无法到达，那么输出`-1`。整数之间用空格隔开，行末不允许有多余的空格。

样例1

输入

```
6 6 0
0 1 2
0 2 5
0 3 1
2 3 2
1 2 1
4 5 1
```

输出

```
0 2 3 1 -1 -1
```

解释

对应的无向图如下图所示。最短路径：

`0`号顶点：直接到达，距离为`0`；

`1`号顶点：`0->1`，距离为`2`；

`2`号顶点：`0->1->2`和`0->3->2`，距离为`3`；

`3`号顶点：`0->3`，距离为`1`；

`4`号顶点：无法到达；

`5`号顶点：无法到达。

![最短距离.png](https://raw.githubusercontent.com/GMyhf/img/main/img/1123ea31-976a-43fb-bc9d-11eec6ce0f26.png)





```python
import heapq

def dijkstra(n, edges, s):
    graph = [[] for _ in range(n)]
    for u, v, w in edges:
        graph[u].append((v, w))
        graph[v].append((u, w))

    pq = [(0, s)]  # (distance, node)
    visited = set()
    distances = [float('inf')] * n
    distances[s] = 0

    while pq:
        dist, node = heapq.heappop(pq)
        if node in visited:
            continue
        visited.add(node)
        for neighbor, weight in graph[node]:
            new_dist = dist + weight
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                heapq.heappush(pq, (new_dist, neighbor))

    return distances

# Read input
n, m, s = map(int, input().split())
edges = [list(map(int, input().split())) for _ in range(m)]

# Solve the problem
result = dijkstra(n, edges, s)

# Output the result
print(' '.join(map(lambda x: str(x) if x != float('inf') else '-1', result)))

```





### sy388: 最短距离-多边权 简单

https://sunnywhy.com/sfbj/10/4/386

现有一个共个顶点（代表城市）、条边（代表道路）的无向图（假设顶点编号为从`0`到`n-1`），每条边有两种边权，分别代表两个城市之间的距离和花费。求从号城市出发到达号城市的最短距离，并在达到最短距离的路径中计算最少花费。

**输入**

第一行四个整数n、m、s、t（$1 \le n \le 100,0 \le m \le \frac{n(n-1)}2, 0 \le s \le n -1, 0 \le t \le n-1$​），分别表示顶点数、边数、起始编号、终点编号；

接下来m行，每行四个整数u、v、w1、w2（$0 \le u \le n-1,0 \le v \le n-1, u \ne v, 1 \le w_1 \le 100,1 \le w_2 \le 100$），表示一条边的两个端点的编号及边权距离、边权花费。数据保证不会有重边。

**输出**

输出两个整数，代表最短距离与最少花费。

数据保证最短路径一定存在。

样例1

输入

```
4 5 0 2
0 1 2 1
0 2 5 1
0 3 1 2
1 2 1 6
3 2 2 3
```

输出

```
3 5
```

解释

对应的无向图如下图所示，其中边上的第一个数字为边权距离，第二个数字为边权花费。

共有`3`条从`0`号顶点到`2`号顶点的路径：

1. `0->3->2`：距离为`3`，花费为`5`；
2. `0->2`：距离为`5`，花费为`1`；
3. `0->1->2`：距离为`3`，花费为`7`。

因此最短距离为`3`，最短路径有`2`条，在这`2`条最短路径中的最少花费是`5`。

![最短距离-多边权.png](https://raw.githubusercontent.com/GMyhf/img/main/img/93f41b80-9756-48bf-b0af-a35bc94e9616.png)



```python
import heapq

def dijkstra(n, edges, s, t):
    graph = [[] for _ in range(n)]
    for u, v, d, c in edges:
        graph[u].append((v, d, c))
        graph[v].append((u, d, c))

    pq = [(0, 0, s)]  # (distance, cost, node)
    visited = set()
    distances = [(float('inf'), float('inf'))] * n
    distances[s] = (0, 0)

    while pq:
        dist, cost, node = heapq.heappop(pq)
        if node == t:
            return dist, cost
        if node in visited:
            continue
        visited.add(node)
        for neighbor, d, c in graph[node]:
            new_dist = dist + d
            new_cost = cost + c
            if new_dist < distances[neighbor][0] or (new_dist == distances[neighbor][0] and new_cost < distances[neighbor][1]):
                distances[neighbor] = (new_dist, new_cost)
                heapq.heappush(pq, (new_dist, new_cost, neighbor))

# Read input
n, m, s, t = map(int, input().split())
edges = [list(map(int, input().split())) for _ in range(m)]

# Solve the problem
result_distance, result_cost = dijkstra(n, edges, s, t)

# Output the result
print(result_distance, result_cost)

```







### sy389: 最短路径条数 简单

https://sunnywhy.com/sfbj/10/4/389

现有一个共个顶点（代表城市）、条边（代表道路）的无向图（假设顶点编号为从`0`到`n-1`），每条边有各自的边权，代表两个城市之间的距离。求从号城市出发到达号城市的最短距离和最短路径条数。

**输入**

第一行四个整数n、m、s、t（$1 \le n \le 100,0 \le m \le \frac{n(n-1)}2, 0 \le s \le n -1, 0 \le t \le n-1$​），分别表示顶点数、边数、起始编号、终点编号；

接下来m行，每行三个整数u、v、w（$0 \le u \le n-1,0 \le v \le n-1, u \ne v, 1 \le w \le 100$），表示一条边的两个端点的编号及边权距离。数据保证不会有重边。

**输出**

输出两个整数，表示最短距离和最短路径条数，中间用空格隔开。

数据保证最短路径一定存在。

样例1

输入

```
4 5 0 2
0 1 2
0 2 5
0 3 1
1 2 1
3 2 2
```

输出

```
3 2
```

解释

对应的无向图如下图所示。

共有`3`条从`0`号顶点到`2`号顶点的路径：

1. `0->3->2`：距离为`3`；
2. `0->2`：距离为`5`；
3. `0->1->2`：距离为`3`。

因此最短距离为`3`，最短路径有`2`条。

![最短路径条数.png](https://raw.githubusercontent.com/GMyhf/img/main/img/c88685d8-84ac-45dd-8cca-d1450fac9a71.png)



以下是使用 Dijkstra 算法解决这个问题的 Python 代码：

```python
import heapq

def dijkstra(n, edges, s, t):
    graph = [[] for _ in range(n)]
    for u, v, d in edges:
        graph[u].append((v, d))
        graph[v].append((u, d))

    pq = [(0, s)]  # (distance, node)
    visited = set()
    distances = [float('inf')] * n
    distances[s] = 0
    count = [0] * n
    count[s] = 1

    while pq:
        dist, node = heapq.heappop(pq)
        if node == t:
            return dist, count[node]
        if node in visited:
            continue
        visited.add(node)
        for neighbor, d in graph[node]:
            new_dist = dist + d
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                heapq.heappush(pq, (new_dist, neighbor))
                count[neighbor] = count[node]
            elif new_dist == distances[neighbor]:
                count[neighbor] += count[node]

# Read input
n, m, s, t = map(int, input().split())
edges = [list(map(int, input().split())) for _ in range(m)]

# Solve the problem
result_distance, result_count = dijkstra(n, edges, s, t)

# Output the result
print(result_distance, result_count)
```

这段代码首先构建了一个无向图的邻接表，并使用 Dijkstra 算法计算了从起始顶点到终点的最短距离和最短路径条数。然后将结果输出为两个整数，分别表示最短距离和最短路径条数。



### sy390: 最短路径 中等

https://sunnywhy.com/sfbj/10/4/390

现有一个共n个顶点（代表城市）、m条边（代表道路）的无向图（假设顶点编号为从`0`到`n-1`），每条边有各自的边权，代表两个城市之间的距离。求从s号城市出发到达t号城市的最短距离和最短路径。

**输入**

第一行四个整数n、m、s、t（$1 \le n \le 100,0 \le m \le \frac{n(n-1)}2, 0 \le s \le n -1, 0 \le t \le n-1$​），分别表示顶点数、边数、起始编号、终点编号；

接下来m行，每行三个整数u、v、w（$0 \le u \le n-1,0 \le v \le n-1, u \ne v, 1 \le w \le 100$），表示一条边的两个端点的编号及边权距离。数据保证不会有重边。

**输出**

```text
min_d v_1->v_2->...->v_k
```

按上面的格式输出最短距离和最短路径，其中`min_d`表示最短距离，`v_1`、`v_2`、...、`v_k`表示从起点到终点的最短路径上的顶点编号。

数据保证最短路径存在且唯一。

样例1

输入

```
4 5 0 2
0 1 2
0 2 5
0 3 1
1 2 2
3 2 2
```

输出

```
3 0->3->2
```

解释

对应的无向图如下图所示。

共有`3`条从`0`号顶点到`2`号顶点的路径：

1. `0->3->2`：距离为`3`；
2. `0->2`：距离为`5`；
3. `0->1->2`：距离为`4`。

因此最短距离为`3`，最短路径为`0->3->2`。

![最短路径条数.png](https://raw.githubusercontent.com/GMyhf/img/main/img/c88685d8-84ac-45dd-8cca-d1450fac9a71.png)







```python
import heapq

def dijkstra(graph, start, end):
    n = len(graph)
    visited = [False] * n
    distance = [float('inf')] * n
    prev = [-1] * n
    distance[start] = 0
    pq = [(0, start)]  # priority queue

    while pq:
        dist_u, u = heapq.heappop(pq)
        if visited[u]:
            continue
        visited[u] = True

        for v, w in graph[u]:
            if not visited[v] and dist_u + w < distance[v]:
                distance[v] = dist_u + w
                prev[v] = u
                heapq.heappush(pq, (distance[v], v))

    # Reconstruct path
    path = []
    while end != -1:
        path.append(end)
        end = prev[end]
    path.reverse()

    return distance, path

def main():
    n, m, s, t = map(int, input().split())
    graph = [[] for _ in range(n)]
    for _ in range(m):
        u, v, w = map(int, input().split())
        graph[u].append((v, w))
        graph[v].append((u, w))

    distance, path = dijkstra(graph, s, t)
    print(distance[t], '->'.join(map(str, path)))

if __name__ == "__main__":
    main()

```





### sy391: 最短路径-多边权 中等

https://sunnywhy.com/sfbj/10/4/391

现有一个共n个顶点（代表城市）、m条边（代表道路）的无向图（假设顶点编号为从`0`到`n-1`），每条边有两种边权，分别代表两个城市之间的距离和花费。求从s号城市出发到达t号城市的最短距离，并在达到最短距离的路径中计算最少花费，同时给出相应的最短路径。

**输入**

第一行四个整数n、m、s、t（$1 \le n \le 100,0 \le m \le \frac{n(n-1)}2, 0 \le s \le n -1, 0 \le t \le n-1$​），分别表示顶点数、边数、起始编号、终点编号；

接下来m行，每行四个整数u、v、w1、w2（$0 \le u \le n-1,0 \le v \le n-1, u \ne v, 1 \le w_1 \le 100, 1 \le w_2 \le 100$），表示一条边的两个端点的编号及边权距离、边权花费。数据保证不会有重边。

**输出**

```text
min_d min_c v_1->v_2->...->v_k
```

按上面的格式输出满足题意的最短距离（即`mid_d`）、达到最短路径的路径中的最少花费（即`min_c`）、相应的最短路径（`v_1`、`v_2`、...、`v_k`表示从起点到终点的最短路径上的顶点编号）。

数据保证这样的路径存在且唯一。

样例1

输入

```
4 5 0 2
0 1 2 1
0 2 5 1
0 3 1 2
1 2 1 6
3 2 2 3
```

输出

```
3 5 0->3->2
```

解释

对应的无向图如下图所示，其中边上的第一个数字为边权距离，第二个数字为边权花费。

共有`3`条从`0`号顶点到`2`号顶点的路径：

1. `0->3->2`：距离为`3`，花费为`5`；
2. `0->2`：距离为`5`，花费为`1`；
3. `0->1->2`：距离为`3`，花费为`7`。

因此最短距离为`3`，最短路径有`2`条，在这`2`条最短路径中的最少花费是`5`，对应的路径为`0->3->2`。

![最短路径-多边权.png](https://raw.githubusercontent.com/GMyhf/img/main/img/202403291932945.png)





你可以稍微修改 Dijkstra 算法，使其能够同时考虑最短距离和最少花费的路径。下面是一个修改后的示例代码：

```python
import heapq


def dijkstra(graph, n, start, end):
    # 初始化距离、花费和路径
    dist = [float('inf')] * n
    cost = [float('inf')] * n
    prev = [-1] * n  # 用于重建路径

    dist[start] = 0
    cost[start] = 0

    # 优先队列
    queue = [(0, 0, start)]  # (距离，花费，顶点)
    while queue:
        current_dist, current_cost, u = heapq.heappop(queue)

        if u == end:  # 若到达终点，则停止
            break

        # 遍历所有邻接点
        for v, d, c in graph[u]:
            temp_dist = current_dist + d
            temp_cost = current_cost + c
            # 如果找到更短的路径或花费更低的相同长度路径
            if temp_dist < dist[v] or (temp_dist == dist[v] and temp_cost < cost[v]):
                dist[v] = temp_dist
                cost[v] = temp_cost
                prev[v] = u
                heapq.heappush(queue, (temp_dist, temp_cost, v))

    # 重建路径
    if dist[end] == float('inf'):
        return "无法到达"
    else:
        path = []
        at = end
        while at != -1:
            path.append(at)
            at = prev[at]
        path = path[::-1]  # 反转路径
        return dist[end], cost[end], '->'.join(map(str, path))


def solve(n, m, s, t, edges):
    graph = [[] for _ in range(n)]
    # 构建图
    for u, v, w1, w2 in edges:
        graph[u].append((v, w1, w2))
        graph[v].append((u, w1, w2))

    # 使用Dijkstra算法
    min_distance, min_cost, shortest_path = dijkstra(graph, n, s, t)
    return f"{min_distance} {min_cost} {shortest_path}"


# 输入
n, m, s, t = map(int, input().split())
edges = []
for _ in range(m):
    u, v, w1, w2 = map(int, input().split())
    edges.append((u, v, w1, w2))

# 解决问题
print(solve(n, m, s, t, edges))

```

这段代码首先定义了一个修改后的 `dijkstra` 函数，该函数除了记录最短距离外，还记录了达到每个城市的最少花费。然后，在 `main` 函数中，它接受输入，构建图，并调用 `dijkstra` 函数来找到最短路径和相应的最少花费，最后输出结果。





### sy392: 最短路径-多路径 中等

https://sunnywhy.com/sfbj/10/4/392

现有一个共n个顶点（代表城市）、m条边（代表道路）的无向图（假设顶点编号为从`0`到`n-1`），每条边有各自的边权，代表两个城市之间的距离。求从s号城市出发到达t号城市的最短路径条数和最短路径（可能有多条）。

**输入**

第一行四个整数n、m、s、t（$1 \le n \le 100,0 \le m \le \frac{n(n-1)}2, 0 \le s \le n -1, 0 \le t \le n-1$​），分别表示顶点数、边数、起始编号、终点编号；

接下来m行，每行三个整数u、v、w（$0 \le u \le n-1,0 \le v \le n-1, u \ne v, 1 \le w \le 100$），表示一条边的两个端点的编号及边权距离。数据保证不会有重边。

**输出**

第一行输出一个整数，表示最短路径的条数；

接下来若干行，每行一条最短路径，格式如下：

```text
v_1->v_2->...->v_k
```

其中`v_1`、`v_2`、...、`v_k`表示从起点到终点的最短路径上的顶点编号。

注意，路径的输出顺序满足基于整数的字典序，即如果有两条路径`a[1]->a[2]->...->a[k]->a[k+1]->...`与`b[1]->b[2]->...->b[k]->b[k+1]->...`，满足`a[1]==b[1]`、`a[2]==b[2]`、...、`a[k]==b[k]`、`a[k+1]<b[k+1]`，那么把路径`a`优先输出。

样例1

输入

```
4 5 0 2
0 1 2
0 2 5
0 3 1
1 2 1
3 2 2
```

输出

```
2
0->1->2
0->3->2
```

解释

对应的无向图如下图所示。

共有`3`条从`0`号顶点到`2`号顶点的路径：

1. `0->3->2`：距离为`3`；
2. `0->2`：距离为`5`；
3. `0->1->2`：距离为`3`。

因此最短距离为`3`，最短路径为`0->3->2`和`0->1->2`。

![最短路径条数.png](https://raw.githubusercontent.com/GMyhf/img/main/img/c88685d8-84ac-45dd-8cca-d1450fac9a71.png)



要解决这个问题，我们可以使用一个稍微修改的Dijkstra算法来找出所有最短路径，同时记录每个节点的前驱节点列表以便于最后重建路径。这种方法可以确保我们找到所有的最短路径，并且能够在完成搜索后重构路径。此外，我们需要记录到达每个节点的最短路径数目。

以下是步骤的详细说明：

1. **初始化**：对于图中的每个节点，我们维护一个到达该节点的最短路径长度、到达该节点的最短路径数目，以及一个从该节点回溯到起点的前驱节点列表。
2. **Dijkstra算法**：使用优先队列（最小堆）来选择当前未处理节点中距离起点最近的节点，进行松弛操作。不同于标准Dijkstra，当找到一条更短的路径到达某节点时，我们需要更新到达该节点的最短路径数目，并清空前驱节点列表，因为找到了一条更短的路径。如果找到的是一条等长的路径，我们则添加这条路径的最后一个节点到前驱节点列表中，并增加到达该节点的最短路径数目。
3. **重建路径**：从目标节点开始，使用前驱节点列表回溯至起点，这样可以找到所有的最短路径。这一步需要用到递归或队列的方法来实现。
4. **输出路径**：根据题目要求，我们需要按照特定的顺序输出所有找到的最短路径。可以通过递归回溯生成所有路径后，对路径进行排序，以满足输出要求。

下面是实现这一算法的Python代码：

```python
from heapq import heappush, heappop
from collections import defaultdict

def dijkstra(graph, n, s, t):
    dist = [float('inf')] * n
    dist[s] = 0
    prev = defaultdict(list)
    queue = [(0, s)]
    path_count = [0] * n
    path_count[s] = 1

    while queue:
        d, u = heappop(queue)
        if d > dist[u]:
            continue
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heappush(queue, (dist[v], v))
                prev[v] = [u]
                path_count[v] = path_count[u]
            elif dist[u] + w == dist[v]:
                prev[v].append(u)
                path_count[v] += path_count[u]

    return dist, prev, path_count

def find_paths(prev, t):
    def dfs(node):
        if node == s:
            return [[s]]
        paths = []
        for pred in prev[node]:
            for path in dfs(pred):
                paths.append(path + [node])
        return paths

    return dfs(t)

def solve(n, m, s, t, edges):
    graph = defaultdict(list)
    for u, v, w in edges:
        graph[u].append((v, w))
        graph[v].append((u, w))

    dist, prev, path_count = dijkstra(graph, n, s, t)
    paths = find_paths(prev, t)
    sorted_paths = sorted(paths, key=lambda x: [int(i) for i in x])

    print(path_count[t])
    for path in sorted_paths:
        print("->".join(map(str, path)))

# 输入示例
n, m, s, t = map(int, input().split())
edges = []
for _ in range(m):
    edges.append(tuple(map(int, input().split())))


solve(n, m, s, t, edges)

```

这段代码定义了`dijkstra`函数来执行Dijkstra算法并计算最短路径的数量，`find_paths`函数来找到所有最短路径，最后的`solve`函数来处理输入数据和格式化输出结果。请注意，这个解决方案可能需要根据实际数据的不同进行调整以确保性能。





### sy393: 最短距离-多起点多终点 中等

https://sunnywhy.com/sfbj/10/4/393

现有一个共n个顶点（代表城市）、m条边（代表道路）的无向图（假设顶点编号为从`0`到`n-1`），每条边有各自的边权，代表两个城市之间的距离。求从所有城市出发到达所有城市的所有最短距离。

**输入**

第一行两个整数n、m（$1 \le n \le 50,0 \le m \le \frac{n(n-1)}2$​），分别表示顶点数、边数；

接下来m行，每行三个整数u、v、w（$0 \le u \le n-1,0 \le v \le n-1, u \ne v, 1 \le w \le 100$），表示一条边的两个端点的编号及边权距离。数据保证不会有重边。

**输出**

输出多行多列的整数，其中i号行j号列表示从号顶点到达号顶点的最短距离。如果无法到达，那么输出`-1`。整数之间用空格隔开，行末不允许有多余的空格。

样例1

输入

```
4 5
0 1 2
0 2 5
0 3 1
1 2 1
3 2 2
```

输出

```
0 2 3 1
2 0 1 3
3 1 0 2
1 3 2 0
```

解释

对应的无向图如下图所示。

![最短路径条数.png](https://raw.githubusercontent.com/GMyhf/img/main/img/c88685d8-84ac-45dd-8cca-d1450fac9a71.png)



这是一个典型的多源最短路径问题，可以使用 Floyd-Warshall 算法来解决。下面是一个实现：

```python
def floyd_warshall(graph, n):
    dist = [[float('inf')] * n for _ in range(n)]

    # Initialize distances for direct edges
    for u in range(n):
        dist[u][u] = 0
        for v, w in graph[u]:
            dist[u][v] = w

    # Update distances using Floyd-Warshall algorithm
    for k in range(n):
        for i in range(n):
            for j in range(n):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

    return dist

def main():
    n, m = map(int, input().split())
    graph = [[] for _ in range(n)]
    for _ in range(m):
        u, v, w = map(int, input().split())
        graph[u].append((v, w))
        graph[v].append((u, w))

    distances = floyd_warshall(graph, n)
    
    for row in distances:
        print(' '.join(str(d) if d != float('inf') else '-1' for d in row))

if __name__ == "__main__":
    main()
```

这段代码首先定义了一个 `floyd_warshall` 函数来执行 Floyd-Warshall 算法。然后，在 `main` 函数中，它接受输入，构建图，并调用 `floyd_warshall` 函数来找到所有城市之间的最短距离，并输出结果。



Floyd-Warshall 时间复杂度？不能用Dijkstra完成吗？

Floyd-Warshall 算法的时间复杂度为 O(n^3)，其中 n 是顶点的数量。这是因为 Floyd-Warshall 算法通过三重循环来更新所有顶点之间的最短距离。

虽然使用 Dijkstra 算法也可以解决多源最短路径问题，但是它需要对每个顶点运行一次单源最短路径算法，因此时间复杂度为 O(n * (n + m) * logn)，其中 m 是边的数量。在稠密图（边数量接近 n^2）的情况下，这可能比 Floyd-Warshall 算法更慢。

因此，在顶点数量较小且图比较稠密时，Floyd-Warshall 算法通常更加高效，因为它的时间复杂度不依赖于边的数量。而在稀疏图或者顶点数量很大时，Dijkstra 算法可能更适合，因为它的时间复杂度相对较低。



下面是一个使用 Dijkstra 算法解决多源最短路径问题的 Python 实现：

```python
import heapq


def dijkstra(graph, start, n):
    distances = [float('inf')] * n

    for u in range(n):
        if u == start:
            distances[start] = 0
            continue
        visited = [False] * n
        pq = [(0, start)]  # priority queue，元素为 (distance, vertex)

        while pq:
            dist_u, u = heapq.heappop(pq)
            if visited[u]:
                continue
            visited[u] = True

            for v, weight in graph[u]:
                new_dist = dist_u + weight
                if new_dist < distances[v]:
                    distances[v] = new_dist
                    heapq.heappush(pq, (new_dist, v))

    return distances


def main():
    n, m = map(int, input().split())
    graph = [[] for _ in range(n)]
    for _ in range(m):
        u, v, w = map(int, input().split())
        graph[u].append((v, w))
        graph[v].append((u, w))

    all_shortest_paths = []
    for i in range(n):
        shortest_paths_from_i = dijkstra(graph, i, n)
        all_shortest_paths.append(shortest_paths_from_i)

    for row in all_shortest_paths:
        print(' '.join(str(d) if d != float('inf') else '-1' for d in row))


if __name__ == "__main__":
    main()

```

这个实现首先定义了一个 `dijkstra` 函数来执行 Dijkstra 算法，计算从给定起点到所有其他顶点的最短距离。然后，在 `main` 函数中，它接受输入，构建图，并调用 `dijkstra` 函数来找到所有城市之间的最短距离，并输出结果。





### sy394: 最短路径-多边权II 中等

https://sunnywhy.com/sfbj/10/4/394

现有一个共n个顶点（代表城市）、m条边（代表道路）的无向图（假设顶点编号为从`0`到`n-1`），每条边有各自的边权，代表两个城市之间的距离；每个顶点有各自的点权，代表城市的堵车指数。求从s号城市出发到达t号城市的最短距离与最短路径。

**输入**

第一行四个整数n、m、s、t（$1 \le n \le 100,0 \le m \le \frac{n(n-1)}2, 0 \le s \le n -1, 0 \le t \le n-1$​），分别表示顶点数、边数、起始编号、终点编号；

第二行为空格隔开的n个整数w1（$1 \le w_1 \le 100$），表示编号从`0`到`n-1`的城市的堵车指数；

接下来m行，每行三个整数u、v、w2（$0 \le u \le n-1,0 \le v \le n-1, u \ne v, 1 \le w2 \le 100$），表示一条边的两个端点的编号及边权距离。数据保证不会有重边。

**输出**

```text
min_d v_1->v_2->...->v_k
```

按上面的格式输出最短距离和最短路径，其中`min_d`表示最短距离，`v_1`、`v_2`、...、`v_k`表示从起点到终点的最短路径上的顶点编号。如果最短路径存在多条，那么输出路径上所有城市的堵车指数平均值最小的路径。数据保证这样的路径存在且唯一。

样例1

输入

```
4 5 0 2
2 3 2 1
0 1 2
0 2 5
0 3 1
1 2 1
3 2 2
```

输出

```
3 0->3->2
```

解释

对应的无向图如下图所示，其中边上的数字为边权距离，顶点旁的数字为点权堵车指数。

共有`3`条从`0`号顶点到`2`号顶点的路径：

1. `0->3->2`：距离为`3`，堵车指数平均值为$\frac{2+1+2}3=\frac53$；
2. `0->2`：距离为`5`，堵车指数平均值为$\frac{2+2}2=2$；
3. `0->1->2`：距离为`3`，堵车指数平均值为$\frac{2+3+2}3=\frac73$。

因此最短距离为`3`，最小堵车指数平均值的最短路径为`0->3->2`。

<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/202403292212947.png" alt="image-20240329221157501" style="zoom:50%;" />





这里是一个Dijkstra算法加DFS（深度优先搜索）来解决最短路径问题，并在路径长度相同的情况下优先选择平均堵车指数最小的路径。主要逻辑是先使用Dijkstra算法找出所有最短路径，然后通过DFS在这些路径中找到平均堵车指数最小的路径。

```python
from collections import defaultdict

INF = float('inf')

class Edge:
    def __init__(self, v, dis):
        self.v = v
        self.dis = dis

def dijkstra(n, s, G, weight):
    d = [INF] * n
    vis = [False] * n
    pre = [[] for _ in range(n)]
    d[s] = 0

    for _ in range(n):
        u, minDis = -1, INF
        for j in range(n):
            if not vis[j] and d[j] < minDis:
                u, minDis = j, d[j]
        if u == -1:
            break
        vis[u] = True
        for edge in G[u]:
            v, dis = edge.v, edge.dis
            if not vis[v]:
                if d[u] + dis < d[v]:
                    d[v] = d[u] + dis
                    pre[v] = [u]
                elif d[u] + dis == d[v]:
                    pre[v].append(u)
    return d, pre

def DFS(v, s, pre, weight, tempPath, optPath, optValue):
    if v == s:
        tempPath.append(v)
        weightSum = sum(weight[node] for node in tempPath)
        weightAvg = weightSum / len(tempPath)
        if weightAvg < optValue[0]:
            optValue[0] = weightAvg
            optPath[:] = tempPath[:]
        tempPath.pop()
        return

    tempPath.append(v)
    for prev in pre[v]:
        DFS(prev, s, pre, weight, tempPath, optPath, optValue)
    tempPath.pop()

def solve(n, m, s, t, weight, edges):
    G = defaultdict(list)
    for u, v, w in edges:
        G[u].append(Edge(v, w))
        G[v].append(Edge(u, w))

    d, pre = dijkstra(n, s, G, weight)
    
    tempPath, optPath = [], []
    optValue = [INF]
    DFS(t, s, pre, weight, tempPath, optPath, optValue)
    
    optPath.reverse()
    return d[t], '->'.join(map(str, optPath))

# Example usage
n, m, s, t = map(int, input().split())
weight = list(map(int, input().split()))
edges = [tuple(map(int, input().split())) for _ in range(m)]

min_d, path = solve(n, m, s, t, weight, edges)
print(f"{min_d} {path}")
```





### sy395: 交通枢纽 中等

https://sunnywhy.com/sfbj/10/4/395

现有一个共n个顶点（代表城市）、m条边（代表道路）的**无向连通图**（假设顶点编号为从`0`到`n-1`），每条边有各自的边权，代表两个城市之间的距离。为了促进城市间交流，需要从k个备选城市中选择其中一个城市作为交通枢纽，满足从这个交通枢纽出发到达其他所有城市的最短距离之和最小。

**输入**

第一行四个整数n、m、k（$1 \le n \le 100,0 \le m \le \frac{n(n-1)}2, 0 \le k \le min(n,10)$​），分别表示顶点数、边数、备选城市数；

接下来m行，每行三个整数u、v、w（$0 \le u \le n-1,0 \le v \le n-1, u \ne v, 1 \le w \le 100$​），表示一条边的两个端点的编号及边权距离。数据保证不会有重边。

最后一行为空格隔开的k个整数，表示备选城市编号。

**输出**

输出两个整数，分别表示作为交通枢纽的城市编号、从交通枢纽出发到达其他所有城市的最短距离之和，中间用空格隔开。如果有多个满足条件的交通枢纽，那么输出城市编号最小的那个。

样例1

输入

```
4 5 2
0 1 1
0 2 5
0 3 3
1 2 2
2 3 4
0 2
```

输出

```
0 7
```

解释

对应的无向图如下图所示。

从`0`号城市到达`1`号、`2`号、`3`号城市的最短距离分别是`1`、`3`、`3`，因此最短距离之和是`7`；

从`2`号城市到达`0`号、`1`号、`3`号城市的最短距离分别是`3`、`2`、`4`，因此最短距离之和是`9`；

因此应选择`0`号城市作为交通枢纽，对应的最短距离之和为`7`。

![交通枢纽.png](https://raw.githubusercontent.com/GMyhf/img/main/img/202403292302533.png)



您可以使用 Floyd-Warshall 算法来解决这个问题。该算法能够计算图中所有顶点对之间的最短路径。对于这个问题，我们只需要找到每个备选城市作为交通枢纽时，从该城市出发到其他所有城市的最短距离之和，然后选择最小的一个。

下面是一个实现：

```python
def floyd_warshall(graph, n, candidates):
    distances = [[float('inf')] * n for _ in range(n)]

    for u in range(n):
        distances[u][u] = 0

    for u, v, w in graph:
        distances[u][v] = w
        distances[v][u] = w

    for k in range(n):
        for i in range(n):
            for j in range(n):
                distances[i][j] = min(distances[i][j], distances[i][k] + distances[k][j])

    min_total_distance = float('inf')
    chosen_hub = -1
    for hub in candidates:
        total_distance = sum(distances[hub][v] for v in range(n) if v != hub)
        if total_distance < min_total_distance:
            min_total_distance = total_distance
            chosen_hub = hub

    return chosen_hub, min_total_distance

def main():
    n, m, k = map(int, input().split())
    graph = [tuple(map(int, input().split())) for _ in range(m)]
    candidates = list(map(int, input().split()))

    hub, min_total_distance = floyd_warshall(graph, n, candidates)
    print(hub, min_total_distance)

if __name__ == "__main__":
    main()
```

您可以将您的输入和期望输出与上述代码一起运行以进行测试。





## 5 最小生成树 5题



### sy396: 最小生成树-Prim算法 简单

https://sunnywhy.com/sfbj/10/5/396

现有一个共个顶点、条边的无向图（假设顶点编号为从`0`到`n-1`），每条边有各自的边权。在图中寻找一棵树，使得这棵树包含图上所有顶点、所有边都是图上的边，且树上所有边的边权之和最小。使用Prim算法求出这个边权之和的最小值。

**输入**

第一行两个整数n、m（$1 \le n \le 100,0 \le m \le \frac{n(n-1)}2$），分别表示顶点数、边数；

接下来m行，每行三个整数u、v、w（$0 \le u \le n-1,0 \le v \le n-1, u \ne v, 1 \le w \le 100$），表示一条边的两个端点的编号及边权距离。数据保证不会有重边。

**输出**

输出一个整数，表示最小的边权之和。如果不存在这样的树，那么输出`-1`。

样例1

输入

```
4 5
0 1 3
0 2 2
0 3 3
2 3 1
1 2 1
```

输出

```
4
```

解释

对应的无向图如下图所示。加粗的部分即为最小生成树，其边权之和为1+1+2=4。

![最小生成树-Prim算法.png](https://raw.githubusercontent.com/GMyhf/img/main/img/202403292321529.png)

样例2

输入

```
3 1
0 1 1
```

输出

```
-1
```

解释

由于此图不连通，因此不存在最小生成树。



以下是使用 Prim 算法求解的 Python 代码：

```python
import heapq

def prim(graph, n):
    visited = [False] * n
    min_heap = [(0, 0)]  # (weight, vertex)
    min_spanning_tree_cost = 0

    while min_heap:
        weight, vertex = heapq.heappop(min_heap)

        if visited[vertex]:
            continue

        visited[vertex] = True
        min_spanning_tree_cost += weight

        for neighbor, neighbor_weight in graph[vertex]:
            if not visited[neighbor]:
                heapq.heappush(min_heap, (neighbor_weight, neighbor))

    return min_spanning_tree_cost if all(visited) else -1

def main():
    n, m = map(int, input().split())
    graph = [[] for _ in range(n)]

    for _ in range(m):
        u, v, w = map(int, input().split())
        graph[u].append((v, w))
        graph[v].append((u, w))

    min_spanning_tree_cost = prim(graph, n)
    print(min_spanning_tree_cost)

if __name__ == "__main__":
    main()

```





### sy397: 最小生成树-Kruskal算法 简单

https://sunnywhy.com/sfbj/10/5/397

现有一个共n个顶点、m条边的无向图（假设顶点编号为从`0`到`n-1`），每条边有各自的边权。在图中寻找一棵树，使得这棵树包含图上所有顶点、所有边都是图上的边，且树上所有边的边权之和最小。使用Kruskal算法求出这个边权之和的最小值。

**输入**

第一行两个整数n、m（$1 \le n \le 10^4,0 \le m \le 10^5$），分别表示顶点数、边数；

接下来m行，每行三个整数u、v、w（$0 \le u \le n-1,0 \le v \le n-1, u \ne v, 1 \le w \le 100$），表示一条边的两个端点的编号及边权。数据保证不会有重边。

**输出**

输出一个整数，表示最小的边权之和。如果不存在这样的树，那么输出`-1`。

样例1

输入

```
4 5
0 1 3
0 2 2
0 3 3
2 3 1
1 2 1
```

输出

```
4
```

解释

对应的无向图如下图所示。加粗的部分即为最小生成树，其边权之和为1+1+2=4。

![最小生成树-Prim算法.png](https://raw.githubusercontent.com/GMyhf/img/main/img/202403292321529.png)





Kruskal算法是一种用于寻找最小生成树的算法。它的基本思想是按照边的权值从小到大的顺序选择边，如果这条边连接的两个顶点不在同一连通分量中，则选择这条边，否则放弃这条边。重复这个过程，直到图中所有的顶点都在同一连通分量中。

在实现Kruskal算法时，我们需要使用并查集来维护图中的连通分量，以便于快速判断两个顶点是否在同一连通分量中。

以下是Kruskal算法的Python实现：

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
        px, py = self.find(x), self.find(y)
        if self.rank[px] > self.rank[py]:
            self.parent[py] = px
        else:
            self.parent[px] = py
            if self.rank[px] == self.rank[py]:
                self.rank[py] += 1

def kruskal(n, edges):
    uf = UnionFind(n)
    edges.sort(key=lambda x: x[2])
    res = 0
    for u, v, w in edges:
        if uf.find(u) != uf.find(v):
            uf.union(u, v)
            res += w
    if len(set(uf.find(i) for i in range(n))) > 1:
        return -1
    return res

n, m = map(int, input().split())
edges = []
for _ in range(m):
    u, v, w = map(int, input().split())
    edges.append((u, v, w))
print(kruskal(n, edges))
```

这段代码首先定义了一个并查集类`UnionFind`，然后定义了`kruskal`函数来实现Kruskal算法。在`kruskal`函数中，我们首先创建一个并查集实例，然后按照边的权值对边进行排序。然后遍历每一条边，如果这条边连接的两个顶点不在同一连通分量中，就将这两个顶点合并到同一连通分量中，并将这条边的权值加到结果中。最后，如果图中的所有顶点都在同一连通分量中，就返回结果，否则返回-1。



可以看到，kruskal 算法的时间复杂度主要来源于对边进行排序，因此其时间复杂度是O(ElogE)，其中E为图的边数。显然 kruskal 适合顶点数较多、边数较少的情况，这和 prim算法恰好相反。于是可以根据题目所给的数据范围来选择合适的算法，即**如果是稠密图(边多)，则用 prim 算法;如果是稀疏图(边少)，则用 kruskal 算法**。



### Sy398: 最小造路成本 中等

https://sunnywhy.com/sfbj/10/5/398

现有个城市（假设城市编号为从`0`到`n-1`），需要在城市之间建造道路（道路造好后视为双向连通）。已知不同城市间造路的成本，问至少需要多少成本可以让这个城市连通（不需要直接连通，能够间接连通即可）。

**输入**

第一行一个整数 n（$1 \le n \le 100$），表示城市数；

接下来给出 n 行 n 列的整数矩阵，其中i号行j号列表示 i 号城市和 j 号城市之间的造路成本w（当 $i=j$ 时 $w=0$，当$i \ne j$时 $1 \le w \le 100$，且 $w[i][j] == [wj][i]$）。

**输出**

输出一个整数，表示最小的造路成本。

样例1

输入

```
4
0 3 2 3
3 0 1 5
2 1 0 1
3 5 1 0
```

输出

```
4
```

解释

对应的无向图如下图所示。加粗的部分即为最小成本时所建造的道路，其成本之和为。

![最小造路成本.png](https://raw.githubusercontent.com/GMyhf/img/main/img/a48271f9-0963-461e-b32c-86a5f57d1405.png)





这是一个经典的最小生成树问题，可以使用Prim算法或者Kruskal算法来解决。这里我们使用Prim算法来解决这个问题。

Prim算法的基本思想是从一个顶点开始，每次选择一条连接已选顶点和未选顶点的最小边，直到所有的顶点都被选中。

以下是使用Prim算法解决这个问题的Python代码：

```python
def minCost(n, cost):
    visited = [False] * n
    minCost = [float('inf')] * n
    minCost[0] = 0
    res = 0

    while True:
        v = -1
        for u in range(n):
            if not visited[u] and (v == -1 or minCost[u] < minCost[v]):
                v = u
        if v == -1:
            break
        visited[v] = True
        res += minCost[v]
        for u in range(n):
            minCost[u] = min(minCost[u], cost[v][u])

    return res

n = int(input())
cost = [list(map(int, input().split())) for _ in range(n)]
print(minCost(n, cost))
```

在这段代码中，我们首先初始化所有的顶点为未访问状态，然后设置每个顶点的最小造路成本为无穷大，然后将第一个顶点的最小造路成本设置为0。然后在循环中，我们每次选择一个未访问的顶点，其造路成本最小，然后将其标记为已访问，并将其造路成本加入到总成本中，然后更新所有与该顶点相连的顶点的最小造路成本。最后，当所有的顶点都被访问后，返回总成本。





### sy399: 最大删除边权值 中等

https://sunnywhy.com/sfbj/10/5/399

现有一个共n个顶点、m条边的无向图（假设顶点编号为从`0`到`n-1`），每条边有各自的边权。从图中删掉一些边，但保持这个图依然连通。问能删掉的边的权值之和的最大值。

**输入**

第一行两个整数n、m（$1 \le n\le 100, 0 \le m \le \frac{n(n-1)}2$），分别表示顶点数、边数；

接下来m行，每行三个整数u、v、w（$0 \le u \le n-1, 0\le v \le n-1, u \ne v, 1\le w \le 100$），表示一条边的两个端点的编号及边权。数据保证不会有重边。

**输出**

输出一个整数，表示能删掉的边的权值之和的最大值。如果图不连通，那么输出`-1`。

样例1

输入

```
4 5
0 1 3
0 2 2
0 3 3
2 3 1
1 2 1
```

输出

```
6
```

解释

对应的无向图如下图所示。加粗的边是允许删去的最大边权之和的边，其边权之和为。

![最小生成树-Prim算法.png](https://raw.githubusercontent.com/GMyhf/img/main/img/202403292321529.png)

样例2

输入

```
3 1
0 1 1
```

输出

```
-1
```

解释

由于此图不连通，因此输出`-1`。





这是一个最小生成树问题的变种。我们可以先使用Kruskal算法找出最小生成树，然后将所有边的权值之和减去最小生成树的权值之和，就得到了能删掉的边的权值之和的最大值。

以下是使用Kruskal算法解决这个问题的Python代码：

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
        px, py = self.find(x), self.find(y)
        if px != py:
            if self.rank[px] > self.rank[py]:
                self.parent[py] = px
            else:
                self.parent[px] = py
                if self.rank[px] == self.rank[py]:
                    self.rank[py] += 1

def maxEdgeRemoval(n, m, edges):
    uf = UnionFind(n)
    edges.sort(key=lambda x: x[2])
    total_weight = sum(edge[2] for edge in edges)
    mst_weight = 0
    for u, v, w in edges:
        if uf.find(u) != uf.find(v):
            uf.union(u, v)
            mst_weight += w
    if len(set(uf.find(i) for i in range(n))) > 1:
        return -1
    return total_weight - mst_weight

n, m = map(int, input().split())
edges = [list(map(int, input().split())) for _ in range(m)]
print(maxEdgeRemoval(n, m, edges))
```

在这段代码中，我们首先定义了一个并查集类，用于处理图中的连通性问题。然后我们将所有的边按照权值排序，然后依次添加到最小生成树中，如果添加的边不会形成环，那么就将其添加到最小生成树中。最后，我们检查最小生成树是否包含了所有的顶点，如果没有，那么返回-1，否则返回所有边的权值之和减去最小生成树的权值之和。





### sy400: 最小连通成本 中等

https://sunnywhy.com/sfbj/10/5/400

现有个城市（假设城市编号为从`0`到`n-1`），需要在城市之间建造道路（道路造好后视为双向连通）。已知不同城市间造路的成本，并给定若干城市编号，这些城市之间互相不需要建造新的道路就已经连通，问至少需要多少成本来建造新的道路，可以让这个城市连通（不需要直接连通，能够间接连通即可）。

**输入**

第一行两个整数n、k（$1 \le n \le 100, 0 \le k \le n$），分别表示总城市数、已经连通的城市数；

接下来给出 n 行 n 列的整数矩阵，其中i号行j号列表示 i 号城市和 j 号城市之间的造路成本w（当 $i=j$ 时 $w=0$，当$i \ne j$时 $1 \le w \le 100$，且 $w[i][j] == [wj][i]$）。

最后一行共k                                                                                                                                                                                                                                                                                                                                                                                                           个整数，表示已经连通的城市编号。

**输出**

输出一个整数，表示最小的连通成本。

样例1

输入

```
4 2
0 3 2 3
3 0 1 5
2 1 0 1
3 5 1 0
0 1
```

输出

```
2
```

解释

对应的无向图如下图所示。由于`0`号和`1`号顶点已经连通，因此只需要把加粗的边建造好，即可使四个顶点连通，且成本是最小的，其成本为。

![最小连通成本.png](https://raw.githubusercontent.com/GMyhf/img/main/img/446597e3-faca-4ee2-b245-56a466c6fef3.png)



这个问题可以通过修改Prim算法来解决。我们首先将已经连通的城市标记为已访问，然后在Prim算法的循环中，我们只选择未访问的城市。这样，我们就可以找到连接所有城市的最小成本。

以下是修改后的Prim算法的Python代码：

```python
def minCost(n, k, cost, connected):
    visited = [False] * n
    minCost = [float('inf')] * n
    for city in connected:
        minCost[city] = 0
    res = 0

    while True:
        v = -1
        for u in range(n):
            if not visited[u] and (v == -1 or minCost[u] < minCost[v]):
                v = u
        if v == -1:
            break
        visited[v] = True
        res += minCost[v]
        for u in range(n):
            minCost[u] = min(minCost[u], cost[v][u])

    return res

n, k = map(int, input().split())
cost = [list(map(int, input().split())) for _ in range(n)]
connected = list(map(int, input().split()))
print(minCost(n, k, cost, connected))
```

在这段代码中，我们首先初始化所有的城市为未访问状态，然后设置每个城市的最小造路成本为无穷大，然后将已经连通的城市的最小造路成本设置为0。然后在循环中，我们每次选择一个未访问的城市，其造路成本最小，然后将其标记为已访问，并将其造路成本加入到总成本中，然后更新所有与该城市相连的城市的最小造路成本。最后，当所有的城市都被访问后，返回总成本。





## 6 拓扑排序 3题



### sy401: 拓扑排序 中等

https://sunnywhy.com/sfbj/10/6/401

现有一个共n个顶点、m条边的有向无环图（假设顶点编号为从`0`到`n-1`）。输出该图的拓扑排序序列。

注：每次有多个顶点可以选择时，总是选择编号最小的那个。

**输入**

第一行两个整数n、m（$1 \le n \le 100, 0 \le m \le n(n-1)$），分别表示顶点数、边数；

接下来m行，每行两个整数u、v（$0 \le u \le n-1, 0 \le v \le n-1, u \ne v$），表示一条边的起点和终点的编号。数据保证不会有重边。

**输出**

输出n个整数，表示拓扑排序序列。整数之间用空格隔开，行末不允许有多余的空格。

样例1

输入

```
4 5
0 1
0 2
0 3
1 2
3 2
```

输出

```
0 1 3 2
```

解释

对应的有向无环图如下图所示。由于每次选择编号最小的顶点，因此拓扑排序序列为`0 1 3 2`。

![拓扑排序.png](https://raw.githubusercontent.com/GMyhf/img/main/img/b4214671-2ee9-421c-8e1e-26c8f0de224e.png)





这个问题可以使用拓扑排序算法来解决。拓扑排序是对有向无环图的顶点的一种排序，它使得如果存在一条从顶点 u 到顶点 v 的路径，那么在排序中 u 一定在 v 的前面。可以使用深度优先搜索来实现拓扑排序。

以下是使用深度优先搜索实现拓扑排序的Python代码：

```python
from collections import defaultdict

def topoSort(n, edges):
    graph = defaultdict(list)
    indegree = [0] * n
    for u, v in edges:
        graph[u].append(v)
        indegree[v] += 1

    queue = [i for i in range(n) if indegree[i] == 0]
    queue.sort()
    result = []

    while queue:
        u = queue.pop(0)
        result.append(u)
        for v in graph[u]:
            indegree[v] -= 1
            if indegree[v] == 0:
                queue.append(v)
        queue.sort()

    return result

n, m = map(int, input().split())
edges = [list(map(int, input().split())) for _ in range(m)]
result = topoSort(n, edges)
print(*result)
```

在这段代码中，首先构建了一个图，并计算了每个顶点的入度。然后将所有入度为0的顶点添加到队列中，并对队列进行排序。然后在循环中，每次从队列中取出一个顶点，将其添加到结果中，然后遍历其所有的邻接顶点，将邻接顶点的入度减1，如果邻接顶点的入度变为0，那么就将其添加到队列中。最后，返回结果。



### sy402: 有向无环图的判定-拓扑排序 中等

https://sunnywhy.com/sfbj/10/6/402

现有一个共n个顶点、m条边的有向图（假设顶点编号为从`0`到`n-1`）。使用拓扑排序判断其是否是有向无环图，即该有向图中是否有环。

**输入**

第一行两个整数n、m（$1 \le n \le 100, 0 \le m \le n(n-1)$），分别表示顶点数、边数；

接下来m行，每行两个整数u、v（$0 \le u \le n-1, 0 \le v \le n-1, u \ne v$），表示一条边的起点和终点的编号。数据保证不会有重边。

**输出**

如果是有向无环图，那么输出`Yes`；否则输出`No`。

样例1

输入

```
4 5
0 1
0 2
0 3
1 2
3 2
```

输出

```
Yes
```

解释

对应的有向图如下图所示。该图是有向无环图。

![拓扑排序.png](https://raw.githubusercontent.com/GMyhf/img/main/img/b4214671-2ee9-421c-8e1e-26c8f0de224e.png)

样例2

输入

```
4 5
0 1
2 0
0 3
1 2
3 2
```

输出

```
No
```

解释

对应的有向图如下图所示。该图不是有向无环图。

![有向无环图的判定-拓扑排序_样例2.png](https://raw.githubusercontent.com/GMyhf/img/main/img/db11eb3b-d51f-4d1e-bc6f-a446060eac69.png)





这个问题可以通过使用拓扑排序来解决。拓扑排序是对有向无环图的顶点的一种排序，它使得如果存在一条从顶点 u 到顶点 v 的路径，那么在排序中 u 一定在 v 的前面。如果在进行拓扑排序的过程中，发现存在没有被访问的顶点，但是已经没有入度为0的顶点，那么就说明图中存在环。

以下是使用拓扑排序判断有向图是否存在环的Python代码：

```python
from collections import defaultdict

def isDAG(n, edges):
    graph = defaultdict(list)
    indegree = [0] * n
    for u, v in edges:
        graph[u].append(v)
        indegree[v] += 1

    queue = [i for i in range(n) if indegree[i] == 0]
    count = 0

    while queue:
        u = queue.pop(0)
        count += 1
        for v in graph[u]:
            indegree[v] -= 1
            if indegree[v] == 0:
                queue.append(v)

    return count == n

n, m = map(int, input().split())
edges = [list(map(int, input().split())) for _ in range(m)]
print("Yes" if isDAG(n, edges) else "No")
```

在这段代码中，我们首先构建了一个图，并计算了每个顶点的入度。然后我们将所有入度为0的顶点添加到队列中。然后在循环中，我们每次从队列中取出一个顶点，然后遍历其所有的邻接顶点，将邻接顶点的入度减1，如果邻接顶点的入度变为0，那么就将其添加到队列中。最后，我们检查访问的顶点数是否等于总的顶点数，如果等于，那么说明图是有向无环图，否则说明图中存在环。



### sy403: 先导课程 中等

https://sunnywhy.com/sfbj/10/6/403

现有n门课程（假设课程编号为从`0`到`n-1`），课程之间有依赖关系，即可能存在两门课程，必须学完其中一门才能学另一门。现在给出个依赖关系，问能否把所有课程都学完。

注：能同时学习多门课程时总是先学习编号最小的课程。

**输入**

第一行两个整数n、m（$1 \le n \le 100, 0 \le m \le n(n-1)$），分别表示顶点数、边数；

接下来m行，每行两个整数u、v（$0 \le u \le n-1, 0 \le v \le n-1, u \ne v$），表示一条边的起点和终点的编号。数据保证不会有重边。

**输出**

如果能学完所有课程，那么输出一行`Yes`，然后在第二行输出学习课程编号的顺序，编号之间用空格隔开，行末不允许有多余的空格；如果不能学完所有课程，那么输出一行`No`，然后在第二行输出不能学习的课程门数。

样例1

输入

```
4 5
0 1
0 2
0 3
1 2
3 2
```

输出

```
Yes
0 1 3 2
```

解释

对应的依赖关系如下图所示。由于每次选择编号最小的顶点，因此学习顺序为`0 1 3 2`。

![拓扑排序.png](https://raw.githubusercontent.com/GMyhf/img/main/img/b4214671-2ee9-421c-8e1e-26c8f0de224e.png)

样例2

输入

```
4 4
0 1
1 2
2 3
3 1
```

输出

```
No
3
```

解释

对应的依赖关系如下图所示。编号为`0`的课程可以直接学习；由于编号`1`、`2`、`3`的课程互相依赖，因此无法学习的课程数为`3`。

![先导课程_样例2.png](https://raw.githubusercontent.com/GMyhf/img/main/img/9579516f-e705-451f-a6d1-6d15314b8a99.png)





这个问题可以通过使用拓扑排序来解决。拓扑排序是对有向无环图的顶点的一种排序，它使得如果存在一条从顶点 u 到顶点 v 的路径，那么在排序中 u 一定在 v 的前面。如果在进行拓扑排序的过程中，发现存在没有被访问的顶点，但是已经没有入度为0的顶点，那么就说明图中存在环，也就是存在不能学习的课程。

以下是使用拓扑排序判断有向图是否存在环并输出学习课程顺序的Python代码：

```python
from collections import defaultdict

def courseSchedule(n, edges):
    graph = defaultdict(list)
    indegree = [0] * n
    for u, v in edges:
        graph[u].append(v)
        indegree[v] += 1

    queue = [i for i in range(n) if indegree[i] == 0]
    queue.sort()
    result = []

    while queue:
        u = queue.pop(0)
        result.append(u)
        for v in graph[u]:
            indegree[v] -= 1
            if indegree[v] == 0:
                queue.append(v)
        queue.sort()

    if len(result) == n:
        return "Yes", result
    else:
        return "No", n - len(result)

n, m = map(int, input().split())
edges = [list(map(int, input().split())) for _ in range(m)]
res, courses = courseSchedule(n, edges)
print(res)
if res == "Yes":
    print(*courses)
else:
    print(courses)
```

在这段代码中，我们首先构建了一个图，并计算了每个顶点的入度。然后我们将所有入度为0的顶点添加到队列中，并对队列进行排序。然后在循环中，我们每次从队列中取出一个顶点，将其添加到结果中，然后遍历其所有的邻接顶点，将邻接顶点的入度减1，如果邻接顶点的入度变为0，那么就将其添加到队列中。最后，我们检查访问的顶点数是否等于总的顶点数，如果等于，那么说明可以学完所有课程，否则说明存在不能学习的课程。





## 7 关键路径 3题



### sy404: 关键路径长度 中等

https://sunnywhy.com/sfbj/10/7/404

现有一个共n个顶点、m条边的有向图（假设顶点编号为从`0`到`n-1`）。判断该图是否是有向无环图，并把顶点视作“事件”、边视作“活动”，求关键路径长度。

**输入**

第一行两个整数n、m（$1 \le n \le 100, 0 \le m \le n(n-1)$），分别表示顶点数、边数；

接下来m行，每行两个整数u、v、w（$0 \le u \le n-1, 0 \le v \le n-1, u \ne v, 1 \le w \le 100$），表示一条边的起点和终点的编号及边权。数据保证不会有重边。

**输出**

如果不是有向无环图，那么只输出`No`；

如果是有向无环图，那么在第一行输出`Yes`，第二行输出一个整数，表示关键路径长度。

样例1

输入

```
6 7
0 1 1
0 2 2
1 3 3
1 4 4
2 4 2
3 5 2
4 5 3
```

输出

```
Yes
8
```

解释

对应的有向无环图如下图所示，加粗的边组成了关键路径，其长度为。

![关键路径长度.png](https://raw.githubusercontent.com/GMyhf/img/main/img/b355f389-786f-4795-83ea-492517c5f24c.png)





要解决这个问题，我们可以分两个步骤进行：首先判断给定的有向图是否是有向无环图（DAG），如果是的话，再使用拓扑排序结合动态规划来求解关键路径的长度。

**判断DAG**

可以通过深度优先搜索（DFS）来检测图中是否存在环。在DFS过程中，如果遇到了正在访问的节点（即该节点在当前路径上），则表示图中存在环。

**关键路径长度**

关键路径可以通过以下步骤求得：

1. **拓扑排序**：首先进行拓扑排序，确保每一个活动（边）只在其前驱事件（顶点）处理完成后才开始。
2. **动态规划求解**：在拓扑排序的基础上，可以通过动态规划来求每个节点的最早开始时间。关键路径的长度即为终点的最早开始时间。

下面是实现这个算法的Python代码：

```python
from collections import deque

def find_critical_path(n, m, edges):
    # 构建图
    graph = [[] for _ in range(n)]
    in_degree = [0] * n  # 记录每个节点的入度
    for u, v, w in edges:
        graph[u].append((v, w))
        in_degree[v] += 1
    
    # 拓扑排序
    queue = deque()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)
    order = []  # 拓扑排序的结果
    while queue:
        u = queue.popleft()
        order.append(u)
        for v, w in graph[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    # 如果拓扑排序的结果数量小于节点数，说明图中存在环
    if len(order) < n:
        return "No"
    
    # 动态规划求解关键路径长度
    dp = [0] * n  # dp[i]表示从起点到达节点i的最长路径长度
    for u in order:
        for v, w in graph[u]:
            dp[v] = max(dp[v], dp[u] + w)
    
    return "Yes\n" + str(max(dp))

if __name__ == "__main__":
    n, m = map(int, input().split())
    edges = [tuple(map(int, input().split())) for _ in range(m)]
    result = find_critical_path(n, m, edges)
    print(result)
```

这段代码首先读入图的顶点数、边数以及每条边的信息，然后通过上述步骤判断图是否是DAG，并计算关键路径长度。如果图不是DAG，则输出"No"；如果是DAG，则输出"Yes"以及关键路径长度。





### sy405: 关键活动 中等

https://sunnywhy.com/sfbj/10/7/405

现有一个共n个顶点、m条边的有向图（假设顶点编号为从`0`到`n-1`）。判断该图是否是有向无环图，并把顶点视作“事件”、边视作“活动”，求出所有的关键活动。

**输入**

第一行两个整数n、m（$1 \le n \le 100, 0 \le m \le n(n-1)$），分别表示顶点数、边数；

接下来m行，每行两个整数u、v、w（$0 \le u \le n-1, 0 \le v \le n-1, u \ne v, 1 \le w \le 100$），表示一条边的起点和终点的编号及边权。数据保证不会有重边。

**输出**

如果不是有向无环图，那么只输出`No`；

如果是有向无环图，那么在第一行输出`Yes`，并从第二行开始，每行输出用空格隔开的两个整数，表示关键活动的起点编号和终点编号（注：总是优先输出起点编号较小的关键活动，起点编号相同时优先输出终点编号较小的关键活动）。

样例1

输入

```
6 7
0 1 1
0 2 2
1 3 3
1 4 4
2 4 2
3 5 2
4 5 3
```

输出

```
Yes
0 1
1 4
4 5
```

解释

对应的有向无环图如下图所示，加粗的边即为关键活动。

![关键路径长度.png](https://raw.githubusercontent.com/GMyhf/img/main/img/b355f389-786f-4795-83ea-492517c5f24c.png)

样例2

输入

```
2 2
1 2 1
2 1 1
```

输出

```
No
```

解释

显然不是有向无环图。





为了解决这个问题，我们可以采用以下策略：

1. **判断有向无环图（DAG）**：通过拓扑排序来判断图是否为DAG。如果在拓扑排序过程中所有的顶点都被排序了，那么图是DAG；如果排序过程中存在顶点不能被排序（即存在环），则图不是DAG。
2. **求关键活动**：
   - 在DAG中，首先通过拓扑排序计算每个顶点的最早开始时间（`est`）。
   - 然后，逆向遍历拓扑排序，计算每个顶点的最晚开始时间（`lst`）。
   - 边`u -> v`是关键活动，当且仅当`est[u] + w(u, v) == lst[v]`。

以下是实现上述逻辑的Python代码：

```python
from collections import deque, defaultdict

def topo_sort(graph, in_degrees, n):
    queue = deque([i for i in range(n) if in_degrees[i] == 0])
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for adj, weight in graph[node]:
            in_degrees[adj] -= 1
            if in_degrees[adj] == 0:
                queue.append(adj)
    return topo_order if len(topo_order) == n else None

def find_critical_activities(n, m, edges):
    graph = defaultdict(list)
    in_degrees = [0] * n
    for u, v, w in edges:
        graph[u].append((v, w))
        in_degrees[v] += 1

    topo_order = topo_sort(graph, in_degrees, n)
    if topo_order is None:
        return ["No"]

    est = [0] * n
    lst = [float('inf')] * n
    for node in topo_order:
        for adj, weight in graph[node]:
            est[adj] = max(est[adj], est[node] + weight)
    
    lst[topo_order[-1]] = est[topo_order[-1]]
    for node in reversed(topo_order):
        for adj, weight in graph[node]:
            lst[node] = min(lst[node], lst[adj] - weight)
        if node not in graph:  # Leaf node
            lst[node] = min(lst[node], est[node])

    critical_activities = []
    for u, v, w in edges:
        if est[u] + w == lst[v]:
            critical_activities.append((u, v))

    critical_activities.sort()
    return ["Yes"] + ["{} {}".format(u, v) for u, v in critical_activities]

# 读取输入
n, m = map(int, input().split())
edges = [tuple(map(int, input().split())) for _ in range(m)]

# 求解关键活动
result = find_critical_activities(n, m, edges)
for line in result:
    print(line)
```

这段代码首先尝试对给定的有向图进行拓扑排序。如果成功，说明图是DAG；然后，它计算每个顶点的最早和最晚开始时间，最后找出所有满足条件的关键活动并输出。如果图不是DAG，即存在环，那么拓扑排序的结果不会包含图中的所有顶点，函数将返回“No”。



### sy406: 关键路径 中等

https://sunnywhy.com/sfbj/10/7/406

现有一个共个顶点、条边的有向图（假设顶点编号为从`0`到`n-1`）。判断该图是否是有向无环图，并把顶点视作“事件”、边视作“活动”，求出所有关键路径。

**输入**

第一行两个整数n、m（$1 \le n \le 100, 0 \le m \le n(n-1)$），分别表示顶点数、边数；

接下来m行，每行两个整数u、v、w（$0 \le u \le n-1, 0 \le v \le n-1, u \ne v, 1 \le w \le 100$），表示一条边的起点和终点的编号及边权。数据保证不会有重边。

**输出**

如果不是有向无环图，那么只输出`No`；

如果是有向无环图，那么在第一行输出`Yes`，从第二行开始每行输出一条关键路径`v[1]->v[2]->...`。

注意，路径的输出顺序满足基于整数的字典序，即如果有两条路径`a[1]->a[2]->...->a[k]->a[k+1]->...`与`b[1]->b[2]->...->b[k]->b[k+1]->...`，满足`a[1]==b[1]`、`a[2]==b[2]`、...、`a[k]==b[k]`、`a[k+1]<b[k+1]`，那么把路径`a`优先输出。

数据保证关键路径条数不超过`1000`条。

样例1

输入

```
6 7
0 1 1
0 2 3
1 3 3
1 4 4
2 4 2
3 5 2
4 5 3
```

输出

```
Yes
0->1->4->5
0->2->4->5
```

解释

对应的有向无环图如下图所示，加粗的边组成了关键路径，共两条：`0->1->4->5`、`0->2->4->5`。

![关键路径.png](https://raw.githubusercontent.com/GMyhf/img/main/img/9732e494-940b-4777-bed2-b6a82d749ebd.png)

样例2

输入

```
2 2
1 2 1
2 1 1
```

输出

```
No
```

解释

显然不是有向无环图。



```python
from collections import defaultdict, deque

class Edge:
    def __init__(self, v, w):
        self.v = v
        self.w = w

def topo_sort(n, G, in_degree):
    q = deque([i for i in range(n) if in_degree[i] == 0])
    ve = [0] * n
    topo_order = []

    while q:
        u = q.popleft()
        topo_order.append(u)
        for edge in G[u]:
            v = edge.v
            in_degree[v] -= 1
            if in_degree[v] == 0:
                q.append(v)
            if ve[u] + edge.w > ve[v]:
                ve[v] = ve[u] + edge.w

    if len(topo_order) == n:
        return ve, topo_order
    else:
        return None, None

def get_critical_path(n, G, in_degree):
    ve, topo_order = topo_sort(n, G, in_degree.copy())
    if ve is None:
        return -1, []

    maxLength = max(ve)
    vl = [maxLength] * n

    for u in reversed(topo_order):
        for edge in G[u]:
            v = edge.v
            if vl[v] - edge.w < vl[u]:
                vl[u] = vl[v] - edge.w

    activity = defaultdict(list)
    for u in G:
        for edge in G[u]:
            v = edge.v
            e, l = ve[u], vl[v] - edge.w
            if e == l:
                activity[u].append(v)

    return maxLength, activity

def print_critical_path(u, activity, in_degree, path=[]):
    path.append(u)
    if u not in activity or not activity[u]:
        print("->".join(map(str, path)))
    else:
        for v in sorted(activity[u]):
            print_critical_path(v, activity, in_degree, path.copy())
    path.pop()

# Main
n, m = map(int, input().split())
G = defaultdict(list)
in_degree = [0] * n  # Correctly define in_degree here
for _ in range(m):
    u, v, w = map(int, input().split())
    G[u].append(Edge(v, w))
    in_degree[v] += 1

maxLength, activity = get_critical_path(n, G, in_degree)
if maxLength == -1:
    print("No")
else:
    print("Yes")
    for i in range(n):
        if in_degree[i] == 0:  # Correctly check for start points
            print_critical_path(i, activity, in_degree)

```

