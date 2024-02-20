# Assignment #1: 拉齐大家Python水平

Updated 0940 GMT+8 Feb 19, 2024

2024 spring, Complied by ==同学的姓名、院系==



**说明：**

1）数算课程的先修课是计概，由于计概学习中可能使用了不同的编程语言，而数算课程要求Python语言，因此第一周作业练习Python编程。如果有同学坚持使用C/C++，也可以，但是建议也要会Python语言。

2）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

3）课程网站是Canvas平台, https://pku.instructure.com, 学校通知3月1日导入选课名单后启用。**作业写好后，保留在自己手中，待3月1日提交。**

提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

4）如果不能在截止前提交作业，请写明原因。



**编程环境**

==（请改为同学的操作系统、编程环境等）==

操作系统：macOS Ventura 13.4.1 (c)

Python编程环境：Spyder IDE 5.2.2, PyCharm 2023.1.4 (Professional Edition)

C/C++编程环境：Mac terminal vi (version 9.0.1424), g++/gcc (Apple clang version 14.0.3, clang-1403.0.22.14.1)



## 1. 题目

### 20742: 泰波拿契數

http://cs101.openjudge.cn/practice/20742/



思路：a,b,c=b,c,a+b+c



##### 代码

```python
#
n = int(input())
a,b,c = 0,1,1
for i in range(n):
    a,b,c = b,c,a+b+c
print(a)
```



代码运行截图 ==（至少包含有"Accepted"）==

1min完成

![image](https://github.com/GMyhf/2024spring-cs201/assets/160590370/024124c2-034a-482c-bf31-e0ae231245bd)





### 58A. Chat room

greedy/strings, 1000, http://codeforces.com/problemset/problem/58/A



思路：递归？



##### 代码

```python
# 
def func(s,x):
    for i in range(len(s)):
        if s[i]==con[x]:
            if x<=3:
                return func(s[i+1::],x+1)
            if x==4:
                return 1
    return 0
s=input()
con='hello'
if func(s,0):
    print('YES')
else:
    print('NO')
```



代码运行截图 ==（至少包含有"Accepted"）==

5min完成

![image](https://github.com/GMyhf/2024spring-cs201/assets/160590370/afed7479-ee48-418a-86a3-3ccaaf0b79b5)






### 118A. String Task

implementation/strings, 1000, http://codeforces.com/problemset/problem/118/A



思路：直接分拆就行



##### 代码

```python
# 
vowels=['a','e','i','o','u','y']
s=input()
s=s.lower()
ans=''
for i in range(len(s)):
    if s[i] not in vowels:
        ans=ans+s[i]
for i in range(len(ans)):
    print('.'+ans[i],end="")
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

3min完成

![image](https://github.com/GMyhf/2024spring-cs201/assets/160590370/8b9d163b-9b4d-4d36-82ec-42f882b8276b)




### 22359: Goldbach Conjecture

http://cs101.openjudge.cn/practice/22359/



思路：枚举全体分拆



##### 代码

```python
# 
def isprime(n):
    if n==1:
        return False
    if n==2 or n==3:
        return True
    k=2
    while k*k<=n:
        if n%k==0:
            return False
        k+=1
    return True
n=int(input())
if n>=2:
    for i in range(1,n//2,1):
        if isprime(i) and isprime(n-i):
            print(str(i)+' '+str(n-i))
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

4min完成

![image](https://github.com/GMyhf/2024spring-cs201/assets/160590370/4283c4fc-f56c-4b7d-951f-dafd876757b3)




### 23563: 多项式时间复杂度

http://cs101.openjudge.cn/practice/23563/



思路：字符串操作。注意n前面系数为1的时候要单独讨论



##### 代码

```python
# 
lst=input().split('+')
ans=[]
for s in lst:
    i=s.index('n')
    if i==0 or int(s[0:i:1])!=0:
        ans.append(int(s[i+2:len(s):1]))
print("n^"+str(max(ans)))
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

9min完成

![image](https://github.com/GMyhf/2024spring-cs201/assets/160590370/a3e38b99-cee1-4b96-9eb5-4426c656a97e)




### 24684: 直播计票

http://cs101.openjudge.cn/practice/24684/



思路：用字典进行查询，join控制输出格式



##### 代码

```python
# 
lst=list(map(int,input().split()))
s=set(lst)
ans=[]
dic={num:lst.count(num) for num in s}
m=max(dic.values())
for key in dic.keys():
    if dic[key]==m:
        ans.append(key)
ans.sort()
ans1=[str(k) for k in ans]
print(' '.join(ans1))
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

10min完成

![image](https://github.com/GMyhf/2024spring-cs201/assets/160590370/e6d6d9e4-f716-4e6c-a9ec-5aef1d4d7eed)





## 2. 学习总结和收获

==如果作业题目简单，有否额外练习题目，比如：OJ“数算pre每日选做”、CF、LeetCode、洛谷等网站题目。==

题目难度适中，适合唤醒秋季学期的python基础编程知识。



