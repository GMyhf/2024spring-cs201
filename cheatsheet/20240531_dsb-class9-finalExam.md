# 2024春-数据结构与算法B-9班



2024-05-31 13:00 ~ 15:00

http://dsaex.openjudge.cn/2024final/ 



题目挺友好的，6个题目，gpt能做5个。



## 28348: 单链表

http://dsaex.openjudge.cn/2024final/A/

实现一个单链表，链表初始为空，支持三种操作：

1. 向链表头插入一个数；
2. 删除第 k 个插入的数后面的一个数；
3. 在第 k 个插入的数后插入一个数。

现在要对该链表进行 M 次操作，进行完所有操作后，从头到尾输出整个链表。

输入

第一行为整数M，表示操作次数。
接下来M行，每行包含一个操作命令，操作命令可能为以下几种：

1. H x，表示向链表头插入一个数x。
2. D k，表示删除第k个插入的数后面的数（当k为0时，表示删除头节点）。
3. I k x，表示在第k个插入的数后面插入一个数x（此操作k均大于0）。

输出

共一行，将整个链表从头到尾输出。

样例输入

```
10
H 9
I 1 1
D 1
D 0
H 6
I 3 6
I 4 5
I 4 5
I 3 4
D 6
```

样例输出

```
6 4 6 5
```

提示

1≤M≤100000
插入的数x为非负整数且均不超过100
所有操作保证合法。

来源

acwing



```c
#include <stdio.h>
#include <stdlib.h>

typedef struct ListNode {
    int val;
    struct ListNode *next;
} ListNode;

typedef struct {
    ListNode *head;
    ListNode **inserts;
    int insertCount;
} LinkedList;

LinkedList* createLinkedList(int maxSize) {
    LinkedList *list = (LinkedList*) malloc(sizeof(LinkedList));
    list->head = NULL;
    list->inserts = (ListNode**) malloc(sizeof(ListNode*) * maxSize);
    list->insertCount = 0;
    return list;
}

void insertAtHead(LinkedList *list, int x) {
    ListNode *newNode = (ListNode*) malloc(sizeof(ListNode));
    newNode->val = x;
    newNode->next = list->head;
    list->head = newNode;
    list->inserts[list->insertCount++] = newNode;
}

void deleteAfterK(LinkedList *list, int k) {
    if (k == 0 && list->head != NULL) {
        ListNode *toDelete = list->head;
        list->head = list->head->next;
        free(toDelete);
    } else if (k > 0 && k <= list->insertCount) {
        ListNode *target = list->inserts[k - 1];
        if (target->next != NULL) {
            ListNode *toDelete = target->next;
            target->next = toDelete->next;
            free(toDelete);
        }
    }
}

void insertAfterK(LinkedList *list, int k, int x) {
    if (k > 0 && k <= list->insertCount) {
        ListNode *newNode = (ListNode*) malloc(sizeof(ListNode));
        newNode->val = x;
        ListNode *target = list->inserts[k - 1];
        newNode->next = target->next;
        target->next = newNode;
        list->inserts[list->insertCount++] = newNode;
    }
}

void printList(LinkedList *list) {
    ListNode *curr = list->head;
    while (curr != NULL) {
        printf("%d ", curr->val);
        curr = curr->next;
    }
    printf("\n");
}

void freeList(LinkedList *list) {
    ListNode *curr = list->head;
    while (curr != NULL) {
        ListNode *next = curr->next;
        free(curr);
        curr = next;
    }
    free(list->inserts);
    free(list);
}

int main() {
    int M;
    scanf("%d", &M);
    LinkedList *list = createLinkedList(M);

    for (int i = 0; i < M; ++i) {
        char op;
        int x, k;
        scanf(" %c", &op);
        if (op == 'H') {
            scanf("%d", &x);
            insertAtHead(list, x);
        } else if (op == 'D') {
            scanf("%d", &k);
            deleteAfterK(list, k);
        } else if (op == 'I') {
            scanf("%d %d", &k, &x);
            insertAfterK(list, k, x);
        }
    }

    printList(list);
    freeList(list);

    return 0;
}

```





## 28374: 机器翻译

http://dsaex.openjudge.cn/2024final/B/

小晨的电脑上安装了一个机器翻译软件，他经常用这个软件来翻译英语文章。

这个翻译软件的原理很简单，它只是从头到尾，依次将每个英文单词用对应的中文含义来替换。

对于每个英文单词，软件会先在内存中查找这个单词的中文含义，如果内存中有，软件就会用它进行翻译；如果内存中没有，软件就会在外存中的词典内查找，查出单词的中文含义然后翻译，并将这个单词和译义放入内存，以备后续的查找和翻译。 

假设内存中有 M 个单元，每单元能存放一个单词和译义。

每当软件将一个新单词存入内存前，如果当前内存中已存入的单词数不超过 M−1，软件会将新单词存入一个未使用的内存单元；若内存中已存入 M 个单词，软件会清空最早进入内存的那个单词，腾出单元来，存放新单词。 

假设一篇英语文章的长度为 N 个单词。

给定这篇待译文章，翻译软件需要去外存查找多少次词典？

假设在翻译开始前，内存中没有任何单词。

输入

输入文件共2行，每行中两个数之间用一个空格隔开。 
第一行为两个正整数M和N，代表内存容量和文章的长度。 
第二行为N个非负整数，按照文章的顺序，每个数（大小不超过1000）代表一个英文单词。
文章中两个单词是同一个单词，当且仅当它们对应的非负整数相同。

输出

输出文件共1行，包含一个整数，为软件需要查词典的次数。

样例输入

```
3 7
1 2 1 5 4 4 1
```

样例输出

```
5
```

提示

1≤M≤100
1≤N≤1000

来源

acwing



```c++
#include <iostream>
#include <unordered_set>
#include <queue>
#include <vector>

using namespace std;

int main() {
    int M, N;
    cin >> M >> N;
    
    vector<int> words(N);
    for (int i = 0; i < N; ++i) {
        cin >> words[i];
    }
    
    unordered_set<int> memory; // To store the words in memory
    queue<int> order;          // To maintain the order of insertion in memory
    int lookup_count = 0;      // Number of dictionary lookups
    
    for (int word : words) {
        if (memory.find(word) == memory.end()) { // Word not in memory
            lookup_count++;
            if (memory.size() >= M) {
                int oldest = order.front();
                order.pop();
                memory.erase(oldest);
            }
            memory.insert(word);
            order.push(word);
        }
    }
    
    cout << lookup_count << endl;
    return 0;
}

```





## 28375: 信息加密

http://dsaex.openjudge.cn/2024final/C/

在传输信息的过程中，为了保证信息的安全，我们需要对原信息进行加密处理，形成加密信息，从而使得信息内容不会被监听者窃取。

现在给定一个字符串，对其进行加密处理。

加密的规则如下：

1. 字符串中的小写字母，a">a 加密为 b">b，b">b 加密为 c">c，…，y">y 加密为 z">z，z">z 加密为 a">a。
2. 字符串中的大写字母，A">A 加密为 B">B，B">B 加密为 C">C，…，Y">Y 加密为 Z">Z，Z">Z 加密为 A">A。
3. 字符串中的其他字符，不作处理。

请你输出加密后的字符串。

输入

共一行，包含一个字符串。注意字符串中可能包含空格。

输出

输出加密后的字符串。

样例输入

```
Hello! How are you!
```

样例输出

```
Ifmmp! Ipx bsf zpv!
```

提示

输入字符串的长度不超过100000。

来源

acwing



```c++
#include <iostream>
#include <string>

using namespace std;

string encrypt(const string& input) {
    string result;
    for (char c : input) {
        if (c >= 'a' && c <= 'z') {
            result += (c == 'z') ? 'a' : c + 1;
        } else if (c >= 'A' && c <= 'Z') {
            result += (c == 'Z') ? 'A' : c + 1;
        } else {
            result += c;
        }
    }
    return result;
}

int main() {
    string input;
    getline(cin, input);

    string encrypted_string = encrypt(input);
    cout << encrypted_string << endl;

    return 0;
}

```



## 28360: 艾尔文的探险

http://dsaex.openjudge.cn/2024final/D/

在遥远的Teyvat大陆上，有一位聪明但有些古怪的学者，名叫艾尔文。他酷爱研究各种复杂的谜题和数学问题。一天，他听闻了一个神秘的传说，说在遥远的森林深处有一座神秘的神殿，守护着一卷古老的卷轴。据说，这卷卷轴蕴含着巨大的财富，但要解开其中的秘密，需要解决一个复杂的问题。

传说中，神殿里的卷轴上写满了由‘（’和‘）’两种符号组成的文字，隐藏着一个巨大的谜题。谜题的核心是寻找其中最长的格式正确的括号子串。这项任务看似简单，但实际上极为艰巨，因为括号的数量之多令人望而生畏。

艾尔文听闻这个传说，兴奋不已，决心前往挑战。然而，当他终于找到神殿，展开卷轴时，眼前的景象让他大吃一惊。卷轴上密密麻麻的括号让他眼花缭乱，算力不足，他无法一眼看清其中的奥秘。

无奈之下，艾尔文只能无功而返，但他留下了这个问题，希望有更有智慧的人能够解开这个古老的谜题。

现在，你挑战这个问题吧：给出一个仅包含‘（’和‘）’的字符串，计算出其中最长的格式正确的括号子串的长度。

输入

一个仅包含‘（’和‘）’的字符串。

输出

计算出其中最长的格式正确的括号子串的长度。

样例输入

```
(())(()
```

样例输出

```
4
```

提示

字符串的长度最大为2*10^6



```python
#include<bits/stdc++.h>
using namespace std;

int main() {
    string s;
    cin >> s;
    stack<int> stk;
    stk.push(-1);  // 初始化栈，压入-1作为哨兵
    int maxLen = 0;
    for(int i = 0; i < s.size(); i++) {
        if(s[i] == '(') {
            stk.push(i);  // 左括号，索引入栈
        } else {
            stk.pop();  // 右括号，弹出栈顶元素
            if(stk.empty()) {
                stk.push(i);  // 如果栈为空，将当前位置入栈
            } else {
                maxLen = max(maxLen, i - stk.top());  // 计算最大长度
            }
        }
    }
    cout << maxLen << endl;
    return 0;
}
```





## 28361: 能量果实

http://dsaex.openjudge.cn/2024final/E/

在一个宁静的小镇边缘，有一座神秘的古老庄园。这座庄园已经存在了数百年，传说中它的每一块砖石都蕴含着无数的故事。庄园的主人是一位老园丁，他精心呵护着一棵与众不同的古树。这棵古树庞大无比，树上的每一个分枝都是老园丁用心栽培的结果。

这棵古树有着复杂的结构，每个节点都代表着树上的一个分枝，而每个分枝上都结着果实。这些果实不仅味美多汁，每一个果实还有一个特殊的数值，这些数值代表着果实所蕴含的神秘能量。老园丁知道，这些能量可以用来帮助小镇上的人们解决各种难题。然而，果实的采摘并不是那么简单，因为古树有一个古老的禁忌：任何时候采摘果实时，不能同时摘取具有父子关系的两个果实，否则树的魔力将会消失。

为了从这棵古树上获得最大的能量，老园丁决定在小镇上举办一个竞赛，邀请各方高手前来参与。参与者需要设计一个方案，选择一些果实，使得它们的能量总和最大，同时遵守禁忌。庄园门前贴出了这样一份通知：

**欢迎来到古老庄园的能量果实采摘竞赛！**
**竞赛要求**

1. 树的结构：古树的根节点编号为1。树的总结点数为n，每个节点都有一个独特的编号（范围在1~n之间）和一个非负整数值，代表着该节点上的果实所蕴含的能量。
2. 古树的特点：这是一棵 二叉树，每个节点最多有两个孩子。
3. 禁忌规则：为了保护古树的魔力，采摘的果实不能同时来自父子关系的两个节点。
4. 目标：设计一个算法，找出一个最佳的果实采摘方案，使得采摘的果实能量总和最大。

现在，你是这场果实采摘大赛的参与者，请求出果实能量总和的最大值。

输入

第一行：一个整数 n，表示树上的节点个数。

接下来的 n 行，每行三个整数 vi, li, ri，分别表示第 i 号节点的果实能量值、左孩子节点编号和右孩子节点编号。如果某个孩子节点不存在，则用0表示。

输出

一个整数，表示最大能量总和。

样例输入

```
5
6 2 3
2 4 5
3 0 0
4 0 0
5 0 0
```

样例输出

```
15
```

提示

1 ≤ n ≤ 2*10^5, 0 ≤ vi ≤ 5000



典型的树形动态规划(Tree DP)问题。在这个问题中,我们有一棵二叉树,每个节点都有一个值。我们需要找到一种方案,使得我们选择的节点值的总和最大,并且不能选择相邻的节点。

这个问题可以使用动态规划来解决,具体思路如下:

定义 dp[i][0] 表示不选择节点 i 的最大值,dp[i][1] 表示选择节点 i 的最大值。
使用深度优先搜索(DFS)遍历整棵树,并计算每个节点的 `dp[i][0] `和 `dp[i][1]`。
对于每个节点 i:
`dp[i][0] `等于它的左子树和右子树不选择时的最大值,即 `max(dp[left][0], dp[left][1]) + max(dp[right][0], dp[right][1])`;
`dp[i][1] `等于它的值加上它的左子树和右子树都不选择时的最大值,即 `tree[i].value + dp[left][0] + dp[right][0]`。
最终,答案就是 `max(dp[1][0], dp[1][1])`,其中 1 表示根节点。
这个解决方案的时间复杂度是 O(n),其中 n 是树的节点数。空间复杂度是 O(n),因为我们需要存储每个节点的 dp 值。

这个问题是一个经典的树形 DP 问题,通常会出现在面试和算法竞赛中。掌握这种解决方法对于理解和解决更复杂的树形 DP 问题很有帮助。

```c++
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

struct TreeNode {
    int value;
    int left;
    int right;
};

void dfs(int node, const vector<TreeNode>& tree, vector<vector<int>>& dp) {
    if (node == 0) return;

    int left = tree[node].left;
    int right = tree[node].right;

    dfs(left, tree, dp);
    dfs(right, tree, dp);

    dp[node][0] = max(dp[left][0], dp[left][1]) + max(dp[right][0], dp[right][1]);
    dp[node][1] = tree[node].value + dp[left][0] + dp[right][0];
}

int main() {
    int n;
    cin >> n;

    vector<TreeNode> tree(n + 1);
    for (int i = 1; i <= n; ++i) {
        cin >> tree[i].value >> tree[i].left >> tree[i].right;
    }

    vector<vector<int>> dp(n + 1, vector<int>(2, 0));
    dfs(1, tree, dp);

    cout << max(dp[1][0], dp[1][1]) << endl;

    return 0;
}

```





## 28362: 魔法森林

http://dsaex.openjudge.cn/2024final/F/

在古老的幻境国度，有一片传说中的幽秘森林。这片森林里生长着无数的魔法树，每一棵树上都栖息着爱搞恶作剧的精灵。整片森林被神秘的单向魔法路径缠绕，形成了一张扑朔迷离的网络。这些路径只能单向通行，无法逆行。

传说中，森林里的魔法树共有 N 棵，它们被奇异的编号标记，从 1 到 N。森林中存在 M 条单向魔法路径，每条路径将两棵树连接起来。

某天，森林里的精灵们突发奇想，决定玩一个令人费解的游戏：每个精灵都要找到从自己栖息的魔法树出发，沿着单向的魔法路径，最终能够到达的编号最小的魔法树。传说在这棵编号最小的魔法树上，隐藏着一个无尽智慧的宝藏。

精灵们的聪明才智似乎到达了极限，于是它们向外界发布了一个挑战，希望有智慧的冒险者们能够解开这个谜题。你，作为一个足智多谋的冒险者，决定接受这个挑战。

你的任务是：对于每一棵魔法树 v，找出从它出发，沿着单向的魔法路径，能够到达的编号最小的魔法树。你需要编写一个程序来实现这一点，并帮助精灵们找到它们的梦想之树。



输入

第一行包含两个整数 N 和 M，分别表示魔法树的数量和魔法路径的数量。

接下来的 M 行，每行包含两个整数 u 和 v，表示存在一条从魔法树 u 到魔法树 v 的单向魔法路径。

输出

用空格隔开的 N 个数，第 i 个数表示从第 i 棵魔法树出发，能够到达的编号最小的魔法树。

样例输入

```
5 4
1 3
3 4
4 5
4 2
```

样例输出

```
1 2 2 2 5
```

提示

1 ≤ N, M ≤ 10^5



```c++

```



