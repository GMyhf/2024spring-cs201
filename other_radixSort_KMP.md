# 202404~05-Other

Updated 2359 GMT+8 Apr 6, 2024

2024 spring, Complied by Hongfei Yan



**Logs：**





# 一、基数排序

基数排序是一种非比较型整数排序算法，其原理是将整数按位数切割成不同的数字，然后按每个位数分别比较。由于整数也可以表达字符串（比如名字或日期）和特定格式的浮点数，所以基数排序也不是只能使用于整数。



```python
def radixSort(arr):
    max_value = max(arr)
    digit = 1
    while digit <= max_value:
        temp = [[] for _ in range(10)]
        for i in arr:
            t = i // digit % 10
            temp[t].append(i)
        arr.clear()
        for bucket in temp:
            arr.extend(bucket)
        digit *= 10
    return arr

arr = [170, 45, 75, 90, 802, 24, 2, 66]
ans = radixSort(arr)
print(*ans)

# Output:
# 2 24 45 66 75 90 170 802
```



这个程序是一个实现基数排序（Radix Sort）的函数。基数排序是一种非比较型的排序算法，它根据数字的位数来对数字进行排序。

下面是对程序的解读：

1. `radixSort` 函数接受一个整数列表 `arr` 作为输入，并返回排序后的列表。
2. 在函数中，首先找出列表中的最大值 `max_value`，以确定需要排序的数字的最大位数。
3. 然后，通过 `digit` 变量来表示当前处理的位数，初始化为 1。在每次迭代中，`digit` 的值会乘以 10，以处理下一个更高位的数字。
4. 在每次迭代中，创建一个包含 10 个空列表的临时列表 `temp`，用于存储每个数字在当前位数上的分组情况。
5. 对于列表中的每个数字 `i`，计算其在当前位数上的值 `t`（通过取整除和取模操作），然后将数字 `i` 存入对应的桶中。
6. 在填充完所有桶之后，将桶中的数字按照顺序取出，重新放入原始列表 `arr` 中。这样就完成了对当前位数的排序。
7. 继续迭代，直到处理完所有位数为止。
8. 最后，返回排序后的列表 `arr`。

通过基数排序，可以有效地对整数列表进行排序，时间复杂度为 O(d * (n + k))，其中 d 是最大位数，n 是数字个数，k 是基数（这里是 10）。



**Complexity Analysis of Radix Sort**

Time Complexity:

- Radix sort is a non-comparative integer sorting algorithm that sorts data with integer keys by grouping the keys by the individual digits which share the same significant position and value. It has a time complexity of O(d \* (n + b)), where d is the number of digits, n is the number of elements, and b is the base of the number system being used.
- In practical implementations, radix sort is often faster than other comparison-based sorting algorithms, such as quicksort or merge sort, for large datasets, especially when the keys have many digits. However, its time complexity grows linearly with the number of digits, and so it is not as efficient for small datasets.

Auxiliary Space:

- Radix sort also has a space complexity of O(n + b), where n is the number of elements and b is the base of the number system. This space complexity comes from the need to create buckets for each digit value and to copy the elements back to the original array after each digit has been sorted.







# 二、KMP（Knuth-Morris-Pratt）



https://en.wikipedia.org/wiki/Knuth%E2%80%93Morris%E2%80%93Pratt_algorithm

In computer science, the **Knuth–Morris–Pratt algorithm** (or **KMP algorithm**) is a string-searching algorithm that searches for occurrences of a "word" `W` within a main "text string" `S` by employing the observation that when a mismatch occurs, the word itself embodies sufficient information to determine where the next match could begin, thus bypassing re-examination of previously matched characters.

The algorithm was conceived by James H. Morris and independently discovered by Donald Knuth "a few weeks later" from automata theory. Morris and Vaughan Pratt published a technical report in 1970. The three also published the algorithm jointly in 1977. Independently, in 1969, Matiyasevich discovered a similar algorithm, coded by a two-dimensional Turing machine, while studying a string-pattern-matching recognition problem over a binary alphabet. This was the first linear-time algorithm for string matching.



KMP Algorithm for Pattern Searching

<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/image-20231107135044605.png" alt="image-20231107135044605" style="zoom: 33%;" />



<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/image-20231107135333487.png" alt="image-20231107135333487" style="zoom:50%;" />



**Generative AI is experimental**. Info quality may vary.

The Knuth–Morris–Pratt (KMP) algorithm is **a computer science algorithm that searches for words in a text string**. The algorithm compares characters from left to right. 

When a mismatch occurs, the algorithm uses a preprocessed table called a "Prefix Table" to skip character comparisons.

How the KMP algorithm works

- The algorithm finds repeated substrings called LPS in the pattern and stores LPS information in an array.
- The algorithm compares characters from left to right.
- When a mismatch occurs, the algorithm uses a preprocessed table called a "Prefix Table" to skip character comparisons.
- The algorithm precomputes a prefix function that helps determine the number of characters to skip in the pattern whenever a mismatch occurs.
- The algorithm improves upon the brute force method by utilizing information from previous comparisons to avoid unnecessary character comparisons.

Benefits of the KMP algorithm

- The KMP algorithm efficiently helps you find a specific pattern within a large body of text.
- The KMP algorithm makes your text editing tasks quicker and more efficient.
- The KMP algorithm guarantees 100% reliability.





**Preprocessing Overview:**

- KMP algorithm preprocesses pat[] and constructs an auxiliary **lps[]** of size **m** (same as the size of the pattern) which is used to skip characters while matching.
- Name **lps** indicates the longest proper prefix which is also a suffix. A proper prefix is a prefix with a whole string not allowed. For example, prefixes of “ABC” are “”, “A”, “AB” and “ABC”. Proper prefixes are “”, “A” and “AB”. Suffixes of the string are “”, “C”, “BC”, and “ABC”. 真前缀（proper prefix）是一个串除该串自身外的其他前缀。
- We search for lps in subpatterns. More clearly we ==focus on sub-strings of patterns that are both prefix and suffix==.
- For each sub-pattern pat[0..i] where i = 0 to m-1, lps[i] stores the length of the maximum matching proper prefix which is also a suffix of the sub-pattern pat[0..i].



KMP（Knuth-Morris-Pratt）算法是一种利用双指针和动态规划的字符串匹配算法。

```python
""""
compute_lps 函数用于计算模式字符串的LPS表。LPS表是一个数组，
其中的每个元素表示模式字符串中当前位置之前的子串的最长前缀后缀的长度。
该函数使用了两个指针 length 和 i，从模式字符串的第二个字符开始遍历。
"""
def compute_lps(pattern):
    """
    计算pattern字符串的最长前缀后缀（Longest Proper Prefix which is also Suffix）表
    :param pattern: 模式字符串
    :return: lps表
    """

    m = len(pattern)
    lps = [0] * m
    length = 0
    for i in range(1, m):
        while length > 0 and pattern[i] != pattern[length]:
            length = lps[length - 1]    # 跳过前面已经比较过的部分
        if pattern[i] == pattern[length]:
            length += 1
        lps[i] = length
    return lps


def kmp_search(text, pattern):
    n = len(text)
    m = len(pattern)
    if m == 0:
        return 0
    lps = compute_lps(pattern)
    matches = []

    j = 0  # j是pattern的索引
    for i in range(n):  # i是text的索引
        while j > 0 and text[i] != pattern[j]:
            j = lps[j - 1]
        if text[i] == pattern[j]:
            j += 1
        if j == m:
            matches.append(i - j + 1)
            j = lps[j - 1]
    return matches


text = "ABABABABCABABABABCABABABABC"
pattern = "ABABCABAB"
index = kmp_search(text, pattern)
print("pos matched：", index)
# pos matched： [4, 13]


```



# 三、字典与检索

## 06640: 倒排索引

http://cs101.openjudge.cn/2024sp_routine/06640/

给定一些文档，要求求出某些单词的倒排表。

对于一个单词，它的倒排表的内容为出现这个单词的文档编号。

**输入**

第一行包含一个数N，1 <= N <= 1000，表示文档数。
接下来N行，每行第一个数ci，表示第i个文档的单词数。接下来跟着ci个用空格隔开的单词，表示第i个文档包含的单词。文档从1开始编号。1 <= ci <= 100。
接下来一行包含一个数M，1 <= M <= 1000，表示查询数。
接下来M行，每行包含一个单词，表示需要输出倒排表的单词。
每个单词全部由小写字母组成，长度不会超过256个字符，大多数不会超过10个字符。

**输出**

对于每一个进行查询的单词，输出它的倒排表，文档编号按从小到大排序。
如果倒排表为空，输出"NOT FOUND"。

样例输入

```
3
2 hello world
4 the world is great
2 great news
4
hello
world
great
pku
```

样例输出

```
1
1 2
2 3
NOT FOUND
```



要实现一个程序来创建和查询倒排索引，可以使用 字典结构来高效地完成任务。以下是具体的步骤：

1. 首先，解析输入，为每个单词构建倒排索引，即记录每个单词出现在哪些文档中。
2. 使用字典存储倒排索引，其中键为单词，值为一个有序列表，列表中包含出现该单词的文档编号。
3. 对于每个查询，检查字典中是否存在该单词，如果存在，则返回升序文档编号列表；如果不存在，则返回 "NOT FOUND"。

```python
def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()

    n = int(data[0])
    index = 1
    inverted_index = {}   # 构建倒排索引
    for i in range(1, n + 1):
        parts = data[index].split()
        doc_id = i
        num_words = int(parts[0])
        words = parts[1:num_words + 1]
        for word in words:
            if word not in inverted_index:
                inverted_index[word] = []
            inverted_index[word].append(doc_id)
        index += 1

    m = int(data[index])
    index += 1
    results = []

    # 查询倒排索引
    for _ in range(m):
        query = data[index]
        index += 1
        if query in inverted_index:
            results.append(" ".join(map(str, sorted(inverted_index[query]))))
        else:
            results.append("NOT FOUND")

    # 输出查询结果
    for result in results:
        print(result)

if __name__ == "__main__":
    main()
```



## 04093: 倒排索引查询

http://cs101.openjudge.cn/practice/04093/

现在已经对一些文档求出了倒排索引，对于一些词得出了这些词在哪些文档中出现的列表。

要求对于倒排索引实现一些简单的查询，即查询某些词同时出现，或者有些词出现有些词不出现的文档有哪些。

**输入**

第一行包含一个数N，1 <= N <= 100，表示倒排索引表的数目。
接下来N行，每行第一个数ci，表示这个词出现在了多少个文档中。接下来跟着ci个数，表示出现在的文档编号，编号不一定有序。1 <= ci <= 1000，文档编号为32位整数。
接下来一行包含一个数M，1 <= M <= 100，表示查询的数目。
接下来M行每行N个数，每个数表示这个词要不要出现，1表示出现，-1表示不出现，0表示无所谓。数据保证每行至少出现一个1。

**输出**

共M行，每行对应一个查询。输出查询到的文档编号，按照编号升序输出。
如果查不到任何文档，输出"NOT FOUND"。

样例输入

```
3
3 1 2 3
1 2
1 3
3
1 1 1
1 -1 0
1 -1 -1
```

样例输出

```
NOT FOUND
1 3
1
```



在实际搜索引擎在处理基于倒排索引的查询时，搜索引擎确实会优先关注各个查询词的倒排表的合并和交集处理，而不是直接准备未出现文档的集合。这种方法更有效，特别是在处理大规模数据集时，因为它允许系统动态地调整和优化查询过程，特别是在有复杂查询逻辑（如多个词的组合、词的排除等）时。详细解释一下搜索引擎如何使用倒排索引来处理查询：

倒排索引查询的核心概念

1. 倒排索引结构：
   - 对于每个词（token），都有一个关联的文档列表，这个列表通常是按文档编号排序的。
   - 每个文档在列表中可能还会有附加信息，如词频、位置信息等。
2. 处理查询：
   - 单词查询：对于单个词的查询，搜索引擎直接返回该词的倒排列表。
   - 多词交集查询：对于包含多个词的查询，搜索引擎找到每个词的倒排列表，然后计算这些列表的交集。
     这个交集代表了所有查询词都出现的文档集合。
   - 复杂逻辑处理：对于包含逻辑运算（AND, OR, NOT）的查询，搜索引擎会结合使用集合的
     交集（AND）、并集（OR）和差集（NOT）操作来处理查询。特别是在处理 NOT 逻辑时，
     它并不是去查找那些未出现词的文档集合，而是从已经确定的结果集中排除含有这个词的文档。

更贴近实际搜索引擎的处理实现，如下：

```python
import sys
input = sys.stdin.read
data = input().split()

index = 0
N = int(data[index])
index += 1

word_documents = []

# 读取每个词的倒排索引
for _ in range(N):
    ci = int(data[index])
    index += 1
    documents = sorted(map(int, data[index:index + ci]))
    index += ci
    word_documents.append(documents)

M = int(data[index])
index += 1

results = []

# 处理每个查询
for _ in range(M):
    query = list(map(int, data[index:index + N]))
    index += N

    # 集合存储各词的文档集合（使用交集获取所有词都出现的文档）
    included_docs = []
    excluded_docs = set()

    # 解析查询条件
    for i in range(N):
        if query[i] == 1:
            included_docs.append(word_documents[i])
        elif query[i] == -1:
            excluded_docs.update(word_documents[i])

    # 仅在有包含词时计算交集
    if included_docs:
        result_set = set(included_docs[0])
        for docs in included_docs[1:]:
            result_set.intersection_update(docs)
        result_set.difference_update(excluded_docs)
        final_docs = sorted(result_set)
        results.append(" ".join(map(str, final_docs)) if final_docs else "NOT FOUND")
    else:
        results.append("NOT FOUND")

# 输出所有查询结果
for result in results:
    print(result)
```



# 四、B-trees

2-3 树、2-3-4 树、B 树和 B+ 树

**2-3 Tree**:

- A 2-3 tree is a type of balanced search tree where each node can have either 2 or 3 children.
- In a 2-3 tree:
  - Every internal node has either 2 children and 1 data element, or 3 children and 2 data elements.
  - The leaves are all at the same level.
- Insertions and deletions in a 2-3 tree may cause tree restructuring to maintain the balance.

**2-3-4 Tree**:

- A 2-3-4 tree is a generalization of the 2-3 tree where nodes can have either 2, 3, or 4 children.
- In a 2-3-4 tree:
  - Every internal node has either 2, 3, or 4 children and 1, 2, or 3 data elements, respectively.
  - The leaves are all at the same level.
- Like the 2-3 tree, insertions and deletions may cause restructuring to maintain balance.

**B-Tree**:

- A B-tree is a self-balancing tree data structure that maintains sorted data and allows for efficient search, insertion, and deletion operations.
- In a B-tree:
  - Each node contains multiple keys and pointers to child nodes.
  - Nodes can have a variable number of keys within a certain range, determined by the order of the B-tree.
  - B-trees are balanced and ensure that all leaves are at the same level.
- B-trees are commonly used in databases and file systems for their ability to handle large amounts of data efficiently.

**B+ Tree**:

- A B+ tree is a variation of the B-tree with additional features optimized for disk storage systems.
- In a B+ tree:
  - Data entries are stored only in leaf nodes.
  - Internal nodes store keys and pointers to child nodes but do not store actual data.
  - Leaf nodes are linked together in a linked list, making range queries efficient.
- B+ trees are commonly used in database systems because of their efficiency in disk-based storage and their ability to handle range queries effectively.

These tree structures are fundamental in computer science and are widely used in various applications where efficient data storage and retrieval are essential.





Here's a brief tutorial on B-trees and B+ trees, along with Python implementations:

### B-Tree Tutorial:

1. **Introduction**:
   - A B-tree is a self-balancing tree data structure that maintains sorted data and allows for efficient search, insertion, and deletion operations.
   - Each node contains multiple keys and pointers to child nodes.
   - Nodes can have a variable number of keys within a certain range, determined by the order of the B-tree.
   - B-trees are balanced and ensure that all leaves are at the same level.

2. **Operations**:
   - **Search**: Starts from the root and recursively searches down the tree to find the target key.
   - **Insertion**: Starts from the root and recursively inserts the key into the appropriate leaf node. If the leaf node is full, it may split, and the median key is pushed up to the parent node.
   - **Deletion**: Starts from the root and recursively deletes the key from the appropriate leaf node. If deletion causes underflow, nodes may merge or borrow keys from neighboring nodes to maintain balance.

### B-Tree Python Implementation:

B树是一种自平衡的树，主要用于系统中大量数据的存储。它可以保持数据排序，并允许进行高效的插入、删除和查找操作。

B树的主要特性如下：

1. 所有叶子节点都在同一层。
2. 每个节点中的元素从小到大排列，节点当中k个元素正好是k+1个孩子指针的分界点。
3. 非叶子节点的孩子节点数目介于 t 和 2t 之间，其中 t 是树的阶数。每个非根节点至少有 t-1 个关键字，至多有 2t-1 个关键字。
4. 根节点至少有两个孩子，除非它是一个叶节点（即树中只有一个节点）。

以下是一个简单的B树的Python实现：

```python
class BTreeNode:
    def __init__(self, leaf=False):
        self.leaf = leaf
        self.keys = []
        self.child = []

class BTree:
    def __init__(self, t):
        self.root = BTreeNode(True)
        self.t = t

    def insert(self, k):
        root = self.root
        if len(root.keys) == (2*self.t) - 1:
            temp = BTreeNode()
            self.root = temp
            temp.child.insert(0, root)
            self.split_child(temp, 0)
            self.insert_non_full(temp, k)
        else:
            self.insert_non_full(root, k)

    def insert_non_full(self, x, k):
        i = len(x.keys) - 1
        if x.leaf:
            x.keys.append((None, None))
            while i >= 0 and k < x.keys[i]:
                x.keys[i+1] = x.keys[i]
                i -= 1
            x.keys[i+1] = k
        else:
            while i >= 0 and k < x.keys[i]:
                i -= 1
            i += 1
            if len(x.child[i].keys) == (2*self.t) - 1:
                self.split_child(x, i)
                if k > x.keys[i]:
                    i += 1
            self.insert_non_full(x.child[i], k)

    def split_child(self, x, i):
        t = self.t
        y = x.child[i]
        z = BTreeNode(y.leaf)
        x.child.insert(i+1, z)
        x.keys.insert(i, y.keys[t-1])
        z.keys = y.keys[t: (2*t) - 1]
        y.keys = y.keys[0: t-1]
        if not y.leaf:
            z.child = y.child[t: 2*t]
            y.child = y.child[0: t-1]

# 创建一个阶数为3的B树
btree = BTree(3)

# 插入一些键
keys = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for key in keys:
    btree.insert(key)

# 打印B树的根节点的键
print(btree.root.keys)
#output: [(3, 6)] 
```

这个代码实现了B树的基本操作，包括插入和分裂子节点。但是，它并没有实现删除操作，这是一个更复杂的问题，需要更多的代码来处理各种情况。





### B+ Tree Tutorial:

1. **Introduction**:
   - A B+ tree is a variation of the B-tree with additional features optimized for disk storage systems.
   - Data entries are stored only in leaf nodes.
   - Internal nodes store keys and pointers to child nodes but do not store actual data.
   - Leaf nodes are linked together in a linked list, making range queries efficient.

2. **Operations**:
   - Operations are similar to B-trees but are optimized for disk access patterns, making them suitable for database systems.

### 





# 参考

Python数据结构与算法分析(第2版)，布拉德利·米勒 戴维·拉努姆/吕能,刁寿钧译，出版时间:2019-09

Brad Miller and David Ranum, Problem Solving with Algorithms and Data Structures using Python, https://runestone.academy/ns/books/published/pythonds/index.html



https://github.com/wesleyjtann/Problem-Solving-with-Algorithms-and-Data-Structures-Using-Python



数据结构（C语言版 第2版） (严蔚敏)
