# 202404~05-Other

Updated 1515 GMT+8 Apr 4, 2024

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













# 参考

Python数据结构与算法分析(第2版)，布拉德利·米勒 戴维·拉努姆/吕能,刁寿钧译，出版时间:2019-09

Brad Miller and David Ranum, Problem Solving with Algorithms and Data Structures using Python, https://runestone.academy/ns/books/published/pythonds/index.html



https://github.com/wesleyjtann/Problem-Solving-with-Algorithms-and-Data-Structures-Using-Python



数据结构（C语言版 第2版） (严蔚敏)
