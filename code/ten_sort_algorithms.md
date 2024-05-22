$$(Insertion$$ $$Sort)$$

# Python十大排序算法源码

Updated 2052 GMT+8 May 22 2024

2024 spring, Complied by Hongfei Yan



Logs:

2024/5/3 增加了改进冒泡排序，增加了改进的插入排序，增加了双指针实现的快排。

2024/4/13 取自, https://blog.csdn.net/wtandyn/article/details/119577831

还未完全详细验证。算法逻辑没问题，但是具体实现不一样的话，笔试就不好做了。



## 1 前言

最近经常用到各种排序算法，但是网上的Python排序源码质量参差不齐。因此结合网上的资料和个人理解，整理了一份可直接使用的排序算法Python源码。

包括：冒泡排序，插入排序，选择排序，希尔排序，归并排序，快速排序，堆排序，计数排序，桶排序，基数排序



## 2 排序算法的选取规则

根据网上（https://www.php.cn/faq/449050.html）的排序算法选取规则：

（1）元素个数n大，排序码分布随机，稳定性不做要求 --------- 快速排序

（2）元素个数n大，内存空间允许， 要求稳定性 ------------- 归并排序

（3）元素个数n大，排序码可能正序或逆序，稳定性不做要求 --------- 堆排序、归并排序

（4）元素个数n小，排序码基本有序或随机，要求稳定性 ------------- 插入排序

（5）元素个数n小，稳定性不做要求 ------ 选择排序

（6）元素个数n小，排序码不接近逆序 ---- 插入排序

（7）冒泡排序一般很少用（时间空间成本都高）

因此，常使用的排序算法主要包括：快速排序、归并排序、堆排序。





## 3 十大排序算法的Python源码

### 3.1 冒泡排序$$(Bubble$$ $$Sort)$$

方法： 在无序区通过反复交换找到最大元素放在队首（比较次数多，交换次数多）
主要思想： 前后两两比较，大小顺序错误就交换位置

代码思路：
1. 比较相邻元素，如果前者大于后者，就交换位置。
2. 从队首到队尾，每一对相邻元素都重复上述步骤，最后一个元素为最大元素。
3. 针对前n-1个元素重复。

```python
def BubbleSort(arr):
    for i in range(len(arr) - 1):
        for j in range(len(arr) - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

if __name__ == "__main__":
    arr_in = [6, 5, 18, 2, 16, 15, 19, 13, 10, 12, 7, 9, 4, 4, 8, 1, 11, 14, 3, 20, 17, 10]
    print(arr_in)
    arr_out = BubbleSort(arr_in)
    print(arr_out)
```

时间复杂度：$O(n^2)$
空间复杂度：$O(1)$
稳定排序



改进后的冒泡排序是对原始冒泡排序的一种优化。原始冒泡排序的基本思想是依次比较相邻的两个元素，如果它们的顺序错误就交换它们，直到没有需要交换的元素为止。这样的算法效率较低，因为即使序列已经有序，它仍然需要进行多轮的比较和交换。

改进后的冒泡排序通过增加一个标志位来优化。在每一轮比较中，如果没有发生任何交换，说明序列已经有序，不需要再进行后续的比较，因此可以提前结束排序过程。

改进后的冒泡排序的伪代码如下所示：

```python
function improved_bubble_sort(array A)
    n = length(A)
    flag = true
    for i from 0 to n - 1
        flag = false
        for j from 0 to n - 1 - i
            if A[j] > A[j + 1]
                swap(A[j], A[j + 1])
                flag = true
        if flag == false
            break
```

在这个改进后的冒泡排序算法中，如果在一轮比较中没有发生任何交换，就将标志位 `flag` 设置为 `false`，并提前跳出循环，从而减少了不必要的比较次数，提高了效率。



### 3.2 选择排序$$(Selection$$ $$Sort)$$

方法：在无序区找到最小的元素放到有序区的队尾（比较次数多，交换次数少）
主要思想：水果摊挑苹果，先选出最大的，再选出次大的，直到最后。
选择是对冒泡的优化，比较一轮只交换一次数据。

代码思路：
1. 找到无序待排序列中最小的元素，和第一个元素交换位置。
2. 剩下的待排无序序列（2-n）选出最小的元素，和第二个元素交换位置。
3. 直到最后选择完成。

```python
def SelectSort(arr):
    for i in range(len(arr)):
        minIndex = i
        for j in range(i + 1, len(arr)):
            if arr[j] < arr[minIndex]:
               minIndex = j
        arr[i], arr[minIndex] = arr[minIndex], arr[i]
    return arr

if __name__ == "__main__":
    arr_in = [6, 5, 18, 2, 16, 15, 19, 13, 10, 12, 7, 9, 4, 4, 8, 1, 11, 14, 3, 20, 17, 10]
    print(arr_in)
    arr_out = SelectSort(arr_in)
    print(arr_out)
```

时间复杂度：O(n^2)
空间复杂度：O(1)
非稳定排序

### 3.3 插入排序$$(Insertion$$ $$Sort)$$

方法：把无序区的第一个元素插入到有序区的合适位置（比较次数少，交换次数多）
主要思想：扑克牌打牌时的插入思想，逐个插入到前面的有序数中。

代码思路：
1. 选择待排无序序列的第一个元素作为有序数列的第一个元素。
2. 把第2个元素到最后一个元素看做无序待排序列。
3. 依次从待排无序序列取出每一个元素，与有序序列的每个元素比较（从右向左扫描），符合条件交换元素位置。

```python
def InsertSort(arr):
    for i in range(1, len(arr)):
        for j in range(i, 0, -1):
            if arr[j] < arr[j - 1]:
                arr[j], arr[j - 1] = arr[j - 1], arr[j]
    return arr

if __name__ == "__main__":
    arr_in = [6, 5, 18, 2, 16, 15, 19, 13,

 10, 12, 7, 9, 4, 4, 8, 1, 11, 14, 3, 20, 17, 10]
    print(arr_in)
    arr_out = InsertSort(arr_in)
    print(arr_out)
```

时间复杂度：$O(n^2)$
空间复杂度：$O(1)$
稳定排序



上面代码并没有在找到正确位置后立即停止循环，而是一直循环直到内部的 for 循环完成。

改进后的插入排序应该在找到正确位置后立即停止循环。要实现这一点，可以在内部的 for 循环中添加一个判断条件来判断是否需要继续交换。如果当前元素已经大于（或等于）前一个元素，就可以停止内部的循环了。

下面是一个修正过的版本：

```python
def InsertSort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr

if __name__ == "__main__":
    arr_in = [6, 5, 18, 2, 16, 15, 19, 13, 10, 12, 7, 9, 4, 4, 8, 1, 11, 14, 3, 20, 17, 10]
    print(arr_in)
    arr_out = InsertSort(arr_in)
    print(arr_out)
```

这个版本的插入排序算法在找到正确位置后会立即停止内部的循环，从而提高了效率。



### 3.4 希尔排序$$(Shell$$ $$Sort)$$

希尔排序是插入排序的一种更高效的改进版本，其核心思想是将待排序数组分割成若干个子序列，然后对各个子序列进行插入排序，最后再对整个序列进行一次插入排序。希尔排序的关键在于选择合适的间隔序列，以保证最终的排序效率。

代码思路：
1. 选择一个增量序列t1，t2，…，tk，其中ti>tj，tk=1。
2. 按增量序列个数k，对序列进行k趟排序。
3. 每趟排序，根据对应的增量ti，将待排序序列分割成若干长度为m的子序列，分别对各子表进行直接插入排序。仅增量因子为1时，整个序列作为一个表来处理，表长度即为整个序列的长度。

```python
def ShellSort(arr):
    n = len(arr)
    gap = n // 2
    while gap > 0:
        for i in range(gap, n):
            temp = arr[i]
            j = i
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap
            arr[j] = temp
        gap //= 2
    return arr

if __name__ == "__main__":
    arr_in = [6, 5, 18, 2, 16, 15, 19, 13, 10, 12, 7, 9, 4, 4, 8, 1, 11, 14, 3, 20, 17, 10]
    print(arr_in)
    arr_out = ShellSort(arr_in)
    print(arr_out)
```

时间复杂度：O(n log^2 n)
空间复杂度：O(1)
非稳定排序



### 3.5 归并排序$$(Merge$$ $$Sort)$$

归并排序采用分治法，将待排序数组分成若干个子序列，分别进行排序，然后再合并已排序的子序列，直到整个序列都排好序为止。

代码思路：
1. 将待排序数组分成左右两个子序列，递归地对左右子序列进行归并排序。
2. 将两个已排序的子序列合并成一个有序序列。

```python
def MergeSort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = MergeSort(arr[:mid])
    right = MergeSort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

if __name__ == "__main__":
    arr_in = [6, 5, 18, 2, 16, 15, 19, 13, 10, 12, 7, 9, 4, 4, 8, 1, 11, 14, 3, 20, 17, 10]
    print(arr_in)
    arr_out = MergeSort(arr_in)
    print(arr_out)
```

时间复杂度：O(n log n)
空间复杂度：O(n)
稳定排序



### 3.6 快速排序$$(Quick$$ $$Sort)$$

快速排序是一种高效的排序算法，采用分治法的思想，通过将数组分割成较小的子数组，然后分别对子数组进行排序，最终将数组整合成有序序列。

代码思路：
1. 选择数组中的一个元素作为基准（pivot）。
2. 将数组分割成两个子数组，使得左子数组中的所有元素都小于基准，右子数组中的所有元素都大于基准。
3. 对左右子数组递归地进行快速排序。

```python
def QuickSort(arr):
    if len(arr) <= 1:
        return arr
    else:
        pivot = arr[0]  # Choose the first element as the pivot
        left = [x for x in arr[1:] if x < pivot]
        right = [x for x in arr[1:] if x >= pivot]
        return QuickSort(left) + [pivot] + QuickSort(right)

if __name__ == "__main__":
    arr_in = [6, 5, 18, 2, 16, 15, 19, 13, 10, 12, 7, 9, 4, 4, 8, 1, 11, 14, 3, 20, 17, 10]
    print(arr_in)
    arr_out = QuickSort(arr_in)
    print(arr_out)
```

时间复杂度：平均情况下为O(n log n)，最坏情况下为O(n^2)（当数组已经有序时）
空间复杂度：平均情况下为O(log n)，最坏情况下为O(n)（递归调用栈的深度）
不稳定排序



如果用双指针实现，在partition函数中用两个指针 `i` 和 `j` 的方式实现。

```python
def quicksort(arr, left, right):
    if left < right:
        partition_pos = partition(arr, left, right)
        quicksort(arr, left, partition_pos - 1)
        quicksort(arr, partition_pos + 1, right)


def partition(arr, left, right):
    i = left
    j = right - 1
    pivot = arr[right]
    while i <= j:
        while i <= right and arr[i] < pivot:
            i += 1
        while j >= left and arr[j] >= pivot:
            j -= 1
        if i < j:
            arr[i], arr[j] = arr[j], arr[i]
    if arr[i] > pivot:
        arr[i], arr[right] = arr[right], arr[i]
    return i


arr = [22, 11, 88, 66, 55, 77, 33, 44]
quicksort(arr, 0, len(arr) - 1)
print(arr)

# [11, 22, 33, 44, 55, 66, 77, 88]
```





### 3.7 堆排序$$(Heap$$ $$Sort)$$

堆排序利用了堆这种数据结构的特性，将待排序数组构建成一个二叉堆，然后对堆进行排序。

代码思路：
1. 构建一个最大堆（或最小堆），将待排序数组转换成堆。
2. 从堆顶开始，每次将堆顶元素与堆的最后一个元素交换，然后重新调整堆。
3. 重复上述步骤，直到整个堆排序完成。

```python
def heapify(arr, n, i):
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2

    if left < n and arr[left] > arr[largest]:
        largest = left
    if right < n and arr[right] > arr[largest]:
        largest = right

    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)

def HeapSort(arr):
    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(arr, i, 0)
    return arr

if __name__ == "__main__":
    arr_in = [6, 5, 18, 2, 16, 15, 19, 13, 10, 12, 7, 9, 4, 4, 8, 1, 11, 14, 3, 20, 17, 10]
    print(arr_in)
    arr_out = HeapSort(arr_in)
    print(arr_out)
```

时间复杂度：O(n log n)
空间复杂度：O(1)
不稳定排序



### 3.8 计数排序$$(Counting$$ $$Sort)$$

计数排序是一种非比较性的排序算法，适用于待排序数组的取值范围较小且已知的情况。该算法通过统计每个元素出现的次数，然后根据统计结果重构排序后的数组。

代码思路：
1. 统计数组中每个元素出现的次数，并存储在额外的计数数组中。
2. 根据计数数组中的统计结果，重构排序后的数组。

```python
def CountingSort(arr):
    max_value = max(arr)
    count = [0] * (max_value + 1)
    for num in arr:
        count[num] += 1
    sorted_arr = []
    for i in range(max_value + 1):
        sorted_arr.extend([i] * count[i])
    return sorted_arr

if __name__ == "__main__":
    arr_in = [6, 5, 18, 2, 16, 15, 19, 13, 10, 12, 7, 9, 4, 4, 8, 1, 11, 14, 3, 20, 17, 10]
    print(arr_in)
    arr_out = CountingSort(arr_in)
    print(arr_out)
```

时间复杂度：O(n + k)，其中n是数组的长度，k是数组中的最大值与最小值的差值
空间复杂度：O(n + k)



### 3.9 桶排序$$(Bucket$$ $$Sort)$$

桶排序是一种排序算法，它假设输入是由一个随机过程产生的，该过程将元素均匀、独立地分布在[0, 1)区间上。

代码思路：
1. 创建一个定量的桶数组，并初始化每个桶为空。
2. 将每个元素放入对应的桶中。
3. 对每个非空桶进行排序。
4. 从每个桶中将排序后的元素依次取出，得到排序结果。

```python
def BucketSort(arr):
    n = len(arr)
    max_val = max(arr)
    min_val = min(arr)
    bucket_size = (max_val - min_val) / n

    buckets = [[] for _ in range(n+1)]

    for num in arr:
        index = int((num - min_val) // bucket_size)
        buckets[index].append(num)

    sorted_arr = []
    for bucket in buckets:
        sorted_arr.extend(sorted(bucket))

    return sorted_arr

if __name__ == "__main__":
    arr_in = [0.897, 0.565, 0.656, 0.1234, 0.665, 0.3434]
    print(arr_in)
    arr_out = BucketSort(arr_in)
    print(arr_out)

#[0.1234, 0.3434, 0.565, 0.656, 0.665, 0.897]
```

时间复杂度：O(n + k)，其中n是数组的长度，k是桶的数量
空间复杂度：O(n)



### 3.10 基数排序$$(Radix$$ $$Sort)$$

基数排序是一种多关键字的排序算法，它将整数按位数切割成不同的数字，然后按每个位数分别比较。

代码思路：
1. 找出数组中最大值，并确定最大值的位数。
2. 使用计数排序或桶排序，根据当前位数进行排序。

```python
def RadixSort(arr):
    max_val = max(arr)
    digit = len(str(max_val))

    for i in range(digit):
        bucket = [[] for _ in range(10)]
        for num in arr:
            bucket[num // (10 ** i) % 10].append(num)
        arr = [num for sublist in bucket for num in sublist]
    return arr

if __name__ == "__main__":
    arr_in = [170, 45, 75, 90, 802, 24, 2, 66]
    print(arr_in)
    arr_out = RadixSort(arr_in)
    print(arr_out)
```

时间复杂度：O(n * k)，其中n是数组的长度，k是最大值的位数
空间复杂度：O(n + k)

