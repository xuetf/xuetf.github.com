---
title: Leetcode (Week 2)
date: 2017-09-24 09:55:32
tags: [基础算法,Leetcode]
categories: 算法设计与分析
---
　本文是Leetcode刷题计划第二周的总结。
<!--more-->
# Maximum Subarray(\#53)
　　问：找到一个最大和的连续子数组。
　　分析：这道题有O(n)的解法，也有O(nlogn)的解法。先看O(n)动态规划的方法。我们想找到一个子数组A[i..j]使得sum(A[i..j])最大化。分解原问题的规模，如果只有1个数的话，那么直接返回。如果2个数的话，我们已经知道了1个数情况下的最大和，那么2个数最大和肯定是max(A[0], A[1], A[0]+A[1]), A[0]对应的是1个数的最大和，上式可改成max(A[0], max(A[1], A[0]+A[1])), 也就是说如果A[0]>0的话，显然最大和就是A[0]+A[1]。
　　拓展来说，已知A[1..j-1]数组的最大子数组，那么求A[1..j]的最大子数组，实际上只有两种情况。
- A[1..j]的最大子数组就是A[1..j-1]的最大子数组，即最大子数组不包含A[j]该元素。
- A[1..j]的最大子数组包含A[j]这个元素，即最大子数组此时以A[j]作为末端元素，我们需要找到起始端点A[i],其中0<=i<=j。
　　凭直觉而言，对于上述第二种情况，我们可以从A[j]开始往左遍历一直到起始元素，不断求和和比较，找到以A[j]结尾最大的子数组。但是这样会花费额外的时间。实际上我们发现这里面存在大量重复的操作，例如以A[j+1]结尾的子数组势必会和以A[j]结尾的子数组，存在重复操作，都要往左一直遍历到起点。因此我们实际上可以定义一个变量MaxEndHere来记录以A[j]结尾的最大子数组。则MaxEndHere_{j+1} = Max (MaxEndHere_{j}+A[j+1], A[j+1]);
　　这样再定义一个maxSoFar, 则maxSoFar = Max(maxSoFar, maxEndHere);maxSoFar对应第一种情况，maxEndHere对应第二种情况。
```c++
int maxSubArray(vector<int>& nums) {
    if(nums.empty()) return 0;
    
    int maxSoFar = nums[0];//记录[1..i]当前最大子数组
    int maxEndHere = nums[0];//记录以i为结尾端点最大的子数组
    for(int i = 1; i < nums.size();++i){
        maxEndHere = max(maxEndHere + nums[i], nums[i]);//实际上maxEndHere>0即可
        maxSoFar = max(maxSoFar, maxEndHere);//[1..i]最大子数组，要么是[1..i-1]的最大子数组maxSofar, 要么是包含i这个端点，往左数到开始位置最大连续和maxEndHere
    }
    return maxSoFar;
            
}
```
　　第二种方法是分治法，最大子数组只可能是如下几个情况：
- A[i..j]在A的左半边数组出现,其中 low <= i <= j <= mid
- A[i..j]在A的右半边数组出现,其中, mid < i <= j <= high
- A[i..j]跨越左右半边，即 low <= i <= mid < j <= high (j严格大于mid) 
　　前两种情况对应原问题的子问题，使用递归求解。第三种情况是合并问题，实际上就是从mid开始，以mid为结尾端点往左求最大子数组，以mid为开始端点往右求最大子数组，方法同第一种方法里求maxEndHere最原始的方法。
```c++
int maxSubArray(vector<int>& nums) {
    return dq_maxSubArray(nums, 0, nums.size() - 1);
}

int dq_maxSubArray(vector<int>& nums, int low, int high){
    if(low == high) return nums[low];
    int mid = low + (high-low)/2;
    int left_max = dq_maxSubArray(nums, low, mid);
    int right_max = dq_maxSubArray(nums, mid+1, high);
    int cross_max = findCrossMaxSubArray(nums, low, mid, high);
    return max(max(left_max,right_max), cross_max);
}

int findCrossMaxSubArray(vector<int>& nums, int low, int mid, int high){// low <= i <= mid < j <= high
    int left_sum = INT_MIN, right_sum = INT_MIN, temp_sum = 0;
    //left
    int j = mid;
    while(j >= low){
        temp_sum += nums[j];
        left_sum = max(left_sum, temp_sum);
        --j;
    }
    temp_sum = 0;
    j = mid+1;
    while(j <= high){
        temp_sum += nums[j];
        right_sum = max(right_sum, temp_sum);
        ++j;
    }
    return left_sum+right_sum;
}
```

# Best Time To Buy and Sell Stock(\#121)
　　问题：给一个股价序列，要求选择买入价和卖出价，使得盈利最大。
　　分析：可以借助上面求连续和最大的子数组的方法。对于给定的股价数组\\([a_0,a_1,a_2,...,a_n]\\),构造一个数组\\(a_1-a_0, a_2-a_1, a_3-a_2,...,a_n-a_{n-1}\\), 原数组任意两个数相减，可以转化成新数组求该范围内的连续和。
　　例如:对于原数组\\([a_0,a_1,a_2,a_3]\\),构造新数组\\([a_1-a_0,a_2-a_1,a_3-a_2]\\)，则比如\\(a_2-a_0=(a_2-a_1) + (a_1-a_0)\\)等等。
```c++
int maxProfit(vector<int>& prices) {
    if(prices.empty()) return 0;
    int pre = prices[0];//前一个数
    for(int i = 1; i < prices.size(); ++i){
        int t = prices[i] - pre;//先求差
        pre = prices[i];//保存当前
        prices[i] = t;
    }
    int m =  maxSubArray(prices);
    return m > 0? m:0;
}

int maxSubArray(vector<int>& nums) {//[1..nums.size() )
    int maxSoFar = nums[1], maxEndHere = nums[1];
    for(int i = 2; i < nums.size(); ++i){
        maxEndHere = max(maxEndHere + nums[i], nums[i]);
        maxSoFar = max(maxSoFar, maxEndHere);
    }
    return maxSoFar;
}
```
　　还有另一种dp的方法，考虑数组A[1...j]最大差值和A[1..j-1]的关系，显然A[1..j]最大差值可以分成两种情况：
- 不在A[j]处卖出，则A[1..j]的最大差值就是A[1..j-1]数组的最大差值。
- 在A[j]处卖出，这样就需要找到A[1..j]中的最小值。此时最大差值为A[j]-min.
　　代码如下：
```c++
int maxProfit(vector<int> &prices){
    int max_profit = 0, min_price = INT_MAX;
    for(int i = 0; i < prices.size(); ++i){
        min_price = min(min_price, prices[i]);
        max_profit = max(max_profit, prices[i] - min_price);
    }
    return max_profit;
}
```

# Minimum Absolute Difference in BST(\#530)
　　问题:给定一个二叉搜索树，找到任意两个差值的绝对值最小的节点，并返回最小值。
　　分析：假定给的是一个数组，并且已经从小到大排序。显然这个数组差值最小的两个数肯定在所有相邻数对中产生，即\\(min\\{|A[i+1]-A[i]|\\}\\)。同样，由于给定的是排序二叉树，如果能将该二叉树排序，则就可以得出解。但是实际上没必要排完序后，再遍历。可以采用中序遍历的方法，使用pre变量保存前一个值，定义全局min_abs保存最小差值。每次遍历当前点时，都比较min_abs和root->val - pre的大小即可。
```c++
int min_dif = INT_MAX;
int prev = -1; //记录前一个值，按从小到大排列时的前一个值
int getMinimumDifference(TreeNode* root) {
    if(root == NULL) return min_dif;
    getMinimumDifference(root->left);//遍历左边
    if(prev != -1){//第一次到最左下角的点时候不要比较
        min_dif = min(min_dif, root->val - prev);
    }
    prev = root->val;//赋值成当前点
    getMinimumDifference(root->right);
    return min_dif;
}
```

# First Unique Character in a String(\#387)
　　问:找到字符串第一个unique的字符。
　　分析：使用hash表先统计每个字符的个数。然后遍历一遍原字符串，看哪个字符优先只出现一次，则直接返回结果。
```c++
int firstUniqChar(string s) {
    unordered_map<char,int> map;
    for(auto &c : s) {
        m[c]++;
    }
    for(int i = 0; i < s.size(); i++) {
        if(m[s[i]] == 1) return i;
    }
    return -1;
}
```

# Delete Node in a Linked List(\#237)
　　问：给定一个链表中的一个节点，删除该节点。
　　答：正常想法是直接把后面的往前挪一个位置。但是我们无法拿到该节点的前一个节点，因此无法链接到后面。此处采用指针的方式，将后一个节点的地址里的内容直接赋值给当前节点指向的地址单元。“\*指针”在赋值符号右侧的时候，会取得指针指向的地址的内容，“\*指针”在赋值符号左侧的时候，会指向该指针指向的地址单元。
```c++
void deleteNode(ListNode* node) {
    ListNode* t = node->next;
    *node = *(node->next);//node指向node->next指向的内容
    delete t; //t和node内容完全一样
}
```

# Same Tree(\#100)
　　比较两棵二叉树是否结构内容一致。
```c++
bool isSameTree(TreeNode* p, TreeNode* q) {
    if(p == NULL && q == NULL) return true;
    if(p == NULL && q != NULL) return false;
    if(p != NULL && q == NULL) return false;
    if(p->val != q->val) return false;
    return isSameTree(p->left,q->left) && isSameTree(p->right, q->right);
}
```
# Minimum Index Sum of Two Lists(\#599)
　　问题：给定两个存放字符串的数组，找到某个字符串使得该字符串同时在两个数组出现，并且下标之和最小。
　　分析：可以使用hash，先存储list1，hash[list1[i]]=i。然后遍历list2,如果在hash中找到hash[list2[j]]说明该字符串同时存在两个数组中，定义min为当前最小下标，定义sum=hash[list2[j]]+j。如果sum<min,则清空结果数组，放入list2[j],如果sum=min，则不清空数组，放入list2[j]。

```c++
vector<string> findRestaurant(vector<string>& list1, vector<string>& list2) {
    unordered_map<string,int> map;
    vector<string> result;
    for(int i = 0; i < list1.size(); ++i){
        map[list1[i]] = i;
    }
    int min = INT_MAX;
    for(int i = 0; i < list2.size(); ++i){
        if(map.find(list2[i]) != map.end()){//存在相同的才进行考虑
            int sum = map[list2[i]]+i;
            if(sum < min){
                min = sum;
                result.clear();
                result.push_back(list2[i]);
            }else if(sum == min) {
                result.push_back(list2[i]);
            }   
        }
    }
    return result;
}
```

# Image Smoother(\#661)
　　问题：将图像中某个像素点值设置成该像素点周围9个(加上自身)像素点值得平均值。
　　分析：分拆问题，首先需要一个smooth方法来设置某个像素点[i,j]为周围所有9个点像素点平局值。smooth中获取周围9个点，可以构造位移数组[-1,0,1]，[i,j]的行下标i和列下标j的移动范围在位移数组中，两个循环遍历即可，注意要检测边界。最后对原图像二维数组每个位置调用smooth即可。
```c++
int dir[3] = {0, -1, 1};
    /**
    * 将单元格的值替换成相邻9个(包括自己)的单元格的平均值
    */
    vector<vector<int>> imageSmoother(vector<vector<int>>& M) {
        vector<vector<int>> result;
        if(M.empty()) return result;
        int row = M.size(), column = M[0].size();
        result.resize(row, vector<int> (column));//记住该方法
        for(int r = 0; r < row; ++r){
            for(int c = 0; c < column; ++c){
                smooth(M, row, column, r, c, result);
            }
        }
        return result;
    }
    
    void smooth(vector<vector<int>>& M, int row, int col, int r, int c, vector<vector<int>>& result){
        int count = 0;
        int sum = 0;
        for(int i = 0; i < 3; ++i){
            for(int j = 0; j < 3; ++j){
                if (is_valid(row, col, r, c, dir[i], dir[j])){
                    ++count;
                    sum += M[r+dir[i]][c+dir[j]];
                }
            }
        }
        result[r][c] = sum / count;
    }
    
    
    bool is_valid(int row, int col, int r, int c, int r_inc, int c_inc){
        return r + r_inc >= 0 && r + r_inc < row && c + c_inc >= 0 && c + c_inc < col;
    }

```

# Ransom Note(\#383)
　　问题：给定一个note和一个字符串集合magazine，note只能由magazine中的字符构成，并且magazine中字符是有限的。例如canConstruct("aa", "ab")->false,canConstruct("aa", "aab") -> true。
　　分析：先统计magazine每个字符数量，再遍历note，不断扣除，如果不存在或者扣除到0，则返回false。
```c++
bool canConstruct(string ransomNote, string magazine) {
    unordered_map<char,int> map;
    for(char c : magazine){
        if(map.find(c) != map.end()){
            map[c] += 1;
        }
        else{
            map[c] = 1;
        }
    }
    for(char c : ransomNote){
        if(map.find(c) == map.end() || map[c] <= 0) return false;
        else{
            map[c] -= 1;
        }
    }
    return true;
}
```