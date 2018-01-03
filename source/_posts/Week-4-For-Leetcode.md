---
title: Week-4-For-Leetcode
date: 2017-10-13 19:49:56
tags: [基础算法,Leetcode]
categories: 算法设计与分析
---
　　Leetcode第四周总结, 还包括一些课堂作业。
<!--more-->
# Count of Smaller Numbers After Self(\#315)
　　问：给定一个序列，求每个数右边比它小的数的个数。
　　答：类似逆序数定义，逆序数是定义某个数左边比该数大的数。使用merge sort。思路见代码注释，注意保存交换完后数据的原始位置。
```c++
struct Node {
    int value;
    int loc;
    Node(int v, int l) : value(v), loc(l){}
    Node(){}
};
class Solution {
public:
    /**
    * 1)求右边有多少数比左边小
    * 2)还有一种题型是求左边有多少数比右边大
    * 唯一区别是从1)必须从右往左求。2)必须从左往右求。 则在merge的时候,1）先放最大的元素，左右两边都已经求出逆序数个数了，merge的时候，只有mid左边的元素的逆序数会受到影响，因此需要求左边元素的逆序数个数变化情况。 2）先放最小的元素。左右两边都已经求出逆序数个数了，merge的时候，只有mid右边的元素的逆序数会受到影响，因此需要求左边元素的逆序数个数变化情况。两者都是使用c初始化为另一边的总个数,1)左边元素比右边大的时候，更新逆序数+=c,否则c--;2)右边元素比左边小时,更新逆序数+=c,否则c--
    */
    vector<int> countSmaller(vector<int>& nums) {
        int n = nums.size();
        vector<Node> node_nums(n);//辅助
        for (int i = 0; i < nums.size(); ++i)
            node_nums[i] = Node(nums[i], i);//记录下标位置，很关键
        vector<int> count(n);
        merge_sort_count(node_nums, 0, n-1, count);
        return count;
    }


    void merge_sort_count(vector<Node>& nums, int low, int high, vector<int> &count) {
        if (low >= high) return;
        int mid = low + (high - low) / 2;
        merge_sort_count(nums, mid + 1, high, count);
        merge_sort_count(nums, low, mid, count);
        merge(nums, low, mid, high, count);
    }


    void merge(vector<Node>& nums, int low, int mid, int high, vector<int> &count) {
        vector<Node> left(mid - low + 1);
        vector<Node> right(high - mid);
        int k1 = 0, k2 = 0;
        for (int i = low; i <= mid; ++i)
            left[k1++] = nums[i];
        for (int i = mid + 1; i <= high; ++i)
            right[k2++] = nums[i];
        
        int c = k2;//记录右边元素个数
        int p1 = --k1, p2 = --k2;//指向最后一个元素

        for (int i = high; i >= low; --i) {
            if (p1 < 0) nums[i] = right[p2--];
            else if (p2 < 0) 
                nums[i] = left[p1--]; //count[left[p1]] += 0;
            else {
                if (left[p1].value > right[p2].value) {//左边大
                    nums[i] = left[p1];
                    count[left[p1].loc] += c;//左边大的时候统计,loc一定要保存
                    --p1;
                }
                else {//右边大
                    nums[i] = right[p2--];
                    --c;
                }
            }
        }
    }
};
```

# closest pair
　　问：求最近点对。
　　答：分治算法。原数据按照x坐标大小排序得到一个序列px，按照y坐标大小排序得到一个序列py。每次按照x大小从中间切开，左右两边分别递归求最近点对,得到左右两边最小距离d。然后合并两个子问题。令切分点的x坐标为\\(x_m\\), 左边划分出区域\\(p_1\\), 满足\\(x_m - x_{p1} <= d\\)。划分出区域\\(p_2\\),满足\\(x_{p2} - x_m <= d 且 y_{p1}-d<=y_{p2} <= y_{p1}+d\\)。具体实施时，对于\\(P_1\\)区域的每个点，按照序列py进行遍历，内循环对\\(p_2\\)每个点进行遍历，因为是按照y的顺序排列的，首先按照y过滤\\(y_{p2}< y_{p1}-d\\),如果遇到\\(y_{p2} > y_{p1}+d\\),可以直接break内循环，否则再检查满足\\(x_{p2} - x_m <= d\\)的点和该点的距离，这样的点最多只有6个。
```c++
typedef struct point {
    double x;
    double y;
};


int cmp_x(point a, point b) {
    return a.x < b.x;
}

int cmp_y(point a, point b) {
    return a.y < b.y;
}

double distance(point &a, point &b) {
    return sqrt(pow((a.x - b.x), 2) + pow((a.y - b.y), 2));
}

double close_pair_2(vector<point> &px, vector<point> &py) {
    if (px.size() <= 1) return INT_MAX;
    if (px.size() == 2) {//两个点
        return distance(px[0], px[1]);
    }
    int countx = px.size();
    int mid = countx / 2;
    double pivot_x = px[mid].x;

    //partition leftx和lefty点都是一模一样的，只不过是按x排序,还是按y排序的问题
    vector<point> leftx(mid+1), rightx(countx-1-mid);
    vector<point> lefty(mid+1), righty(countx-1-mid);
    int i = 0,k1=0,k2=0,j1=0,j2=0;
    for (int i = 0; i < countx; ++i) {
        if (i <= mid) leftx[k1++] = px[i];//i<=mid的元素，x<pivot_x
        else rightx[k2++] = px[i];
        if (py[i].x <= pivot_x) lefty[j1++] = py[i];//遍历y,按y的顺序放
        else righty[j2++] = py[i];

    }

    double min_distance = min(close_pair_2(leftx, lefty), close_pair_2(rightx, righty));
    double d = min_distance;
    int flag = 0;
    for (int i = 0; i < lefty.size(); ++i) {
        int k = flag;
        if (pivot_x - lefty[i].x <= d) { //p1区域内
            while (k < righty.size() && righty[k].y < lefty[i].y - d) ++k;//-d区域以下
            flag = k; //下次遍历时，直接从k处开始，因为y已经排序好了. right[k].y < left[i].y1 - d < left[i].y2-d(P1区域的y2>y1)
            for (int j = k; j < righty.size(); ++j) {//最多比较6次
                if (righty[k].y > lefty[i].y + d) break;//+d区域以上
                if (righty[j].x - pivot_x <= d) { 
                    min_distance = min(distance(lefty[i], righty[j]), min_distance);
                }
            }
        }
    }
    return min_distance;
}

int main()
{
    int n;
    while (cin >> n && n)
    {
        vector<point> px(n);
        int i;
        for (i = 0; i < n; ++i)
            cin>>px[i].x>>px[i].y;
        sort(px.begin(),px.end(),cmp_x);//按照x排序
        vector<point> py(px);
        sort(py.begin(), py.end(), cmp_y);//按照y排序
        printf("%.2lf\n", close_pair_2(px,py));
    }
    return 0;
}

```
  另外附上AC代码：更简洁点。
```c++
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <iostream>
#include <math.h>
using namespace std;
const double INF = 1e20;
const int MAXN = 100010;
struct Point
{
    double x, y;
};
double dist(Point a, Point b)
{
    return sqrt((a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y));
}
Point p[MAXN];
Point tmp[MAXN];

bool cmpx(Point a, Point b)
{
    return a.x < b.x;
}
bool cmpy(Point a, Point b)
{
    return a.y < b.y;
}
double closest_pair(int left, int right){
    double d = INF;
    if (left == right)return d;
    if (left + 1 == right)
        return dist(p[left], p[right]);

    int mid = (left + right) / 2;
    double d1 = closest_pair(left, mid);
    double d2 = closest_pair(mid + 1, right);
    d = min(d1, d2);
    int k = 0;
    for (int i = left; i <= right; i++)
    {
        if (fabs(p[mid].x - p[i].x) <= d)//P1 P2区域都找出来
            tmp[k++] = p[i];
    }
    sort(tmp, tmp + k, cmpy);
    for (int i = 0; i <k; i++)
    {
        for (int j = i + 1, m = 0; j < k && m < 11 && tmp[j].y - tmp[i].y < d; j++)
        {
            d = min(d, dist(tmp[i], tmp[j]));
            m++;//和自己区域比较5次，另一个区域比较6次。总共不超过11次。
        }
    }
    return d;
}

int main(){
    int n;
    while (cin >> n && n){
        for (int i = 0; i < n; i++)
            cin >> p[i].x >> p[i].y;
        sort(p, p + n, cmpx);
        printf("%.2lf\n", closest_pair(0, n - 1) / 2);
    }
    return 0;
}

```
