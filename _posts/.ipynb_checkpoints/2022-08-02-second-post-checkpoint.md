---
layout: posts
title:  "Understanding the place of `min_leaf` in Jeremy Howard's algorithm"
date:   2022-08-02 11:56:18 +0100
toc: true
toc_label: "Unique Title" # defautl: Content
toc_icon: "heart"  # corresponding Font Awesome icon name without the "fa" prefix
toc_sticky: true   # enables sticky toc
---
While working through the *Intro to ML for Coders (2018)* course from fast.ai, I stumbled for a while on the following (found in Lesson 7) which is a part of the project "Build a Random Forest from Scratch!":

```py
def find_better_split(self, var_idx):
        x,y = self.x.values[self.idxs,var_idx], self.y[self.idxs] # Extract pertinent data: x, y
        sort_idx = np.argsort(x) # Produce an index which will allow data to be sorted by x
        sort_y,sort_x = y[sort_idx], x[sort_idx] # New variables containing data sorted by x
        rhs_cnt,rhs_sum,rhs_sum2 = self.n, sort_y.sum(), (sort_y**2).sum() # Initiate scoring
        lhs_cnt,lhs_sum,lhs_sum2 = 0,0.,0.

    for i in range(0,self.n-self.min_leaf): 
                xi,yi = sort_x[i],sort_y[i]
                lhs_cnt += 1; rhs_cnt -= 1
                lhs_sum += yi; rhs_sum -= yi
                lhs_sum2 += yi**2; rhs_sum2 -= yi**2
                if i<self.min_leaf-1 or xi==sort_x[i+1]:
                    continue

                lhs_std = std_agg(lhs_cnt, lhs_sum, lhs_sum2)
                rhs_std = std_agg(rhs_cnt, rhs_sum, rhs_sum2)
                curr_score = lhs_std*lhs_cnt + rhs_std*rhs_cnt
                if curr_score<self.score: 
                    self.var_idx,self.score,self.split = var_idx,curr_score,xi
```
Credit: [fast.ai course: Intro to Machine Learning for Coders 2018](https://github.com/fastai/fastai1/blob/master/courses/ml1/lesson3-rf_foundations.ipynb)

The problem for me was the way that the `min_leaf` value was influencing the range iterator / loop counter. I couldn't understand the logic where a part of the loop is progressed before the `if i<self.min_leaf-1` statement potentially interupts it. And I found it unintuitive that the loop was started from value 0.

In an earlier version of this function, a simpler (and much less efficient) method for calculating the standard deviation (which is the basis of our score) was used. In that version, it would have made perfect sense to loop from 0 all the way to `self.n-min_leaf` (ie through all possible values which could be used as a split point).

But in this revision, it took me some time to realise that the flow is perfectly practicable & elegant, but not intuitive. The following pseudocode attempts to explain:

1. We count from 0. For each iteration up until the `min_leaf`th value, we simply perform the accumulation of the `cnt`, `sum`, and `sum2` variables: The flow is broken - and a score is not calculated - while our counter remains in the first `min_leaf` values.
2. From this point on, we perform the same accumulation adjustments, AND calculate the score, for all remaining values except the very last `min_leaf` ones.

The effect is that we omit the first and last `min_leaf` values from our scoring: the principle point here is that this doesn't matter because the first and last `min_leaf` values will be collected nonetheless because in the `find_varsplit()` function we create a mask to either side (inclusive) of the newly established split variable.

In order to try to demonstrate this more intuitively, I rearrange the function as follows:

```python
    def find_better_split(self, var_idx):
        
        def update_scores(val):
            nonlocal lhs_cnt, rhs_cnt, lhs_sum, rhs_sum, lhs_sum2, rhs_sum2
            lhs_cnt += 1; rhs_cnt -= 1
            lhs_sum += val; rhs_sum -= val
            lhs_sum2 += val**2; rhs_sum2 -= val**2
        
        x,y = self.x.values[self.idxs,var_idx], self.y[self.idxs] # Extract pertinent data: x, y
        sort_idx = np.argsort(x) # Produce an index which will allow data to be sorted by x
        sort_y,sort_x = y[sort_idx], x[sort_idx] # New variables containing data sorted by x
        rhs_cnt,rhs_sum,rhs_sum2 = self.n, sort_y.sum(), (sort_y**2).sum() # Initiate scoring
        lhs_cnt,lhs_sum,lhs_sum2 = 0,0.,0.
        
        if self.min_leaf > 1: # Accumulate scoring for first min_leaf-1 values
            for i in range(0, self.min_leaf-1):
                yi = sort_y[i]
                update_scores(yi)
        
        for i in range(self.min_leaf-1, self.n-self.min_leaf):
            xi,yi = sort_x[i],sort_y[i]
            update_scores(yi)
            
            if xi==sort_x[i+1]: continue # provisions for instances where we encounter duplicate sequential values

            lhs_std = std_agg(lhs_cnt, lhs_sum, lhs_sum2)
            rhs_std = std_agg(rhs_cnt, rhs_sum, rhs_sum2)
            curr_score = lhs_std*lhs_cnt + rhs_std*rhs_cnt
            
            if curr_score<self.score:
                self.var_idx,self.score,self.split = var_idx,curr_score,xi
```

In this revision, I think the logic flow is a little clearer, or at least more intuitive. In the original version, I think the process of sorting and working through all possible values while catering for dupicates feels quite cumbersome. I feel my revised version at least spells out the weaknesses of this technique a little better.