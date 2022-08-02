---
layout: posts
title:  "Understanding the place of `min_leaf` in Jeremy Howard's algorithm"
date:   2022-08-02 11:56:18 +0100
toc: true
toc_label: "Unique Title" # defautl: Content
toc_icon: "heart"  # corresponding Font Awesome icon name without the "fa" prefix
toc_sticky: true   # enables sticky toc
---
While working through the *Intro to ML for Coders (2018)* course from fast.ai, I stumbled for a while on the following:

```py
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

1. TOC
{:toc}

**Notice:** This is an important info notice.
{: .notice}

## Heading 2

Some more text

### Heading 3

Even more text



