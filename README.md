# new_network_cup
风险评估

## 提升技巧

- 是否去除异常值

判断每一列的异常值(-99)数量是否超过一定比例，如果超过直接去除

       去除异常值：0.70961 (lgb)
       去除异常值：0.70131(lr)
       
       不去除异常值：0.70961 (lgb)
       不去除异常值：0.72579(lr)
    
- 是否进行数据预处理
主要采用下面两种方法：
```text

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
imputer = Imputer(strategy="median")
scaler = StandardScaler()
```

       去除异常值：0.72859 (lr) 目前分数最高
       不去除异常值：0.7...8(lr)
       
- 模型集成
```text
1. 将lgb和lr的结果相加求平均
```