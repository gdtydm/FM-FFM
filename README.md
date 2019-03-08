# FM-FFM
FM and FFM implement with tensorflow by Python

推荐一篇理论的文章： [深入FFM原理与实践](https://tech.meituan.com/2016/03/03/deep-understanding-of-ffm-principles-and-practices.html)

###  2019 03 改进
1. 使用 estimator 修改原代码。
2. 借鉴 [ChenglongChen/tensorflow-DeepFM](https://github.com/ChenglongChen/tensorflow-DeepFM) 的代码，改变数据的储存结构，对于类别特征不再使用 one-hot 的编码方式。当类别特征很多的时候，使用原先的方式就是一个很高维度的稀疏矩阵，浪费内存。
3. 添加 DeepFm 的架构。（通过在 FM 模型中使用 --use_deep=True or False 来控制， 在 FFM 模型下会忽略 --use_deep 参数）

### 运行

详细参数: `hparams.py`
```shell
# FM
python main.py --model="fm" --model_path="./fm"

## ffm
python main.py --model="ffm" --model_path="./ffm"
```