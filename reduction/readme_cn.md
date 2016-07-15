# 规约

包含两遍规约、基于原子操作的单遍规约和非原子操作的单遍规约已经对应的基于模板的循环展开。

测试代码有使用[thrust](http://thrust.github.io/)和不使用thrust两个版本。

# 使用

```shell
make
./reduction_cuda.exe
./reduction_thrust.exe
```

更多请参考：[CUDA并行算法系列之规约](http://zh.5long.me/2016/algorithms-on-cuda-reduction/)。