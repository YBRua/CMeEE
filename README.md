# AI3612 知识表征与推理 课程项目 CMeEE 命名实体识别

## 快速开始

### 环境配置

安装 PyTorch 和其他依赖库

```sh
pip install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html --no-cache
pip install -r requirements.txt
```

### 下载预训练模型

- 本项目默认使用 `bert-base-chinese`，可以在[这里](https://huggingface.co/bert-base-chinese)找到
- 请将 `bert-base-chinese` 置于 `src` 目录同级

### 准备数据集

- 本项目需要 [CBLUE](https://tianchi.aliyun.com/cblue) 下的 CMeEE 数据集
- 请将 CBLUE 数据集放置在 `data` 目录下，如果没有 `data` 目录，请新建一个（x

至此，项目目录应该类似于

```txt
- bert-base-chinese
- data
  - CBLUEDatasets
    - CMeEE
- src
```

### 运行脚本

- 复现脚本全部保存在 `./src` 目录下

在超算平台上可以通过 `sbatch` 提交任务。例如，要复现 W2NER 模型

```sh
cd src
sbatch run_cmeee_w2ner_tuned.sbatch
```

或者也可以使用 `source run_cmeee_w2ner_tuned.sbatch` 在 bash 中直接运行

## `src` 目录架构概述

### 源代码

- `ee_data`: 原 `ee_data.py` 扩增后的模块，负责数据加载
- `model`: 原 `model.py` 扩增后的模块，定义模型逻辑
  - 其中 `bert_*.py` 是整合了 BERT 的完整模型
  - `*_head.py` 是不同的分类器头
- `args.py` 负责解析命令行参数
- `ee_data_tests.py` 是数据加载的一些测试代码
- `logger.py` 是日志工具
- `loss_funcs.py` 包含了部分自定义的损失函数
- `metrics.py` 包含了计算 F1 指标的工具类
- `result_gen.py` 包含了部分建模方式下的解码函数
- `run_cmeee.py` 是主函数所在的位置
- `trainers.py` 是对 Huggingface 的 Trainer 做了部分逻辑覆盖的 Trainer

### 脚本文件

- `run_cmeee_nested.sbatch`: Project 1 用，运行嵌套线性头模型
- `run_cmeee_crf.sbatch`: Project 1 用，运行嵌套 CRF 头模型
- `run_cmeee_globalptr.sbatch`: 运行 [Global Pointer](https://github.com/bojone/GlobalPointer) 模型
- `run_cmeee_w2ner.sbatch`: 运行 [W2NER](https://github.com/ljynlp/W2NER) 模型。使用默认训练超参数。
- `run_cmeee_w2ner_tuned.sbatch`: 运行 W2NER 模型。使用原仓库配置的训练超参数。
