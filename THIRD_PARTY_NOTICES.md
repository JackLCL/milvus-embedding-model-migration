# 数据与模型来源

## SciFact 与 BEIR

实验使用 BEIR 发布的 SciFact 检索数据版本：5,183 篇文献、300 条测试查询。这里的 test split 是 BEIR 检索任务的划分，不是 SciFact 原始事实核查任务的隐藏测试集。

- 数据下载：[BEIR SciFact archive](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip)
- 原始项目：[allenai/scifact](https://github.com/allenai/scifact)
- 检索基准：[beir-cellar/beir](https://github.com/beir-cellar/beir)
- 数据卡：[BeIR/scifact](https://huggingface.co/datasets/BeIR/scifact)

SciFact 的作者为 David Wadden、Shanchuan Lin、Kyle Lo、Lucy Lu Wang、Madeleine van Zuylen、Arman Cohan 和 Hannaneh Hajishirzi，原始论文为 [Fact or Fiction: Verifying Scientific Claims](https://arxiv.org/abs/2004.14974)。

[原始 SciFact 许可说明](https://github.com/allenai/scifact/blob/master/LICENSE.md) 将 claims 与 evidence annotations 标为 CC BY 4.0，将来自 S2ORC 的摘要标为 ODC-By 1.0；BEIR 的 Hugging Face 数据卡另标注 CC BY-SA 4.0。以上为各上游页面的许可标注，第三方数据不因存放在本仓库中而改变其适用许可。

本仓库不分发完整 corpus 或数据归档。`recorded-results/retrieval_*.json` 包含公开查询文本、文献 ID、相关性标注与本次检索排序；`http_example_*.json` 包含少量公开文献标题和检索分数。这些文件是评测输出，不是线上用户或客户数据。文本内容来自上述公开数据集，检索排序和指标由本实验生成。

## Embedding 模型

- [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)：384 维。
- [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5)：768 维。

BAAI 模型页面标注 MIT License。本实验通过 FastEmbed 0.7.4 使用 Qdrant 分发的量化 ONNX 文件；仓库不包含模型权重。实际下载仓库、revision、文件大小和 SHA-256 记录在 `recorded-results/model-provenance.json` 中。

## 依赖与容器

Python 依赖版本见 `requirements.lock.txt`，容器镜像版本见 `docker-compose.yml`。依赖包、数据集、模型与容器镜像分别遵循其上游许可。
