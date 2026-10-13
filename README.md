# 在 Milvus 3.0.2 中切换 embedding 模型：SciFact 端到端实验

[阅读技术文章](ARTICLE.zh-CN.md) · [查看实测结果](RESULTS.md) · [数据与模型来源](THIRD_PARTY_NOTICES.md)

## 获取代码

```bash
git clone https://github.com/JackLCL/milvus-embedding-model-migration.git
cd milvus-embedding-model-migration
```

## 需要什么

- uv（用于创建 Python 3.12 环境）；Docker Compose（若已有 Milvus 3.0.2 可跳过 Docker）。
- 首次运行需要能访问 BEIR 数据下载地址和 Hugging Face 模型仓库。
- 建议至少 8 个可用 CPU 线程；脚本使用 4 个推理 worker，每个 ONNX session 为 2 线程。
- 完整数据：SciFact 5,183 篇文献，test split 300 条查询。每篇文献作为一个 Entity，`text = title + "\n" + abstract`；模型最多处理 512 tokens。向量由模型实际推理生成。

## 启动 Milvus（没有现成实例时）

```bash
mkdir -p volumes/milvus
docker run --rm --user 0:0 --entrypoint chown \
  -v "$PWD/volumes/milvus:/var/lib/milvus" \
  milvusdb/milvus:v3.0.2 999:999 /var/lib/milvus
docker compose up -d
curl -f http://localhost:9091/healthz
```

Milvus 容器以 uid/gid 999 运行，需确保目录权限正确。若服务仍在启动，等待 Compose healthcheck 变为 healthy 后执行脚本。Compose 使用独立的项目名、网络和 volumes；默认 API 绑定 localhost:19530。可按自己的端口情况编辑 Compose。MinIO 使用公开的本地演示凭据 `minioadmin`；示例没有配置生产鉴权，Milvus 端口仅映射到本机回环地址。

## 执行

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.lock.txt
HF_HUB_DISABLE_XET=1 .venv/bin/python -u run_public.py \
  --uri http://localhost:19530 --model-cache ./models
```

脚本启动 HTTP 检索服务（默认 localhost:8080），接收查询文本，执行 Embedding 推理并调用 Milvus。主流程依次完成 BGE-small-en-v1.5 向量入库、新增向量字段、读取原文并回填 BGE-base-en-v1.5 向量、效果评测、灰度切换、回滚和旧字段清理。

脚本运行期间，可通过 `/search` 使用当前默认检索版本。BGE-base-en-v1.5 完成数据准备后，才可指定该版本查询：

```bash
curl http://localhost:8080/search -H 'Content-Type: application/json' \
  -d '{"query":"0-dimensional biomaterials show inductive properties."}'
```

`/route` 用于切换默认检索版本。自动验证期间，路由由脚本控制，手动修改会影响实验时序。此服务仅用于本机实验，绑定 loopback 地址，未配置鉴权，并随脚本结束关闭。

## 验证结果

- 进程退出码 0。
- `artifacts/LATEST_SUCCESS` 给出本次通过的运行目录。
- `artifacts/<run>/summary.json` 的 `success` 为 true。
- 每次运行使用独立、带时间戳的 Collection，不删除其他已有 Collection。

目录包含 schema_A/AB/B、metrics_A/B、每条查询的相关与召回 ID、HTTP 原始请求记录、灰度请求分配和运行时脚本快照。依赖固定在 requirements.lock.txt，数据 SHA-256 在 provenance.json。

## 模型和评测约定

- BGE-small-en-v1.5：BAAI/bge-small-en-v1.5，384 维。
- BGE-base-en-v1.5：BAAI/bge-base-en-v1.5，768 维。
- FastEmbed 0.7.4 的量化 ONNX CPU 实现。
- 查询加 `Represent this sentence for searching relevant passages: ` 前缀；文档不加查询前缀。
- HNSW，M=16、efConstruction=100、COSINE，查询 ef=128、top 10、Strong consistency。
- Recall@10 和 nDCG@10 对所有 300 条测试查询取平均。nDCG 使用 `(2^rel - 1)/log2(rank+1)` 增益；此数据的有效标注为二元相关性。
- 用固定公开语料比较模型；写入更新验证放在评测之后，使用独立 probe ID，并在结束时删除，避免污染模型比较。
- 每次 HTTP 检索均执行模型推理，记录的延迟包含推理、排队、HTTP 通信和 Milvus 检索。

## 适用范围

本实验验证单进程、低并发下的迁移流程，请求成功率不能外推为生产环境的可用性保证。示例为每次请求固定模型与向量字段；生产部署还需接入配置管理、鉴权、灰度发布和在途请求排空机制。持续增删改的数据源还需要协调增量写入与回填的顺序、版本和删除事件。

模型基于官方分发的 ONNX 权重；再次下载若仓库更新，需对照附带的模型来源及 hash。迁移期间同时保存两套向量和索引会增加存储、内存、CPU 及模型推理成本。

## 数据来源与许可

BEIR SciFact 下载：https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip
BEIR：https://github.com/beir-cellar/beir
SciFact 原项目：https://github.com/allenai/scifact
仓库不包含完整文献语料或模型权重，运行时从公开来源下载。`recorded-results/` 保留本次实验的指标、召回 ID、公开查询文本、相关性标注和少量文献标题；来源与上游许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 代码结构

正文展示的是和运行脚本相同的 Milvus API、数据拼接、模型调用和路由逻辑。完整脚本额外包含四个 CPU worker、HTTP 服务、阶段日志、回填后的逐条审计、评测指标以及错误维度测试。复现迁移流程时，需同时完成字段新增与回填、索引构建、字段加载，以及查询模型和目标字段的同步切换。

## HTTP 请求示例与运行记录

自动化流程会依次执行 BGE-small-en-v1.5 服务、两模型混合请求、默认 BGE-base-en-v1.5、回滚 BGE-small-en-v1.5、再次切换至 BGE-base-en-v1.5、删除旧向量字段。HTTP 服务的生命周期与实验一致；请求示例见 `recorded-results/http_example_A.json`、`recorded-results/http_example_B.json`，自动流程运行期间，默认路由由脚本统一控制。

SciFact 数据归档 SHA-256（本次验证）：
`536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165`

两套 ONNX 权重的 HF 仓库、revision、文件大小及 SHA-256 见 `recorded-results/model-provenance.json`，用于确认复测模型与本次一致。

## 连接配置

默认连接地址为 localhost。Compose 使用本机端口 19530 / 9091，HTTP 检索服务使用端口 8080。连接已有 Milvus 实例时，通过脚本的命令行参数指定地址；调整 Compose 映射端口时，同步修改客户端连接配置。

## 原生画板图源

正文总览图使用飞书原生画板。`diagrams/native-overview-nodes.json` 保存可编辑的节点定义；`build_native_board.cjs` 生成这些节点；`native-overview-feishu.jpg` 是飞书导出的跨平台预览。节点定义与预览图均随代码包提供。

## 模型效果对比

`diagrams/comparison-metrics.json` 包含替换前后指标、绝对增量、相对提升和逐查询变化统计。`comparison.svg` 可用于网页或技术博客排版，`comparison-native-nodes.json` 为飞书画板节点定义。计算依据是相同的 300 条测试查询及随包提供的逐查询检索结果。

## 复核指标

完成实验后，运行以下脚本，根据下载的官方相关性标注独立复算本次运行的指标：

```bash
.venv/bin/python verify_metrics.py
```

`recorded-results/` 是本文所报告实验的固定结果。运行时生成的 `artifacts/`、下载的模型、数据库文件和本地环境配置均已加入 `.gitignore`。
