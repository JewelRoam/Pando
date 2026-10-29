# PandoSI 技术蓝图（v0.9）

状态：v0.8。研究预览版已发布（见 RELEASE-NOTES.md）。目标：形成一套能跑通、可开源、可量化的“弹性异构自复制统一智能”完整方法。

## 0. 目标与非目标

**目标**

- 多台任意架构的廉价设备（CPU 盒子 / SBC / 手机 / Mac / GPU）组成一个逻辑上统一的智能体。
- 设备可随时加入、退出、掉线、复活，系统不重启、不丢失能力。
- 智能行为由决策模型群（System 1）高频执行，playbook（System 2 一次性编译）定义目标与策略。
- 自复制协议（Spore）在封闭沙箱中可量化验证：复制时间、成功率、决策/秒、共识准确率。
- 长期：分布式 RLCD 自训练，从“繁殖”走向“进化”。

**非目标**

- 不做通用文本生成（表达层不属于核心）。
- 不追求单机大模型的原始智力。
- 不在开放网络上做自复制实验（仅封闭沙箱）。

## 1. 三层时间尺度：智能的分工

| 层 | 执行者 | 频率 | 位置 | 失效后果 |
|---|---|---|---|---|
| 战略层 Playbook | LLM 或人类，一次性编写 | 写一次，版本化更新 | 离线/开发机 | 集群降级为旧策略，不停机 |
| 策略层 Policy | Laya 群（421M/322M） | 每秒数千次 | 每台设备 | 掉一台少一个 worker，共识仍成立 |
| 簿记层 Bookkeeping | 普通代码 | 持续 | 每台设备 | 可重建，无智能损失 |

关键判断：LLM 的角色从“运行时依赖”降级为“离线顾问”。运行时集群只有代码 + 权重，
零外部模型依赖。这是“能自己跑通”的核心。

## 2. 拓扑分层与调度

```
全局（WAN，高延迟）：数据并行 — 每组是完整副本，加组=加吞吐，零重分片
组内（LAN / 多桶静态服务 (v0.8: iGPU 多桶 25-40ms 变长负载已实测)，低延迟）：流水线并行 — 仅当模型 > 单机内存时启用
决策模型（<1GB）：  单机独立运行 — 永不切分，副本即弹性
```

规则：

1. 模型 ≤ 单机内存 → 每设备一份完整副本（首选，DP 弹性）。
2. 模型 > 单机内存 → 组内 PP 分片（LLM 才需要；Laya 永不需要）。
3. 组间永远 DP，禁止跨 WAN 的 TP/PP（除非走 leyten/shard 式推测解码流水线）。
4. 协调器（Root）不持模型层，必须放在离流水线近的位置
   （实测证据：协调器就近摆放，环延迟 174→102ms，白赚 40%，leyten/shard）。
5. 决策请求 → fan-out 到 k 个副本 → 校准概率多数投票/贝叶斯融合 → 共识输出。

## 2.5 单设备内异构拼装：ExecutionUnit 抽象（v0.6 新增）

原子资源单位从“设备”升级为“算力单元 EU”：一台 AI PC
（CPU+iGPU+NPU+外接 dGPU）最多可贡献 4 个 EU。

三种拼装形态：
- A 单 worker + HETERO 后端：按层切分给不同 EU（OpenVINO
  `HETERO:GPU,CPU`，贪心优先级），切分点必须实测优化；；吞吐型拼装用 OpenVINO MULTI 模式按请求并行（`MULTI:GPU.1,GPU.0`，投票 fan-out 天然多并发）
- B 多虚拟 stem：每 EU 跑一份 Laya 副本，各自参与 quorum——
  直接满足共识对异构副本的数量需求；
- C 混合：NPU 跑高频低延迟判定（Laya 定长 encoder 天然适配
  NPU 静态 shape，batch 需 pad），CPU 跑校准/管理，GPU 跑 draft。

硬件拓扑事实：iGPU/NPU 与 CPU 共享系统内存与同一带宽
（近零拷贝但会争抢带宽）；dGPU/外接 GPU 独立显存走 PCIe。
ONNX Runtime 单 session 单 EP 实例限制 → 双 GPU 用 OpenVINO
（GPU.0/GPU.1 原生枚举）或双 session。

弹性语义升级：外接 GPU 插上 = 新 EU 上线 = 一次 Graft 事件。
Stem/Root/Graft 协议贯穿 EU → 设备 → 集群三个层级。
真机基准（v0.7，2026-10-14）：Laya fp32 在 CPU EU 上 106ms/51ms p50
（int8 动态量化反而更慢 127ms，已弃用）；GPU/NPU 编译失败，根因
确认为决策头的 GatherND 算子无 GPU 内核且 NPU 编译器不识别
（四条修复路径见 research round 8）。当前 Intel 平台 EU 现实：
CPU 是唯一可用 EU，106ms 对决策群完全可用。
Graft 能力探测必须枚举 EU（OpenVINO `core.available_devices`
或 L0/DXVA），不再假设单加速器。
已实现：`graft.enumerate_execution_units()`（OpenVINO 优先，排除虚拟设备，CPU/nvidia-smi/Metal 回退）。注意：OpenVINO AUTO 自动选择默认把 NPU 排除在优先级外，让 NPU 参与必须显式指定优先级。

## 3. 核心协议

### 3.1 Spore（自复制）

Payload = { 权重(~800MB), playbook(签名), harness(二进制/脚本) }。

状态机：`SCAN → GRAFT → SPROUT → VERIFY → LIVE`

- SCAN：发现候选节点（沙箱内 mDNS/gossip/预置清单）。
- GRAFT：能力探测（架构/RAM/运行时/端口），选择角色（worker/relay/root）。
- SPROUT：传输 payload，校验签名与哈希，启动 worker。
  签名方案（v0.5 定案）：Ed25519（PyNaCl）签 JSON manifest；权重按  64KiB 分块哈希构建 Merkle 树，根哈希入 manifest——边传输边验证、  坏块重传、断点可信；内容定义分块（CDC）为 Phase 5 血统  进化提供增量传输（参考 arXiv:2104.02158）。
  文件格式：playbook 用 TOML，通信用 JSON，数据集用 JSONL，  完整性用签名 manifest。
- VERIFY：向 Root 注册，通过 3 次探测决策即视为 LIVE。
- 失败任意步 → 记录原因到 playbook 诊断日志，下个周期重试或放弃。

- 安全门（v0.3）
- 密钥管理 (v0.8): worker 密钥持久化; join --spore TOFU 钉扎 + 变更硬拒绝：借鉴 laya-guard 两层结构——确定性规则
  先行（白名单/黑名单/资源限制），灰区问题才进决策
  模型共识，仍灰则拒绝并上报。guard 服务不可用时
  fail-closed（拒绝而非放行）。

### 3.2 Root（共识）

- 输入：state + question set（来自 playbook 的 schema）。
- fan-out 到 k（默认 5，奇数）个 Stem。
- **副本多样性要求（v0.2 修正）**：同权重副本的决策误差高度相关——
  投票只能防节点故障与随机噪声，防不了模型系统性错误。
  共识价值来自异构：Laya 天然提供三个 checkpoint
  （english 421M / multilingual 322M / typed-decisions 微调版），
  另叠加温度与 schema 变体。Root 调度时强制每次投票覆盖
  ≥2 种 checkpoint。
  量化依据（v0.3）：9 个同质 LLM 评委仅提供约 2 张独立票（arXiv:2605.29800）；2 个异构 agent 可能胜过 16 个同质 agent（arXiv:2602.03794）。更高多样性需要按设备微调的血统变体（见 Phase 5）。
- 每个副本返回 {choice, score, probability}。
- 聚合：多数投票（离散）或概率平均（连续），输出决策 + 置信区间。
- 分歧超过阈值 → 升级路径：换一组副本重试 → 标记 playbook 盲区 → 离线顾问。

### 3.2.1 校准层（v0.3 新增，必须）

独立复现研究（arXiv:2609.33843）证实 Laya typed-decisions 检查点
**系统性低估自身置信度**（uniformly under-confident）。因此 Root 的
置信度门控不能直接使用原始概率：

- 每个 checkpoint 维护一个温度缩放/平台校准参数，定期用
  `laya-evals --max-ece 0.05` 验证；
- 量化（int8）后的 worker 必须重新通过 ECE 门才能以
  `已校准` 身份参与投票，否则降权重或降级为观察者；
- 聚合策略按问题类型选择：推理型问题用投票，
  知识型问题用概率均值（arXiv:2502.19130）。

### 3.3 Graft（异构接入）

**Android 接入默认走 Termux 路线（v0.4 确认可行）**：Termux 的
python-onnxruntime 包已被社区验证可用（issue #31634，仅有 import 警告）。
TV 盒子接入即：Termux + `pip install onnxruntime zeroconf` + pando/ 即可运行
worker，无需安卓原生应用打包；原生 AAR 包装留作 Phase 2+ 优化。
RK3588 类设备可选 RKNN NPU 加速（rknn-toolkit2 转换，需验证编码器
模型的算子支持与 opset 版本）；Amlogic S905 无可用 NPU，走 CPU 路线。

**局域发现选型确定（v0.4）**：python-zeroconf（纯 Python mDNS/DNS-SD）。
服务类型 `_pando._tcp.local.`，TXT 记录携带
{node_id, checkpoint, role, version}；Spore 的 SCAN 阶段即浏览该服务。
WAN 加入仍用 heartbeat/URL 注册。

新设备上线流程：自述能力 → Root 下发角色 → 运行基准（3 条标准决策）→ 定级
（worker/observer）→ 加入调度池。全流程零人工配置，目标 <60 秒。

### 3.4 拓扑感知调度

- 每 10s 交换 RTT 心跳，维护延迟矩阵。
- 调度优先级：就近副本 > 负载最低 > 最异构（刻意维持软硬件多样性以测试 Graft）。

## 4. 决策模型群

- 首选模型：Laya（convaiinnovations/laya 421M 英文 / 322M 多语言，Apache-2.0）。

**score 政策 (v0.8)**: score 头三血统均不具备决策级可靠性 (27 次测量)。结构性强制: score 问题必须声明 informational=true。风险评分使用 choice 格式 (18/27=67%)。
- 运行时：首选 ONNX Runtime / OpenVINO（CPU int8 已有社区验证），备选 PyTorch。
- 服务形态：每设备一个 FastAPI `/decide` 端点（或合并进 harness 单二进制）。
- Router：多语言场景由 Laya 自带 router 按语言分发 checkpoint。

## 4.5 设备分级与单机验收基准（v0.2，基于公开 CPU 实测）

LLM 侧的公认分界（Pi 5 实测）：交互门槛 ≈ decode ≥10 tok/s，
现实上限 ≈ ≤1B；Gemma 3 270M int8 @1280ctx 达 23.7 tok/s，
Llama 3.2 1B 10-15 tok/s，2B+ 多数仅 2-8。
持续负载无散热 Pi 5 会降频，吞吐掉约 1/3。
4GB 电视盒零公开整机实测，仅带宽推算：
1B Q4 实际约 4-8 tok/s，3B+ 大概率 OOM。

**对 PandoSI 的推论**：上述 decode 门槛主要约束自回归 LLM。
决策模型非自回归，一次前向 = 一次决策，
decode 门槛不适用，换成**单决策延迟预算**
（P50 <500ms LAN 共识）。因此：

| 设备档位 | 判定 | PandoSI 角色 |
|---|---|---|
| Pi 5 / x86 桌面（≥4GB） | 确定可跑 Laya | worker |
| **AI PC (Core Ultra 7 255H)** | **已实测**: CPU 106ms + iGPU 25.6ms(多桶) | **双 EU 异构共识 ✅** |
| 4GB 电视盒 | RK3588 已有首个真机数据点：Laya 多语言版 RKNN **509ms**/决策；Amlogic S905 仍未实测 | Phase 1 首个验证目标 |
| 2GB 以下盒子 | 存疑（ONNX int8 后 Laya 权重约 200-400MB） | 观察者模式 |

**Graft 验收集（v0.5 定案）**：16 个固定决策任务（覆盖 choice/score/noul 三类问题 × 三个 checkpoint），验收标准 16/16 正确 + p50/p95 延迟 + ECE 报告，按设备公开发布。原有基准项保留：cold load 时间、峰值 RSS、
单决策延迟、30 分钟持续热测降频曲线、标准任务集准确率。
任何设备的实测数据都是开源贡献。

**开发机实测结果 (v0.8)**：Core Ultra 7 255H + Arc 140T iGPU + AI Boost NPU
— CPU 106ms + iGPU 25.6ms（多桶）。NPU ✅ 95.7ms于算子支持（4 条路径归档）。

## 5. 演进路线（从繁殖到进化）

- 阶段 1：静态 playbook + 决策群 + 共识（本文所有内容）。
- 阶段 2：playbook 自动迭代 — 失败日志 → 离线 LLM 分析 → playbook 新版本 → 推送。
- 阶段 3：分布式 RLCD — 集群用自身决策数据自训练。v0.2 可行性增强：laya-rs 已有 RLCD 训练循环；Flower 支持 Android/Pi/Jetson 真机联邦客户端；决策模型每次 rollout 仅一次前向（非自回归），弱设备收集 rollout + 中心更新路径成立；参照 INTELLECT-2 / Hivemind，其 4xRTX3090 门槛是 LLM RL 的，不适用于本路线，
  权重版本化，A/B 对比新旧副本共识准确率，胜者成为新 root lineage。

## 6. 工程选型

| 组件 | 选择 | 理由 |
|---|---|---|
| 决策模型 | Laya | 唯一同时满足开源/小/非自回归/校准概率 |
| 设备运行时 | ONNX Runtime CPU | 异构覆盖面最广 |
| 节点通信 | gRPC 或 nanomsg | 轻量、跨语言 |
| 发现 | mDNS(LAN) + gossip(WAN) | 零配置接入 |
| 状态存储 | SQLite per device | 嵌入式、无依赖 |
| 需自建 | Spore/Root/Graft 协议、调度器、仪表盘 | 无现成引擎覆盖（空白即贡献点） |

## 7. 安全与伦理

## 代理免疫 (v0.8)

join/worker/worker 均自动设置 NO_PROXY={localhost,127.0.0.1,LAN_IP} - Windows urllib 遵循系统代理会黑洼回环/LAN 地址（实战发现，round 36）。

- 所有自复制实验仅在封闭沙箱（防火墙 VPC / 独立物理网段）进行。
- 开源发布以防御研究（blue-team）框架呈现，附检测特征说明。
- playbook 强制签名，Spore 拒绝无签名 payload。

## 8. 可量化验收指标

| 指标 | Phase 1 目标 |
|---|---|
| Graft 接入时间 | < 60s |
| 单决策共识延迟（LAN） | < 500ms |
| 决策吞吐（k=5, 10 台设备） | > 50 decisions/s |
| 单节点掉线后共识可用性 | 100%（k-1 ≥ 3） |
| Spore 复制成功率（沙箱） | > 90% |
