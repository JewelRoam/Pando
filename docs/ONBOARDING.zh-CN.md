# PandoSI 接入与授权体验（v0.2）

> 目标不是“支持所有系统”，而是让任何设备先得到一个诚实的准入计划，
> 再由用户逐项授权。默认路径必须零特权、可回退、五分钟内可验证。

## 产品原则

1. **一个入口产出计划，而不是直接改系统。**
   `python -m pandosi.doctor` 只读探测能力；`onboard` 后续只执行
   非特权步骤，敏感步骤打印精确命令。
2. **失败信息必须带下一步。**
   不说“NPU 不可用”，要说“缺哪个包/驱动、用户需要在哪个入口授权”。
3. **敏感操作是授权，不是安装器副作用。**
   装驱动、开防火墙端口、注册自启动、访问加速器都不应藏在
   `pip install` 或 bootstrap 脚本里。
4. **沙箱按能力分级，而不是一刀切。**
   本地推理不需要 Docker；拉取/执行新副本必须可丢弃沙箱。

## 渐进式用户旅程

```text
Step 0  只读体检       pandosi doctor --json
Step 1  单机价值       tools/demo.py：本机 root + 多 stem/多 EU 共识
Step 2  本机加速       用户选择安装 onnxruntime/openvino；仍无需管理员
Step 3  显式入群       用户粘贴 root 地址 + 配对码，stem 只服务该 root
Step 4  沙箱实验室     Docker/Podman/WSL 中运行 Spore 复制实验
Step 5  常驻服务       systemd --user / Windows 计划任务 / Termux sv，显式开启
```

当前仓库已有 Step 0/1/3/4 的最小实现。Step 3 已使用 HTTP pairing token；
Step 5 仍应先补授权过期和平台服务适配，再追求常驻。

## 授权记录

使用本地可审计的 `grants.toml`，不要用全局“已信任”开关：

```toml
[grants.lan_listen]
scope = "192.168.1.0/24"
resource = "0.0.0.0:8080"
approved_by = "interactive"
expires_at = "2026-10-21T00:00:00Z"
evidence = "doctor plan #<hash>"

[grants.npu_driver]
action = "open-vendor-installer"
approved_by = "interactive"
command = "vendor-specific-command-from-doctor"
```

规则：

- 授权范围最小化：先 loopback，再 LAN 单播，不默认广播全网。
- 每个敏感动作单独过期；变更网络、驱动或 Python 环境后必须重新探测。
- root/stem 的长连接必须绑定显式 root 身份；不能因为同网段就加入。

## Bootstrap 脚本怎么设计

推荐形态不是 `curl | sh`，而是一个可审阅的 `pandosi onboard` 流程：

```text
pandosi onboard --dry-run
  -> 输出 3 张表：已可用 / 可无提示安装 / 需要人工授权

pandosi onboard --apply-nonprivileged
  -> 只 pip/pipx 安装纯 Python 依赖、建缓存目录、写本地计划

pandosi onboard --grant lan_listen
  -> 展示将监听的地址和防火墙影响，等待 yes

pandosi onboard --grant npu_driver
  -> 打开厂商页面或打印 sudo/UAC 命令；由用户执行
```

平台差异用“适配器输出命令”，不追求大而全的静默安装：

| 平台 | 默认无特权路径 | 需要授权的扩展 |
|---|---|---|
| Windows AI PC | Python + pip + CPU/iGPU | OpenVINO/NPU 驱动、Docker Desktop、防火墙 |
| Linux/SBC | pipx + CPU + systemd --user | 包管理器、firewalld/ufw、Podman/Docker |
| macOS | Python + CPU/Metal | Container/VM 运行时、辅助工具授权 |
| Android/Termux | Termux + pip + CPU | termux-services、存储/电池白名单 |

## 沙箱决策

**不是所有功能都要 Docker。** 能力降级比拒绝接入更好：

| 能力 | 无沙箱能否使用 | 理由 |
|---|---|---|
| 只读 doctor | 可以 | 不安装、不联网、不改配置 |
| 本机模型推理/共识 | 可以 | 不拉新代码，不执行远端载荷 |
| 显式 root/stem 入群 | 可以 | 只允许用户给出的地址和配对令牌 |
| LAN 自动发现 | 默认关闭 | 打开需 LAN 授权，且 mDNS 保持可见 |
| 模型/Spore 拉取 | 不可以 | 必须容器/VM/一次性目录，且签名验证先行 |
| 远端 playbook 注入 | 不可以 | 必须离线审阅、hash-chain 验证和 quorum 批准 |

实现上应保持现在的 fail-closed：`lab.detect_sandbox()` 发现
Docker、Podman 容器或 WSL 才解锁 Ring 2；检测不到时允许继续作为
observer，但禁止繁殖/执行载荷。

## 易用性指标

- `git clone` 到本机 mock 共识成功：**≤3 条命令，≤5 分钟**。
- `doctor` 中每个 `[miss]` 都有精确修复命令；每个 `[locked]` 都说明
  是缺沙箱、缺授权还是缺硬件。
- 默认安装不请求管理员；任何提升权限动作都可以被单独拒绝。
- 用户拒绝 Docker 后，产品必须仍有本地价值，而不是变成空壳。

## 下一步实现清单

1. [x] `doctor --json` 已包含 `plan`；另有只读的 `doctor --plan`。
2. [x] `pandosi/onboard.py --dry-run` 已实现；明确拒绝安装/提权/改授权。
3. join 协议加 root pairing token：root 启动生成短码，stem 必须携带。
4. `grants.toml` 读写与过期检查，接入 join/lab 的启动前置条件。
5. 发布前先补 `pip install pandosi`/PyPI 或 `pipx install git+...` 验收。
