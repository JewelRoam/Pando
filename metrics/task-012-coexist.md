# Task 012 result

Status: done

## 条件与做法

- 同一台 MacBook Pro（Apple 芯片，16 GB），同一 8B 服务 `127.0.0.1:8083` / 外代理 `:8082`。
- 同一 `heldout4_test.jsonl` 240 行、prompt variant B、temperature 0、max_tokens 64。
- A：4B+8B 同驻；B：停掉 4B 服务（8081）与 4B 代理（8080），确认进程退出后仅跑 8B。
- 跑完后已重新启动 4B，两个服务都恢复在线。mac-text stem、rootd、邮箱服务未触碰。

## A/B 对照

| 指标 | A：4B+8B 同驻 | B：8B 独占 | B 相对 A |
|---|---:|---:|---:|
| 240 行耗时 | 60.1 s | 54.8 s | -5.3 s |
| 平均 tok/s | 25.4229 | 25.5165 | +0.368% |
| p50 延迟 | 248.32 ms | 245.82 ms | -1.008% |
| p95 延迟 | 300.97 ms | 250.72 ms | -16.696% |
| 平均延迟 | 250.28 ms | 228.19 ms | -8.824% |
| 准确率 | 0.9250 | 0.9250 | 0 |
| 8B 进程 footprint（跑后） | 1947 MB | 1975 MB | +28 MB（B 反而略高，量级像噪声） |
| 4B 进程 footprint（跑后） | 5616 MB | 0 | -5616 MB |
| swap used（跑后） | 20606.69 MB | 20524.00 MB | -82.69 MB |
| swap free（跑后） | 897.31 MB | 980.00 MB | +82.69 MB |

补充原始快照：

- A 跑前 swap：total 23552.00 / used 21901.69 / free 1650.31 MB。
- A 跑后 swap：total 21504.00 / used 20606.69 / free 897.31 MB。
- B 跑前 swap：total 21504.00 / used 20644.06 / free 859.94 MB。
- B 跑后 swap：total 21504.00 / used 20524.00 / free 980.00 MB。

`ps` 在这台 macOS 上返回的 RSS 只有 1–2 MB（与 `top`/`footprint` 明显不一致），因此进程内存用 `footprint` 的 `phys_footprint`，并以 `top MEM` 交叉确认。原始逐次快照在 `~/.pandosi/task012/*-before.json` 与 `*-after.json`。

## 一句结论

在 16GB 机器上，4B+8B 共存的代价主要是 4B 进程额外占用约 5.6 GB；对 8B 的平均吞吐影响很小（约 -0.37%），但 p95 从 250.7 ms 升到 301.0 ms（同驻侧约 +20%）。如果 4B 的能力值得常驻，平均吞吐几乎不受损；如果 8B 的尾延迟或释放约 5.6 GB 更重要，应停掉 4B。

## 最终服务状态

- 4B：PID 54501（8081）/ 代理 PID 54504（8080），`/v1/models` 返回 `qwen3-4b-instruct-2507`。
- 8B：PID 13739（8083）/ 代理 PID 13836（8082），`/v1/models` 返回 `qwen3-8b-instruct`。
- 未触碰：rootd 8765、mac-text 8766、agent mailbox 8776；邮箱 `/health` 返回 ok/open 0。

## 产出

- 必需的合并 summary：`/Users/zhengenrong/Downloads/PandoSI/metrics/task-012-coexist.json`
- 两份详细 summary：`~/.pandosi/task012/condition-{a,b}-summary.json`
- 两份原始 votes：`~/.pandosi/task012/condition-{a,b}-votes.json`
- 运行日志：`~/.pandosi/task012/condition-{a,b}.log`
- 恢复记录：`~/.pandosi/task012/restore.log`

## 可复现命令

```bash
# A：同驻
/Users/zhengenrong/Downloads/PandoSI/.venv/bin/python \
  /private/tmp/pandosi-task-012/tools/llm_decision_check.py \
  --suite /private/tmp/pandosi-task-012/assets/heldout4_test.jsonl \
  --variant b --base-url http://127.0.0.1:8082/v1 \
  --api-key-file /Users/zhengenrong/.pandosi/llm-api-8b.token \
  --model qwen3-8b-instruct --timeout 180 \
  --out /Users/zhengenrong/.pandosi/task012/condition-a-votes.json \
  --summary-out /Users/zhengenrong/.pandosi/task012/condition-a-summary.json

# B：停 4B 后跑同样的 240 行（原 PID 70733/70730）
kill -TERM 70733 70730
# 然后重复上面命令，输出改为 condition-b-*。

# 恢复 4B
/Users/zhengenrong/Downloads/PandoSI/.venv/bin/python /Users/zhengenrong/.pandosi/llm-daemon.py
/Users/zhengenrong/Downloads/PandoSI/.venv/bin/python /Users/zhengenrong/.pandosi/llm-proxy-daemon.py
```
