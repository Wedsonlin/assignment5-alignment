# W&B Alignment MCP Analysis Report

数据来源：W&B MCP server  
Entity：`ltao02845-sun-yat-sen-university`  
项目：`sft`, `ei`, `grpo-lr-sweep`, `grpo-baseline-ablation`, `grpo-length-norm-ablation`, `grpo-std-norm-ablation`, `grpo-off-policy-sweep`, `grpo-prompt-ablation`  
CSV：`wandb_mcp_run_summary.csv`

## 数据范围

本次文件基于 W&B MCP 二次校验得到的 run-level summary 数据生成，共 `47` 个 runs：

| 状态 | 数量 |
|---|---:|
| finished | 44 |
| killed | 2 |
| crashed | 1 |

所有项目的用户 config keys 均为空。GraphQL `config` 中只有 `_wandb` 运行元数据，因此学习率、样本量、ablation 条件等实验变量只能从 `run_name` 解析。

## 项目总览

| 项目 | runs | 状态 | 主指标最佳 run | 最佳值 |
|---|---:|---|---|---:|
| `sft` | 7 | 7 finished | `num_example_1024` | `eval/acc=0.2226` |
| `ei` | 12 | 12 finished | `G_5_E_2` | `eval/acc=0.1664` |
| `grpo-lr-sweep` | 6 | 4 finished, 2 killed | `grpo_reinforce_lr3e-5` | `eval/total_reward=0.3955078125` |
| `grpo-baseline-ablation` | 2 | 2 finished | `grpo_reinforce_with_baseline` | `eval/total_reward=0.37109375` |
| `grpo-length-norm-ablation` | 2 | 2 finished | `grpo_dr-grpo` | `eval/total_reward=0.3447265625` |
| `grpo-std-norm-ablation` | 2 | 2 finished | `grpo_no_std_norm_lr1e-5` | `eval/total_reward=0.373046875` |
| `grpo-off-policy-sweep` | 14 | 13 finished, 1 crashed | `grpo_ep2_bs128` | `eval/total_reward=0.4013671875` |
| `grpo-prompt-ablation` | 2 | 2 finished | `grpo_question-only` | `eval/total_reward=0.5869140625` |

## 主要发现

1. **SFT**：最佳 run 是 `num_example_1024`，`eval/acc=0.2226`。样本量扩大整体有收益；filtered 版本在 128/256/512 的可比设置下没有稳定提升。
2. **EI**：最佳 run 是 `G_5_E_2`，`eval/acc=0.1664`。按 run name 推断，`G=5` 组整体优于较小 G。
3. **GRPO learning-rate sweep**：最佳 finished run 是 `grpo_reinforce_lr3e-5`，`eval/total_reward=0.3955078125`。`lr=1e-4` 和 `lr=3e-4` 均为 killed，且 `eval/total_reward=0`，说明高学习率明显不稳定。
4. **GRPO baseline ablation**：`with_baseline` 的 `eval/total_reward=0.37109375`，高于 `no_baseline=0.197265625`。
5. **GRPO length normalization**：`dr-grpo` 的 `eval/total_reward=0.3447265625`，高于 `grpo=0.25`。
6. **GRPO std normalization**：`no_std_norm` 的 `eval/total_reward=0.373046875`，高于 `std_norm=0.2978515625`。
7. **GRPO off-policy sweep**：最佳为 `grpo_ep2_bs128`，`eval/total_reward=0.4013671875`。`grpo_ep6_bs64` crashed，且 `train/clip_fraction=0.4191955029964447`，是该项目中明显的失败点。
8. **GRPO prompt ablation**：`grpo_question-only` 是全部 GRPO finished runs 中最高，`eval/total_reward=0.5869140625`，比 `grpo_r1-zero=0.38671875` 高 `0.2001953125`。同时它的 `eval/response_length=576.916015625`，明显长于 `r1-zero=255.4658203125`。

## 二次校验说明

使用 W&B MCP 完成以下校验：

- `list_entities_tool` 确认 entity 为 `ltao02845-sun-yat-sen-university`。
- `query_wandb_entity_projects` 确认 8 个目标项目都存在。
- `probe_project_tool` 确认每个项目的 run 数、状态分布、metric keys、config keys 和 history 状态。
- `query_wandb_tool` 拉取全部 47 个 runs 的 `summaryMetrics`、`state`、`historyLineCount`、`config`。
- `get_run_history_tool` 抽样核对关键 run 的 history：
  - `sft/gtj71m79`：`num_example_1024`，`total_steps=1559`
  - `ei/t8btj8jt`：`G_5_E_2`，`total_steps=1580`
  - `grpo-prompt-ablation/gu7kix6p`：`grpo_question-only`，`total_steps=219`
  - `grpo-prompt-ablation/auiix20i`：`grpo_r1-zero`，`total_steps=219`
  - `grpo-off-policy-sweep/ens5k83f`：`grpo_ep2_bs128`，`total_steps=204`
  - `grpo-off-policy-sweep/0tphuooo`：`grpo_ep6_bs64`，`total_steps=1008`

## 建议

下一轮实验应优先修复 W&B config logging：把 `learning_rate`、`num_examples`、`filtered`、`G`、`E`、`baseline`、`length_norm_variant`、`std_norm`、`off_policy_epochs`、`batch_size`、`prompt_variant` 等字段写入 `wandb.init(config=...)`。否则后续分析只能依赖 run name，容易出错，也无法在 W&B UI 中可靠分组。

机制验证上，最值得优先复跑的是 prompt ablation：固定模型、数据、seed、optimizer、训练步数和评估协议，只改变 `prompt_variant`，比较 `r1-zero` 与 `question-only`。若 `question-only` 仍保持约 `0.20` 的 `eval/total_reward` 优势，则 prompt 约束是主要机制；若优势消失，则当前差异更可能来自随机性或未记录配置。
