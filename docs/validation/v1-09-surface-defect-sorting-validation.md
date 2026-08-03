# V1-09 构件表面缺陷检测与合格/缺陷分流：最终验证事实记录

> 本文件是 V1-09 Task 12 的有界事实记录，不替代协调端的独立验收或集成裁决。所有在线结果均引用既有、已保留的唯一在线证据；本任务没有重跑 CoppeliaSim。

## 1. 版本与检查点

- authority base：`origin/main=1c11302811a070d62c4d440a012a0399027a1683`。
- branch：`codex/v2-2-v1-09-surface-defect-sorting`。
- 写报告前候选：`9e2b9500bb546ee37f8055f772194b91b9afa5f3`。
- 写报告前 merge-base：`1c11302811a070d62c4d440a012a0399027a1683`。
- 工作树：`C:\Users\yqzhe\.config\superpowers\worktrees\robot-vision-lab\v2-2-v1-09-surface-defect-sorting`，写报告前 clean，无 upstream，远端 feature 尚不存在。
- Python：`C:\Users\yqzhe\.config\superpowers\worktrees\robot-vision-lab\v2-2-v1-09-surface-defect-sorting\.venv-vision\Scripts\python.exe`，Python `3.11.9`，pytest `9.1.1`。
- 本报告提交/推送后的 local 与 remote SHA 由提交后 `git rev-parse HEAD` 和 `git ls-remote --heads origin codex/v2-2-v1-09-surface-defect-sorting` 原样核对并在本次交付回报中记录；报告不对包含自身的 Git commit SHA 做不可实现的自引用。

## 2. base..candidate 原始 Git 历史

以下清单由 `git log --reverse` 与逐提交 `git show --name-status` 提取；路径是该提交实际变更的仓库相对路径。

1. `326f1ca9a2e44c2bef1e364f34778c2dcb7accd` — `feat(v1-09): define defect sorting contracts`：`RETAINED_FILES.txt`、`tests/test_experiments/test_defect_sorting.py`、`vision_platform/experiments/__init__.py`、`vision_platform/experiments/defect_sorting.py`。
2. `61a21631e7c612b45ea22fe2286e0666d390c0e9` — `feat(v1-09): guard defect sort execution`：`RETAINED_FILES.txt`、`tests/test_student_programs/test_v1_09_defect_sort_guard.py`、`vision_platform/student/defect_sort_guard.py`。
3. `c1db18718180d133545c42805999dcc941485a3e` — `fix(v1-09): align defect crops and route whitelist`：`tests/test_experiments/test_defect_sorting.py`、`vision_platform/experiments/defect_sorting.py`。
4. `bbd0fed28287e73da603cfc71200d4815e3ca055` — `feat(v1-09): generate original defect assets`：七张 V1-09 PNG、`simulation/vision_defect_sorting_lab/defect_assets_manifest.json`、`tests/test_simulation/test_v1_09_defect_assets.py`、`tools/vision_lab/generate_v1_09_defect_assets.py`。
5. `0148651d219c42668d0dabbbcf4e37363a577bf0` — `feat(v1-09): define defect sorting scene contract`：`.gitattributes`、`RETAINED_FILES.txt`、`simulation/training_scenes/build_scene.py`、`simulation/vision_defect_sorting_lab/acceptance_ground_truth.json`、`profiles.json`、`scene_spec.json`、V1-09 scene contract/builder tests、两个 scene builder 工具。
6. `6807a85f78bfd9bba9afba969cae5da7d479733c` — `feat(v1-09): analyze hash-bound defect batch`：`RETAINED_FILES.txt`、缺陷资产/服务/可视化生产模块及其三组实验测试。
7. `8cc16090064c84ca8e356bf8a4b45e8a579994bf` — `feat(v1-09): expose controlled defect analysis`：协议、SDK、capabilities、gateway 与对应 V1-09 测试文件。
8. `95641c8169e90d30c0df3fba584ff4d872993b2a` — `feat(v1-09): execute private defect sort actions`：V1-09 runner 测试、`vision_platform/student/experiment_gateway.py`、`vision_platform/student/runner.py`。
9. `f2abe59a1dfdff2fa7a0edae117395ca46ca2364` — `chore(v1-09): retain private runner tests`：`RETAINED_FILES.txt`。
10. `9cc99a0b0334c777f1d29741c195c00ee3d2519e` — `feat(v1-09): publish surface defect sorting lab`：V1-09 配置、catalog、课程、scene manifest、学生模板、CLI/catalog/UI 与相关测试。
11. `3e9e4c5f4b1f0b2f2eab7c49b3cbd2959957f873` — `feat(v1-09): record controlled defect evidence`：`RETAINED_FILES.txt`、V1-09 CLI/probe/evidence 测试、实验运行脚本、CLI/probes/gateway/runner。
12. `4ec5d5fcdf3065477d2aa9f91be6f66279bc1958` — `feat(v1-09): explain defect sorting evidence`：结果面板及其测试。
13. `846e3e3ef9c64af486bd408ec03f02d192b076c9` — `fix(v1-09): align scene contract with fixed rois`：scene builder/spec、缺陷服务/排序、场景合同与 guard 测试。
14. `fbd17ecfbec0c3b9ce6897af0f23d4eae51c3e99` — `fix(v1-09): synchronize published camera profile`：scene builder、profiles/spec、SDK/catalog 测试与生产模块。
15. `9a9eee328c8790635f5c7c4f776cc07e13ddd34f` — `fix(v1-09): preserve guarded final probe context`：V1-09 runner 测试与 runner。
16. `8bf5909c7175528fb634452e7f043300e170f8d0` — `fix(v1-09): honor student log contract`：V1-09 学生模板及材料测试。
17. `52088b3fa30fadbb41e5f7d163d7b334155c5f8d` — `fix(v1-09): align defect receipt schema`：receipt serializer、defect sorting 测试、协议 SDK 测试。
18. `de1d6a9e5d16f1ed84c0dec523ba224447a8910c` — `feat(v1-09): deliver defect sorting scene`：唯一 V1-09 `.ttt`、scene manifest、acceptance/readiness/delivery/gateway/readiness/scene setup 与 `RETAINED_FILES.txt`。
19. `2738d42db920f893e71c084d8889e6c136e65b50` — `fix(v1-09): retain wrapper-owned runtime evidence`：`tests/test_acceptance/test_coppeliasim_v1_09.py`、`tools/vision_lab/run_v1_09_surface_defects.ps1`。
20. `9e2b9500bb546ee37f8055f772194b91b9afa5f3` — `test(v1-09): align historical catalog expectations`：仅三个历史材料测试文件，11 insertions/3 deletions。

## 3. 设计、计划与修复依据

这些设计/计划提交来自只读设计工作树；它们是依据记录，不宣称已合入 feature 分支：

- 原 Task 11 场景合同：`6636a9c1a18075600a503bf83b72527c39bd1644`，`docs/superpowers/specs/2026-08-03-v1-09-task11-scene-contract-revision-design.md`；`e9e1d97db72789abd480baeaaa520d776aa76751`，`docs/superpowers/plans/2026-08-03-v1-09-task11-scene-contract-revision-plan.md`。
- 渲染合同：`0f799352f6019b5df6d6d947c3c44dccfacdc1bc`，`docs/superpowers/plans/2026-08-03-v1-09-task11-rendering-contract-repair-plan.md`；修订 `6998a85b0cdb47381d51e5219797b8b6bec976ea`，同一路径。
- profile/calibration：`89564df8a285b170bb855ac5413c5e7e0b910987` / `0a042e1d5856ccc74f4a1577fdef073842a5f7ac`，分别为 `docs/superpowers/specs/2026-08-03-v1-09-profile-calibration-contract-revision-design.md` 与 `docs/superpowers/plans/2026-08-03-v1-09-profile-calibration-contract-revision-plan.md`。
- mask evidence：`f1d2cd74dc082dc11f31871fda1900b36ba60114` / `8d85e1b289e0523919498a44fe8fbf16f6cda1e3`，分别为 `docs/superpowers/specs/2026-08-03-v1-09-mask-evidence-contract-repair-design.md` 与 `docs/superpowers/plans/2026-08-03-v1-09-mask-evidence-contract-repair-plan.md`。
- runner/guard 生命周期：`97a86227e51cead584850dec7f063d73bc271ad6` / `5da05de20bf272fbf7944fbd9c28c5565719961b`，分别为 `docs/superpowers/specs/2026-08-03-v1-09-runner-guard-lifecycle-repair-design.md` 与 `docs/superpowers/plans/2026-08-03-v1-09-runner-guard-lifecycle-repair-plan.md`。
- template log：`eda2fb9d2c506059748559662553b6f0951a8a24` / `59879088213868b2649ca119673a2b995015a88e`，分别为 `docs/superpowers/specs/2026-08-03-v1-09-template-log-contract-repair-design.md` 与 `docs/superpowers/plans/2026-08-03-v1-09-template-log-contract-repair-plan.md`。
- receipt schema：`82c4bec389a9d7bbf66aa03d0805f36543105fda` / `d8e564cf24b87e27d00e5bb8e8984906c8ea667b`，分别为 `docs/superpowers/specs/2026-08-03-v1-09-defect-receipt-schema-repair-design.md` 与 `docs/superpowers/plans/2026-08-03-v1-09-defect-receipt-schema-repair-plan.md`。
- final probe context：`04bb074efc764a9fb53316610507c571483a0825` / `0deecbb7366e37198beb0dceeab069674468fc1d`，分别为 `docs/superpowers/specs/2026-08-03-v1-09-final-probe-context-import-repair-design.md` 与 `docs/superpowers/plans/2026-08-03-v1-09-final-probe-context-import-repair-plan.md`。
- online evidence retention：`c9b0fcf944644fcc37c0029b0c1142b863122ac0` / `ded79476c40000978cb267ebdd037af8cf573356`，分别为 `docs/superpowers/specs/2026-08-04-v1-09-online-evidence-retention-repair-design.md` 与 `docs/superpowers/plans/2026-08-04-v1-09-online-evidence-retention-repair-plan.md`。
- catalog regression：`9d862425bd62230e671e05f3ce314d1a79a89464` / `a7d1a7ba05adb79808c8e6b6ce0e8174c79b9ab4`，分别为 `docs/superpowers/specs/2026-08-04-v1-09-catalog-regression-repair-design.md` 与 `docs/superpowers/plans/2026-08-04-v1-09-catalog-regression-repair-plan.md`。

## 4. 协调端 Task 11 在线事实

以下事实来自已保留的唯一在线目录 `artifacts\\vision_lab\\v1-09-task11-evidence-retention-final-online`；本 Task 12 未再次运行在线命令：

- 协调端独立复核结论：`P0=0, P1=0, P2=0`。
- JUnit：`tests=1, failures=0, errors=0, skipped=0`；wrapper `runtime_evidence` step 为 `PASS`。
- wrapper 原生保留：`1 run / 38 files / 1 bundle / 12 probes / 15 PNG`。
- 六个 decision：`qualified / missing / hole / foreign / broken / dimension`。
- `final6/6=true`、`same_run=true`、`home=true`、`tool_off=true`、`safety=0`。
- 进程所有权：PID `51640`，路径 `E:\CoppeliaSim\coppeliaSim.exe`，start ticks `639213715088667526`。
- 结束清理：23010 listener `0`，CoppeliaSim process `0`。
- 这是真实在线 PASS 的唯一采用证据；本报告任务没有重新启动 CoppeliaSim 或复用端口 23010。

## 5. 场景、资产与受保护文件

- 新场景：`simulation/vision_defect_sorting_lab/BL23_vision_defect_sorting_lab.ttt`，SHA256 `b86e8f565ff28251f2ec0b6c88bc40c3b7908c991509917e699545c7db58a18`。
- V1-09 scene manifest SHA256：`53da55dd9505794fede62e5e4657dfe47441818f8311a3c9e24220be6b28f907`。
- V1-09 defect asset manifest SHA256：`96799f00b4ffee96dc0d090977269fd3730c061897fd127a4f2b3f84dbd95282`。
- V1-09 definition SHA256：`4887e5cc07b339b2cfc765c1a1ee918b0fba9823ee8064f51708cc6398c56a52`。
- V1-09 profiles SHA256：`3dcb02b240659ae83b455a156b929c4e45604cdf1dc7aa7858772e7871956406`。

七个既有正式 `.ttt` 在 base/head 的 Git blob 完全相同；以下为 Lane 2 取得的文件 SHA256：

| 文件 | base/head blob 相同 | SHA256 |
| --- | --- | --- |
| `robot_backends/models/BLX_openr6.ttt` | 是 | `a850f5ab06d964db286d73871cd375788eab96bd300890e0b340832842b5b6d4` |
| `simulation/logistics_lab/BL23_logistics_lab.ttt` | 是 | `8e6d381de7208dd4a4a9d9f10428a7d693c3eabc91498ba6911fbae642e29725` |
| `simulation/robot_basics/BL23_robot_basics.ttt` | 是 | `9a2ce252e3f61f08c7498e2cdaf682bfe5255c99d8871dc242b0255555413ea4` |
| `simulation/vision_code_routing_lab/BL23_vision_code_routing_lab.ttt` | 是 | `90dd6e744343102a9bdbe36b497676381ca4562f3b9fc5923b59f228c45598d6` |
| `simulation/vision_lab/BL23_vision_lab.ttt` | 是 | `5154dfc3132cd29aafcbe5de83c1a6b61960595681ac5c90d2d92a5b642bda55` |
| `simulation/vision_ocr_sorting_lab/BL23_vision_ocr_sorting_lab.ttt` | 是 | `3bd1cb20d78be81f5f1358d8601e384a5cd07ab346062d55e120e031c3b993c1` |
| `simulation/vision_quality_lab/BL23_vision_quality_lab.ttt` | 是 | `ce4c0189bb486caa2bfbd557d0b5148775d846a80e07983423f1b29d8d9eb310` |

## 6. Task 12 Lane 1/2 静态事实

### Lane 1（detached clean HEAD `9e2b950`）

- catalog 三测试：`3 passed`，0 skipped，耗时 `0.41s`（wall `0.910s`）。
- affected/canonical：`86 passed`，0 skipped，耗时 `0.66s`（wall `1.193s`）。
- V1-09 focused：`125 passed, 1 skipped, 1 deselected`，耗时 `3.29s`（wall `4.016s`）；skip 原因是 Task 3 asset manifest 已集成，deselected 是 online marker。
- protected defect kernel：`18 passed`，0 skipped，耗时 `0.28s`（wall `0.845s`）。
- V1-08 回归：`172 passed, 1 skipped`，耗时 `12.40s`（wall `13.211s`）；skip 原因是 Windows worker 不提供符号链接。
- 正确的 `git diff --check base..HEAD`：exit `0`、无输出；补证时间窗 `01:12:19.784–01:12:19.994`（`0.211s`）。第一次 exit `129` 是 range 被错误拆参的命令构造缺口，不是候选测试失败；随后已用正确命令复核。

### Lane 2（detached clean HEAD `9e2b950`）

- delivery/readiness/V1-09 non-online：`85 passed, 1 deselected, 0 skipped`，耗时 `0.50s`。
- delivery contract：`48 passed, 0 skipped`，耗时 `0.34s`。
- `git diff --check` exit `0`。
- placeholder `rg` 无匹配，exit `1` 为预期无匹配结果。
- `RETAINED_FILES.txt` required `41`、missing retained `0`、missing files `0`；报告路径追加一次。
- repair commit 相对父提交为精确三路径；denylist 为空。
- 65 个 changed text 文件严格 UTF-8：errors `0`、BOM `0`、NUL `0`；index 为 LF。Windows working tree 的 60 个 CRLF 是 checkout 表现，5 个 scene JSON 由 attributes 保持 LF；PNG/TTT 按二进制单列。
- 七个 protected `.ttt` blob/hash 不变，新场景 SHA 正确，HEAD/status 正确。

### 协调端全仓基线回归

- 主实施工作树 fresh 全仓：`2351 passed, 23 skipped, 0 failures/errors`，总耗时 `91.87s`。
- 23 个 skip 单列：20 个 opt-in CoppeliaSim、1 个 Windows symlink、1 个 Task 3 asset integrated、1 个 UI evidence；这些均不是在线 PASS。

## 7. 变更边界与未决项

- 本轮 worker 只新增本报告并追加 `RETAINED_FILES.txt`；没有修改、重建或复制任何生产实现、测试、配置、PNG、scene、manifest、正式 `.ttt`、URDF/STL/mesh/robot asset。
- 没有 merge main、没有 tag/release、没有在本任务重新在线运行。
- 下一轮 backlog：“实验区离基座过近”；不属于当前 V1-09 交付。

PENDING_HARDWARE: real robot, Hikvision MVS, emergency stop, pneumatics, physical grasp

PENDING_HUMAN_ACCEPTANCE: classroom explanation, student understanding, teaching usability
