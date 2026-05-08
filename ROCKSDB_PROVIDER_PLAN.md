# RocksDB Graph Provider —— 渐进式集成 plan

## 进度（2026-05-08 更新）

| Phase | 状态 | 实际产出 |
|---|---|---|
| Phase 1 最小可跑 | ✅ 完成 | commit `a63d072f`；11/11 测试通过；CI gate 全绿 |
| Phase 2 持久化 | ⏸ Hold | 用户决定先做共享提取再回来加 save/load |
| Phase 3 函数级共享提取 | ✅ 完成 | 共享 `kv_codec/` 模块（neighbor/vector/quant）+ 13 个 codec 单元测试；bf_tree 18/18 + rocksdb 11/11 测试通过 |
| Phase 4 trait 化 | ⏳ 待评估 | Phase 3 完成后看剩余差异 |

### Phase 3 实际成果（2026-05-08）

抽出 `provider/async_/kv_codec/` 子模块（feature gate `bf_tree | rocksdb_provider`）：

| 共享模块 | 内容 | LOC |
|---|---|---|
| `kv_codec/neighbor.rs` | `serialize<I>` / `validate_and_count<I>` —— `\|VectorId\|...\|len\|` layout 5 个 validation 规则 | 143（含 6 个测试） |
| `kv_codec/vector.rs` | `validate_set` / `validate_read_size` —— full-precision 向量 dim/bounds/size 校验 | 80（含 5 个测试） |
| `kv_codec/quant.rs` | `validate_set` / `validate_set_quant` / `validate_read_size` —— PQ 量化向量校验 | 85（含 3 个测试） |

收益：5 条 neighbor list 验证规则、3 条 vector 验证、3 条 quant 验证现在
**只在一个地方维护**；两个 backend 的 `*_provider.rs` 现在差异收敛到
backend specific 的 read / insert / delete 调用 + 错误模型映射。

注意：plan 原期望"每个 backend 文件 < 100 LOC"过于乐观——大部分文件
LOC 来自单元测试，没抽掉。实际算法层（非测试）抽出去了 ~150 LOC 重复。
测试 fixtures 抽提是一个未来可以做的小步骤，但不在 Phase 3 范围内。

Phase 1 的真实差异面：4310 LOC 的 bf_tree 目录里只有约 200 LOC 是
BfTree-specific（< 5%），其余 95% 是图算法 + accessor + DataProvider trait
impl。复制一份 RocksdbProvider 后机械替换 + 删除 SaveWith/LoadWith 部分，
最终 rocksdb/ 是 ~3300 LOC（其中 provider.rs 已经从 3098 删到 2066）。

## 背景

目标：把 RocksDB 接入 DiskANN 图索引，作为 vector + neighbors 存储后端，
平行 `BfTreeProvider`。

约束：
- `BfTreeProvider`（4310 LOC）里真正 BfTree-specific 的代码只占 < 5%（200-300 LOC），
  其余是图算法 + accessor + DataProvider trait impl，对存储后端不可见
- 一次性做"trait 抽象 + 内层泛型化"侵入式重构成本太高（5-6 天，6 个 PR）
- 平行复制完整 bf_tree/ → rocksdb/ 短期可行，后续维护重复成本可接受
  前提是后续 Phase 提取共享代码

## 阶段切分

### Phase 1（本次）：最小可跑 RocksdbProvider，接受临时重复

**Scope**：
- 平行复制 `diskann-providers/src/model/graph/provider/async_/bf_tree/` →
  `.../rocksdb/`，5 个文件全部复制
- 机械替换 BfTree-specific API：
  - `BfTree::with_config(config, None)` → `DB::open(opts, path)`
  - `tree.read(key, &mut buf) -> LeafReadResult` → `db.get(key) -> Option<Vec<u8>>`
  - `tree.insert(key, value) -> LeafInsertResult` → `db.put(key, value) -> Result`
  - `tree.delete(key)` → `db.delete(key)`
  - `bf_tree::StorageBackend::{Memory, Std}` → RocksDB 用 tempdir 模拟 Memory
  - `tree.snapshot()` / `snapshot_memory_to_disk` → `Checkpoint::create_checkpoint`
  - `BfTree::new_from_snapshot` / `new_from_snapshot_disk_to_memory` → `DB::open(snapshot_path)`
- 顶层 `BfTreeProvider` 不重构，**复制一份成 `RocksdbProvider`**，type alias 平行
- `RocksdbProviderParameters`、`RocksdbParams`、`RocksdbPaths` 等类型名平行命名
- feature flag：`rocksdb_provider`（与 `bf_tree` 平行）
- in-memory 模式：rocksdb 路径只支持 disk；测试用 `tempfile::TempDir` 替代 Memory backend

**验收**：
- `cargo check --features rocksdb_provider` 通过
- `cargo test -p diskann-providers --features rocksdb_provider` 通过，
  包括复制过来的 6 个 save/load 测试（用 tempdir 替代 in-memory 路径）
- 默认 build（不开 feature）不破坏

**不做**：
- 不抽 trait（KvBackend、GraphStorage 一律推迟）
- 不做内层泛型化（VectorProvider<T> 不改成 VectorProvider<T, S>）
- 不动 `BfTreeProvider`、`bf_tree/` 目录

**预期**：
- 新增 ~4500 LOC（机械复制 + 适配）
- 大部分是模板化复制，少量是真改动

### Phase 2：持久化（HOLD）

原计划：让 RocksdbProvider 走 RocksDB Checkpoint API，平行 6 个 save/load
测试。**用户决定先做 Phase 3 共享提取**——代码先收敛再加新 feature，
避免 codec 抽出来时 save/load 还在重复维护一遍。

Phase 1 已经把 `RocksdbParams`、`QuantParams`、`SavedParams`、`RocksdbPaths`
保留下来，仅用 `#![allow(dead_code)]` 抑制告警，等 Phase 2 直接接回去。

### Phase 3（本次）：函数级共享提取

扫两份代码找重复：
- `vector_provider.rs` 两份的 `key 序列化` `value layout` `fill()` `set_element()`
  算法层 → 抽到 `shared/vector_kv_codec.rs`
- `neighbor_provider.rs` 两份的 adjacency list layout
  （`|VectorId|...|Invalid|...|len|`） → 抽到 `shared/neighbor_kv_codec.rs`
- `quant_vector_provider.rs` 两份的 PQ codec → 抽到 `shared/quant_kv_codec.rs`

预期每个 backend-specific 文件能压到 < 150 LOC，只剩"调谁的
read/insert/delete"差异。

**Phase 3 不做**：
- 不引入 trait（继续留到 Phase 4 看 pattern）
- 不动 backend specific I/O 调用
- 不动 DataProvider trait impl 这一层（这一层和后端无关，已经是共享的）

### Phase 4（可选）：trait 化 / 顶层泛型化

看 Phase 3 抽完后剩什么——如果两后端方法签名 100% 一致，加个
`trait KvBackend` 统一，顶层 `BfTreeProvider` / `RocksdbProvider` 合并为
`GraphProvider<T, S, ...>`；如果还有差异（zero-alloc read），保持泛型 +
结构性约束。

## 关键设计决策（Phase 1 已敲定）

1. **不重构 BfTreeProvider**：保持现状，rocksdb 路径独立平行
2. **不引入 trait**：留到 Phase 4
3. **rocksdb 路径不支持 in-memory backend**：测试用 tempdir 模拟
4. **diskann-providers 直接 use rocksdb crate**：不通过 label-filter 的 RocksdbStore
   （RocksdbStore 是 KvStore trait 实现，给 inverted index 用；graph storage 这层
   用 RocksDB 直接 API 更顺）

## 已知 mismatch 处理

| BfTree | RocksDB | Phase 1 处理 |
|---|---|---|
| `read(key, &mut buf)` zero-alloc | `db.get(key) -> Vec<u8>` 有 alloc | 接受 alloc，Phase 3 看是否瓶颈 |
| `LeafReadResult::{Found, NotFound, Deleted, InvalidKey}` | `Result<Option<Vec<u8>>>` | match 改成 `match db.get(key) { Ok(Some) / Ok(None) / Err }`；InvalidKey 在 RocksDB 没等价物，调用方已经 validate key 长度 |
| `LeafInsertResult::{Success, InvalidKV}` | `Result<()>` | 同上 |
| `StorageBackend::{Memory, Std}` | 没有 in-memory | 测试用 tempdir |
| `tree.snapshot()` / `snapshot_memory_to_disk()` | `Checkpoint::create_checkpoint(path)` | 改写 save_bftree → save_rocksdb |
| `BfTree::new_from_snapshot` | `DB::open(path)` | 一个 disk path 直接打开 |

## 工作量估计

- 复制 + 机械替换：1 天
- BfTree-specific API 适配：0.5 天
- 6 个 save/load 测试 port：0.5 天
- cargo check / clippy / test 调试：0.5 天
- 合计 Phase 1：~2-3 天

## CI / Build 注意事项

- `rocksdb_provider` feature 是 opt-in，**不进** CI 默认 `DISKANN_FEATURES`
  矩阵（避免强制 libclang 依赖）
- 单独的 CI job 跑 rocksdb_provider feature，需要镜像装 libclang
- Windows 本地：`$env:LIBCLANG_PATH = "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Tools\Llvm\x64\bin"`
