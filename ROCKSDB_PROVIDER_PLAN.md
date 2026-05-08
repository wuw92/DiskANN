# RocksDB Graph Provider —— 渐进式集成 plan

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

### Phase 2：积累使用案例

跑一些 build + search 的 smoke test / benchmark，让 rocksdb 路径在真实
workload 下有几次执行。Phase 3 提取时 pattern 才稳定。

### Phase 3：函数级共享提取

Phase 1 + 2 跑通后，扫两份代码找重复：
- `vector_provider.rs` 两份的 `key 序列化` `value layout` `fill()` `set_element()`
  算法层 → 抽到 `shared/vector_kv_codec.rs`
- `neighbor_provider.rs` 两份的 adjacency list layout
  （`|VectorId|...|Invalid|...|len|`） → 抽到 `shared/neighbor_kv_codec.rs`
- `quant_vector_provider.rs` 两份的 PQ codec → 抽到 `shared/quant_kv_codec.rs`

预期每个 backend-specific 文件能压到 < 100 LOC，只剩"调谁的
read/insert/delete"差异。

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
