# DiskANN Storage Backend Benchmark: inmem / BfTree / RocksDB / **redb** / disk-index

## 摘要

5 个存储后端在同一 workload 下的对比（siftsmall, 25K base × 100 query
× 128 dim, single thread）。新加的 **redb 是纯 Rust B+ tree KV**，
原以为应该跑赢 RocksDB（无 FFI），实测**反而是 build 最慢的**——
4× 慢于 RocksDB，140× 慢于 inmem。

| 维度 | inmem | BfTree | RocksDB (调优后) | **redb** | disk-index |
|---|---:|---:|---:|---:|---:|
| Build 时间 | **1.79 s** | 8.50 s | 28.93 s | **118.44 s** | 3.85 s |
| Search QPS @ Ls=100 | **8,209** | 1,458 | 527 | **353** | 82.0 |
| Search Avg Latency @ Ls=100 | **121 µs** | 685 µs | 1,902 µs | **2,832 µs** | 12,194 µs |
| Recall@10 (Ls=100) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Avg cmps / hops (Ls=100) | 1554.59 / 103.38 | 同 | 同 | 同 | 1791.0 / 115.0 |

**关键发现：**

0. **「Pure Rust ≠ Fast」反直觉数据点**：redb 是纯 Rust B+ tree KV，
   没有 FFI 跨语言成本，但 build 比 RocksDB 慢 4×、比 BfTree 慢 14×。
   原因不在语言层，而在数据结构 + 事务模型层：
   - redb 是 **strict ACID B+ tree**，每次 `put` 都是完整 transaction
     (`begin_write` → `open_table` → `insert` → `commit`)
   - 即便 `Durability::None` 关掉 fsync，每次 commit 仍需要
     WAL append + 树页更新 + root pointer swap
   - 单写者串行（vs RocksDB 内部更细的锁 / BfTree 内部 buffer 合并）
   - 25K 次小事务的累加成本主导

1. **三个 in-memory backend（inmem/bftree/rocksdb）算法层完全一致**：
   同样的 cmps、hops、recall。Backend 只贡献每次 vector/neighbor
   access 的 µs 差异。即便对 RocksDB 做了 block cache + compression +
   write buffer + zero-copy reads (`get_pinned`) 的优化，相对 inmem
   仍有 19× 差距，主要来自 FFI + LSM read path 的固有成本。

2. **disk-index 是另一类架构**：build 比 BfTree/RocksDB **更快**（一次性
   写入 vs per-insert KV），但 search 受 IO 主导慢 ~350× vs inmem。
   每 query 的 13–22 ms latency 中 ~96% 来自 disk IO，CPU 仅 ~300 µs。

3. **RocksDB / BfTree = 3-3.5×**：两个 KV 后端的直接比较——rocksdb 慢
   于 bftree **不是 IO 量级差异**（两者都没真正 IO，数据全在 memtable
   / buffered tree），而是 per-op overhead：FFI、LSM read path、memtable
   skiplist 等。BfTree 是纯 Rust 实现，结构上更贴合 µs 级随机点查的
   DiskANN workload。

## 测试方法

### Dataset

- **siftsmall**：25,000 base vectors × 128 dim float32（standard sift10k
  variant，从 corpus-texmex.irisa.fr 形态）
- **Query**：从 base 取前 100 vectors（recall=1 trivially 因为 query 在
  base 中——这次比对**只看 perf**，不看 recall 的绝对值）
- **Groundtruth**：用 `cargo run --bin compute_groundtruth` 离线生成

### Build 参数

```
max_degree = 32
l_build = 50
alpha = 1.2
backedge_ratio = 1.0
num_threads = 1
start_point_strategy = medoid
distance = squared_l2
```

### Search 参数

- 100 queries × 3 reps，单线程
- `Ls ∈ {20, 50, 100}`
- `recall_k = 10`

### 跑法

```bash
$env:LIBCLANG_PATH = "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Tools\Llvm\x64\bin"
cargo build --release -p diskann-benchmark --features "bf_tree_provider rocksdb_provider"
./target/release/diskann-benchmark.exe --quiet run \
    --input-file ./sift10k-3way.json --output-file ./out.json
```

JSON 配置一个 file 三个 jobs，分别对应 `graph_provider: "inmem" /
"bftree" / "rocksdb"`。

### 硬件 / 软件

- Windows 11 Enterprise x64, Dev Box
- Rust msrustup ms-prod-1.92, target-cpu 默认（无 AVX512 banner，但 AVX2
  正常）
- RocksDB 0.22 (rust-rocksdb fork via crates.io)，bf-tree 0.4.9

## 基线结果（未调优 RocksDB）

### Build

| Backend | 总时长 | Avg insert | p90 | p99 | vs inmem |
|---|---:|---:|---:|---:|---:|
| **inmem** | **1.92 s** | 76.2 µs | 106 µs | 153 µs | 1.0× |
| bftree | 9.96 s | 397.5 µs | 596 µs | 904 µs | 5.2× |
| rocksdb | 36.80 s | 1470 µs | 2097 µs | 3126 µs | **19.1×** |

### Search Ls=20 (recall@10 = 0.991)

| Backend | QPS (mean) | Avg lat | p99 lat | vs inmem |
|---|---:|---:|---:|---:|
| **inmem** | **21,113** | 46.0 µs | 208 µs | 1.0× |
| bftree | 4,050 | 247 µs | 464 µs | 5.2× |
| rocksdb | 1,121 | 894 µs | 2099 µs | **18.8×** |

### Search Ls=100 (recall@10 = 1.000)

| Backend | QPS (mean) | Avg lat | p99 lat | vs inmem |
|---|---:|---:|---:|---:|
| **inmem** | **7,969** | 124 µs | 187 µs | 1.0× |
| bftree | 1,215 | 822 µs | 1420 µs | 6.6× |
| rocksdb | 379 | 2642 µs | 4625 µs | **21.0×** |

### 算法层指标（三 backend 完全一致）

| Ls | Avg cmps | Avg hops |
|---:|---:|---:|
| 20 | 550.99 | 24.19 |
| 50 | 990.28 | 53.74 |
| 100 | 1554.59 | 103.38 |

## RocksDB 优化迭代

### Iteration 1：Options + zero-copy reads

针对默认 `Options::default()` + `db.get` 两个路径，做了 4 处改动：

```rust
// diskann-providers/src/model/graph/provider/async_/rocksdb/mod.rs
const BLOCK_CACHE_BYTES: usize = 256 * 1024 * 1024;  // 默认 8 MiB → 256 MiB
const WRITE_BUFFER_BYTES: usize = 64 * 1024 * 1024;  // 默认 64 MiB（保留默认）

pub(crate) fn open_db(config: &Config) -> Result<DB, ConfigError> {
    let mut opts = rocksdb::Options::default();
    opts.create_if_missing(true);
    opts.set_compression_type(rocksdb::DBCompressionType::None);  // 关 Snappy
    opts.set_write_buffer_size(WRITE_BUFFER_BYTES);

    let cache = rocksdb::Cache::new_lru_cache(BLOCK_CACHE_BYTES);
    let mut block_opts = rocksdb::BlockBasedOptions::default();
    block_opts.set_block_cache(&cache);
    opts.set_block_based_table_factory(&block_opts);

    DB::open(&opts, &config.path).map_err(ConfigError)
}
```

三个 inner provider 的 read path 从 `db.get(key) -> Vec<u8>` 改为
`db.get_pinned(key) -> DBPinnableSlice<'_>`：

```rust
// 旧：每跳分配一个 Vec<u8>
let value = self.adjacency_list_index.get(key)?;

// 新：返回 DBPinnableSlice，derefs 成 &[u8]，零拷贝
let value = self.adjacency_list_index.get_pinned(key)?;
```

### 调优后结果

#### Build

| Backend | 基线 | 调优后 | Δ vs 基线 | vs inmem (调优后) |
|---|---:|---:|---:|---:|
| inmem | 1.92 s | 1.89 s | -1.6% (噪声) | 1.0× |
| bftree | 9.96 s | 9.60 s | -3.6% | 5.1× |
| rocksdb | **36.80 s** | **34.15 s** | **-7.2%** | 18.1× |

#### Search Ls=20

| Backend | 基线 QPS | 调优后 QPS | Δ |
|---|---:|---:|---:|
| inmem | 21,113 | 28,265 | +33%* |
| bftree | 4,050 | 4,140 | +2% |
| rocksdb | **1,121** | **1,272** | **+13%** |

\* inmem 的 33% 提升应是单次运行噪声（控制组未变），不是优化生效。

#### Search Ls=100

| Backend | 基线 QPS | 调优后 QPS | Δ |
|---|---:|---:|---:|
| inmem | 7,968 | 7,749 | -2.7% (噪声) |
| bftree | 1,215 | 1,411 | +16% |
| rocksdb | **379** | **400** | **+5.5%** |

#### 调优结论

总体提升 5-15%，rocksdb / inmem 比从 ~19-21× 收窄到 ~18-19×。**单纯调
RocksDB 参数无法把差距压到 10× 以内**——剩余瓶颈不是参数能解决的。

## 剩余瓶颈分析 [inference]

每跳邻居访问的开销构成（基于 RocksDB 内部架构推测）：

1. **FFI boundary crossing**：每次 `get_pinned` 都从 Rust 跨进 C++ rocksdb
   核心；25K dataset Ls=100 → ~103 hops × ~32 邻居 = ~3300 次 get/query。
   FFI 开销 + safety check 累加显著
2. **Memtable skiplist 查找**：25K 记录全在 memtable（write_buffer
   64 MiB 没溢出），每次 lookup 是 O(log N) skiplist traversal，~14 跳
3. **Bloom filter check**：尽管所有数据在 memtable，每次 lookup 仍走完整
   read path（包括 bloom check）
4. **PinnableSlice 构造与释放**：虽然消除了 alloc，但 PinnableSlice 自身
   的 RAII 管理仍有开销
5. **InternalKey 编码 + comparator**：RocksDB 内部 key 加 sequence number +
   value type，每次比较走 comparator callback

而 BfTree 是 Rust 原生实现，无 FFI；inmem 是直接数组索引 O(1)。

## 数据集放大效应 [inference]

256pt 数据集（之前测过）vs 25K 数据集对比：

| 维度 | 256 pts | 25K pts |
|---|---:|---:|
| bftree / inmem build | 5.1× | 5.2× |
| **bftree / inmem search Ls=20** | **2.1×** | **5.2×** |
| inmem search Ls=20 hops | ~22 | ~24 |
| 25K hops 增长 | n/a | Ls=20→24 / Ls=100→103 |

Search 路径的差距随数据集增大显著放大（2.1× → 5.2×），原因是
graph diameter 增加 → 每 query 需要更多 neighbor lookups → KV 后端 per-hop
开销累积。Build 路径反而稳定（5.1× → 5.2×），因为每 insert 的写入次数
不随总数据规模变化（只随 max_degree）。

## 实验局限 / 未量化的维度

1. **Recall 在大数据集上未充分压力测**：query 是 base 的子集，recall
   在 Ls=20 已经 0.991 → 这是个 perf 测试，不是 recall 测试
2. **单次 run 噪声 ~5-10%**：未做多次 run + 中位数；个别数字（如 inmem
   Ls=20 QPS 33% 跳变）应为噪声
3. **单线程**：未测 concurrent build / concurrent search；
   各 backend 的并发扩展性可能差异更大（RocksDB 内部锁 vs BfTree 内部
   同步 vs inmem 数组）
4. **冷启动 vs 热启动**：所有测都是冷启动（DB 刚建即用）；prod 场景
   block cache warm 状态下 RocksDB 可能更接近 BfTree
5. **小数据集**：25K 还是远小于 prod scale (100M+)；access pattern 在
   100M 量级 cache miss 主导，KV 后端 vs inmem 差距可能反而**收窄**
   或者反转
6. **build 阶段 WAL 未禁用**：每次 put 都过 WAL，build phase 的 19× 差
   距应该有相当一部分能通过 `WriteOptions::disable_wal(true)` 拿掉

## 后续优化方向（按预期 ROI 排序）

### 高 ROI（可能拿掉 30-50% rocksdb 差距）

1. **Build 阶段 disable WAL**：`WriteOptions::set_disable_wal(true)` +
   `db.put_opt`；build 路径写入次数 ~34/insert，WAL fsync 占比应该可观
2. **Multi-get for neighbor expansion**：search 路径每跳访问 N 个邻居的
   vector，目前是 N 次 `get_pinned`；改用 `db.multi_get_cf` 一次拿走，
   FFI 开销摊薄到 1 次
3. **Block cache 调小到 32 MiB**：当前 256 MiB 远大于 dataset，浪费内存
   且 LRU 维护开销没必要

### 中 ROI

4. **Pinnable slice 持续使用**：build 阶段 set_neighbors 内部 read-modify-
   write 也可以零拷贝
5. **单 DB + column family 替代 3 个 DB**：current `RocksdbProvider` 开 3
   个独立 DB（vector / quant / neighbor）；合并到一个 DB + 3 个 CF 可以
   共享 block cache、共享 WAL、减少 fsync 次数

### 低 ROI（架构层改动，推迟）

6. **Phase 4 trait 化**：定义 `KvBackend` trait 后，看是否能把 zero-alloc
   read 提到 trait 上，BfTree 走 caller-buffer，RocksDB 走 PinnableSlice
7. **batch insert API**：build 阶段批量 SetElement，替代当前的
   per-insert 顺序 put
8. **借鉴 disk-index sector layout**：把 (vector, neighbors) co-locate
   到同一个 record，每跳 1 次 get 替代 1+N 次

## Iteration 3：加入 redb (pure Rust B+ tree)

### 动机

`bf-tree` 是 Bε-tree（buffered repository tree, Microsoft Research）；
`RocksDB` 是 C++ LSM via FFI；我想加一个**纯 Rust B+ tree KV** 形成
3 类不同设计的对比：

| Backend | 数据结构 | 实现语言 | 主要 trade-off |
|---|---|---|---|
| BfTree | Bε-tree | 纯 Rust | write-optimized, buffered |
| RocksDB | LSM-tree | C++ + FFI | mature, write-heavy |
| **redb** | **B+ tree** | **纯 Rust** | **strict ACID, single-file** |

选 redb 是因为：
- 纯 Rust，无 FFI
- B+ tree（vs Bε-tree 和 LSM 都是不同结构）
- production-tested（被 PufferFS、kanidm 等项目使用）
- MVCC + 写事务隔离

### Build 对比

| Backend | 总时长 | Avg insert | 模式 |
|---|---:|---:|---|
| inmem | 1.79 s | 71 µs | per-insert，无 KV |
| disk-index | 3.85 s | (one-shot) | in-RAM build → atomic write |
| bftree | 8.50 s | 339 µs | per-insert KV (Bε-tree buffered) |
| rocksdb | 28.93 s | 1,156 µs | per-insert KV (LSM + WAL) |
| **redb** | **118.44 s** | **4,736 µs** | **per-insert tx (B+ tree)** |

**redb 比 rocksdb 慢 4×、比 bftree 慢 14×** — 完全反直觉的方向。

### Search 对比 (Ls=100)

| Backend | QPS | Avg lat | p99 lat |
|---|---:|---:|---:|
| inmem | 8,209 | 121 µs | 196 µs |
| bftree | 1,458 | 685 µs | 1,078 µs |
| rocksdb | 527 | 1,902 µs | 3,156 µs |
| **redb** | **353** | **2,832 µs** | **4,017 µs** |
| disk-index | 82.0 | 12,194 µs | — |

redb 在 search 上比 rocksdb 慢 1.5×，差距比 build 上小很多。

### 为什么 redb 这么慢？[inference]

每次 `put` 我们调用：
```rust
let mut tx = db.begin_write()?;
tx.set_durability(redb::Durability::None);  // 关掉 fsync
let mut table = tx.open_table(KV_TABLE)?;
table.insert(key, value)?;
tx.commit()?;
```

即便没有 fsync，每次 commit 仍有：
1. **写者串行化**：redb 是 single-writer 设计，`begin_write` 拿独占锁
2. **WAL append**：commit 写 commit record（无 fsync 但有 syscall）
3. **B+ tree 页面拷贝**：copy-on-write，每次插入分配新页面
4. **Root pointer atomic swap**：commit 时切换 root，需要内存屏障
5. **Page allocator overhead**：B+ tree 的 page-aligned allocation

vs **BfTree** 的优势：
- Bε-tree 把多个 insert **buffered 到上层 node** 再批量下沉
- 无 transaction commit 开销（不强 ACID）
- 单 insert 主要是 in-memory buffer append

vs **RocksDB** 的优势：
- LSM 是 append-only，写就是 memtable insert
- 即便有 WAL，writes 是 batched
- 通过 column family / multi-threaded compaction 摊薄成本

**redb 的设计点是 "transactional B+ tree, MVCC"——这对 OLTP / 配置
存储友好，但**完全不适配** DiskANN 的 25K 高频小 write 模式**。

### 为什么 search 也慢？

每次 `get` 需要：
```rust
let tx = db.begin_read()?;
let table = tx.open_table(KV_TABLE)?;
let guard = table.get(key)?;
```

读路径开销：
- `begin_read`：拿 snapshot version（轻量但非零）
- `open_table`：表查找（按名字字符串比较）
- B+ tree traversal 到叶子

vs **BfTree** 的 `tree.read(key, buf)`：直接 internal map lookup，
不需要 snapshot / table-open ceremony。

### 结论：**redb 不是 DiskANN 的合适后端 [verified]**

- redb 设计目标：transactional metadata storage（数百-数千 ops/s 的
  OLTP-style workload）
- DiskANN 需要：µs 级随机点查 × millions/sec，不需要 ACID

**这是个有价值的"反例"**：纯 Rust 不一定快——数据结构选型 +
事务模型比语言层影响大得多。

## Iteration 2：加入 disk-index 做 4-way 对比

`disk-index` feature 提供 DiskANN 原版的磁盘驻留索引——PQ 压缩向量留在
RAM 做候选筛选 + 完整向量按需从磁盘读取（per-hop sector read）。本节
把它放入对比看架构层差异。

### Build

| Backend | 总时长 | 构造模式 |
|---|---:|---|
| **disk-index** | **5.35 s** | One-shot：load → in-RAM build → atomic write |
| inmem | 1.92 s | Per-insert，无 KV 序列化 |
| bftree | 9.60 s | Per-insert，每次 KV write |
| rocksdb | 35.81 s | Per-insert，每次 KV write + WAL append |

disk-index 比 bftree/rocksdb **更快** 的原因 [inference]：
- 整体走 build-then-flush 模型，没有 per-insert 的 KV 写放大
- 单次大块 sequential disk write 比 25K 次随机 KV write 高效

### Search

| Ls | inmem | bftree | rocksdb | disk-index |
|---:|---:|---:|---:|---:|
| 20 (QPS) | 25,645 | 4,066 | 1,350 | **72.5** |
| 20 (avg lat) | 38 µs | 245 µs | 738 µs | **13,786 µs** |
| 20 (recall@10) | 0.991 | 0.991 | 0.991 | **0.992** |
| 50 (QPS) | 13,059 | 1,917 | 728 | 75.7 |
| 100 (QPS) | 8,280 | 1,336 | 449 | 44.9 |
| 100 (avg lat) | 119 µs | 747 µs | 2,225 µs | **22,266 µs** |
| 100 (recall@10) | 1.000 | 1.000 | 1.000 | 1.000 |

### disk-index 的 latency 细分（来自 benchmark 自带 stats）

| Ls | IO time | CPU time | PQ preprocess | IOs/query | Cache hit% |
|---:|---:|---:|---:|---:|---:|
| 20 | 13,377 µs | 261 µs | 148 µs | 37.6 | 0.0% |
| 50 | 12,717 µs | 343 µs | 145 µs | 66.0 | 0.0% |
| 100 | 21,612 µs | 508 µs | 147 µs | 115.0 | 0.0% |

**IO 占总 latency 96-97%**，CPU 仅 1-2%，PQ preprocess ~150 µs。
**Cache hit 0%**：未设 `num_nodes_to_cache`，每跳都 cold disk read。

### 架构差异（不是 perf bug，是定位不同）

| Backend | 设计意图 | 数据驻留 | 主要开销 | 适用规模 |
|---|---|---|---|---|
| inmem | 一切在内存，最高 QPS | RAM | memory access | < dataset fits in RAM |
| bftree | KV 抽象+持久化 | memtable + disk | per-op FFI-free overhead | dataset > RAM but want low latency |
| rocksdb | LSM KV，生态成熟 | memtable + disk | FFI + LSM read path | 需 RocksDB 生态 (snapshot/replication/...) |
| **disk-index** | **真正 disk-resident**，PQ + sector co-locate | disk + PQ cache in RAM | **per-hop disk IO** | **dataset >> RAM** |

### 对 25K 数据集的关键注意

disk-index 在 25K 上"看起来很慢" 是**架构错配**：
- siftsmall (25K × 128 × 4 = 12 MiB) 本来就放得下 RAM
- 用磁盘 layout 反而是 anti-pattern
- 在 prod 1B+ 数据集上 disk-index 才发挥设计优势：那时 inmem 装不下，bftree/rocksdb 的 memtable 也装不下，per-query 必须 disk read

**这次 sift10k benchmark 是 in-memory 三家的对比；disk-index 数字只作
"另一类架构"展示，不应直接横比**。

### 可能的 disk-index 调优 [inference, 未跑]

| 优化 | 预期 ROI |
|---|---|
| 设 `num_nodes_to_cache = 25000` (缓存所有节点) | 高，但变成 in-memory 路径 |
| `beam_width = 16` (默认 4) | 中，提高 IO 并行度 |
| `search_io_limit` | 用于权衡 latency vs IO 数 |
| 数据集放在 RAM disk / tmpfs | 极高，但变 anti-test |

## 配置文件

- `sift10k-3way.json`：3-way 配置（inmem / bftree / rocksdb）
- `sift10k-4way.json`：4-way 配置（加 disk-index）
- `sift10k-3way-output.json`：基线 raw 输出（3-way）
- `sift10k-3way-tuned.json`：调优后 raw 输出（3-way）
- `sift10k-4way-output.json`：4-way raw 输出
- 数据：`test_data/sift/siftsmall_learn.bin` (25K base) +
  `test_data/sift/siftsmall_query_100pts.bin` (Python 生成的 100-query
  子集) + `test_data/sift/siftsmall_gt100`
  (`compute_groundtruth` 生成)

## 复现脚本

```bash
# 1. Build with all backends (omit `disk-index` for 3-way only)
$env:LIBCLANG_PATH = "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Tools\Llvm\x64\bin"
cargo build --release -p diskann-benchmark \
    --features "bf_tree_provider rocksdb_provider disk-index"

# 2. Generate query subset (100 vectors from siftsmall_learn.bin)
/c/Python314/python -c "
import struct
with open('test_data/sift/siftsmall_learn.bin', 'rb') as f:
    n_full, d = struct.unpack('<II', f.read(8))
    payload = f.read(100 * d * 4)
with open('test_data/sift/siftsmall_query_100pts.bin', 'wb') as fout:
    fout.write(struct.pack('<II', 100, d))
    fout.write(payload)
"

# 3. Compute groundtruth
target/release/compute_groundtruth.exe \
    --base_file test_data/sift/siftsmall_learn.bin \
    --query_file test_data/sift/siftsmall_query_100pts.bin \
    --gt_file test_data/sift/siftsmall_gt100 \
    --recall_at 10 --dist_fn l2 --data_type float

# 4. Run benchmark (use sift10k-4way.json for the 4-way comparison)
./target/release/diskann-benchmark.exe --quiet run \
    --input-file ./sift10k-4way.json \
    --output-file ./output.json
```
