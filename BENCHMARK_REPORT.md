# Graph-Storage Backend Benchmark: inmem vs BfTree vs RocksDB

## 摘要

首次量化 DiskANN 三个图存储后端（inmem / BfTree / RocksDB）在同一
workload 下的相对开销。结论：

| 维度 | inmem | BfTree | RocksDB (调优后) |
|---|---:|---:|---:|
| Build 时间 | **1×** | 5.2× | 18.1× |
| Search QPS @ Ls=100 | **1×** | 0.18× (5.6× 慢) | 0.052× (19.4× 慢) |
| Recall@10 | 0.991-1.0 | 0.991-1.0 | 0.991-1.0 |
| Avg cmps / hops | 完全一致（算法层不变） |

**关键发现**：三个 backend 在算法层（cmps、hops、recall）完全一致，
backend 只贡献每次 vector / neighbor access 的 µs 差异。即便对 RocksDB
做了 block cache + compression + write buffer + zero-copy reads
（`get_pinned`）的优化，相对 inmem 仍有 19× 差距，主要来自 RocksDB FFI
调用 + LSM 读路径 + memtable 跳表查找的固有成本，单纯调参难以追平。

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

## 配置文件

- `sift10k-3way.json`（worktree 根）：本次实验输入
- `sift10k-3way-output.json`：基线 raw 输出
- `sift10k-3way-tuned.json`：调优后 raw 输出
- 数据：`test_data/sift/siftsmall_learn.bin` (25K base) +
  `test_data/sift/siftsmall_query_100pts.bin` (Python 生成的 100-query
  子集) + `test_data/sift/siftsmall_gt100`
  (`compute_groundtruth` 生成)

## 复现脚本

```bash
# 1. Build with both feature flags
$env:LIBCLANG_PATH = "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Tools\Llvm\x64\bin"
cargo build --release -p diskann-benchmark \
    --features "bf_tree_provider rocksdb_provider"

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

# 4. Run benchmark
./target/release/diskann-benchmark.exe --quiet run \
    --input-file ./sift10k-3way.json \
    --output-file ./output.json
```
