# 2026-10-28 レビューへの回答（v0.2.0 bfe8725 → v0.3.0）

指摘はすべて、まず手元で再現してから直した。各項目の回帰テストは tests/ にある。R の対照値は、レビューの
`metric_r_oracle.R`（関数ソースを直接読み込む方式）ではなく、インストール済みの lidR 4.3.3 / lidRmetrics 0.1.5
の `metrics_set3` を `validation/ref_lidr.R` と同じ引数で呼んで作った（tests/make_oracle.R、
tests/data/ref/oracle_cases.csv）。レビューの R の対照値（observed/metric_r_oracle_results.csv）とも一致する。

## 優先度の高い4点

| 指摘 | 再現 | 修正 | テスト |
|---|---|---|---|
| ① DTM と格子の位置ずれ（小数画素の window） | 1 m DTM・原点 (0.4, 4.6) で 1 m の誤差を確認 | DTM 自身の画素格子で window を読み、その transform で点の画素を引く（`_dtm_window`） | `test_dtm_on_its_own_pixel_grid`（解像度 0.5 / 1 / 2 m、rasterio の sample と比較） |
| ① 整数 DTM・nodata 未設定の範囲外が 0 m | 確認 | `masked=True` で読み、欠損・範囲外を NaN にして点を除く | `test_integer_dtm_without_nodata_outside_coverage` |
| ② rumple のセル間干渉 | `[1, NaN]` を確認 | 最高点の選択を（格子, 1 m 画素）ごとに（`rumple_pixels` が格子ごとの画素番号を返す） | `test_rumple_fractional_grid_cells_independent`、`test_cells_same_alone_or_together`（原点 0.4、2.5 m、7.3 m の格子で、全体・単独が一致） |
| ③ 地面・低植生セルの欠落、列の不一致 | 確認 | 出力は前処理後に点が1つ以上ある格子。`n_points` を追加し、`n` は lidRmetrics と同じく zmin より上に点がなければ NA。列はオプションだけで決まる（`output_columns`）。木・ギャップは同じ格子表へ左結合 | `test_ground_cells_kept_and_columns_fixed`（trees あり・なし、地面だけ、範囲外） |
| ④ 高さ方向の固定長ビン | `[0.168, 1.0]` と reshape 例外を確認 | エントロピー、LAD、ボクセル、leafR / PyForestScan の層数をブロックの最大高さから決める。合計は左から順に足す（層数が変わっても値が変わらない） | `test_high_canopy_does_not_leak`、oracle の `high_canopy`、`tall_500`（82 m まで） |

## lidR / lidRmetrics との数値・定義

| 指摘 | 修正 |
|---|---|
| ⑤ LAD のビン境界 | `hist(breaks = c(-Inf, seq(z0, top)))` の fuzz（区切り数で 1e-7 × 中央値幅 / 最小幅 / データの範囲）をそのまま再現。oracle の `lad_boundary_up/down`、`lad_small_range`、`near_integers` で一致 |
| ⑥ 同高・少数点・不正 return | zpcum は同高で 0、LAD は n > 2、L モーメントは n ≥ 4、echo は不正 return を除いた分母、`zmin=2` で丸めたボクセルが空でも落ちない。追加で見つけたもの: エントロピーは全点が整数の zmax 上にあると NA、`pz_above_30` は `hist(right = FALSE)` の区切りのずれ（30 − 7.5e-7）と空集合の 0、`n` は空で NA、`lad_max` は全層 NA で −Inf |
| ⑦ KDE のピーク数 | R の `density()` を演算順まで移植（BinDist を点の順に足す、`seq.int`、`dnorm4`、R の FFT〔Singleton 法、長さ 1024 の経路〕、`approx1`）。density の y が R とビット単位で一致（9 ケースで確認）。全バックエンドでホストの float64 で計算するので、バックエンド間の差もない。`[1,2,3,4]` → 1、`[5,5,5,5]` → 2 で R と一致 |

既知の差（残したもの）: 格子内の全点が同じ高さのとき、R の Lmoments は L2 = 0 のため Lskew・Lkurt に丸めの雑音
（NaN、±Inf、−12〜48 の任意の値。長さと値を変えて確認）を返す。本パッケージは NA。テストでは明示した例外として扱う。

## 樹冠処理とメモリ制御

| 指摘 | 修正 | テスト |
|---|---|---|
| ⑧ 凸包の候補削減 | ブロックごとに厳密な凸包（8方向の端点の多角形の内側の点を device で除き、残りの凸包を正確に計算）。全体の凸包はブロックの凸包の頂点から作る | `test_hull_vertices_exact`（円周 80 点の例ほか、分割しても全点の凸包と同じ）、`test_hull_prefilter_drops_interior_points` |
| 2点で QhullError | 点・線分の退化した凸包も扱う | `test_sparse_clouds_with_trees`（1〜3点） |
| 2.5 m セルの spike filter の host / device 差 | 1 m 列を格子の行の境界で区切る定義に統一（整数 m の格子では従来と同じ） | `test_spike_filter_same_on_host_and_device_paths`、`test_device_and_host_paths_fractional_cells`（GPU） |
| ⑨ メモリ予算の下限 | 50 万点・0.5 GiB の隠れた下限を廃止。8 MiB 未満はエラー。LAZ の読み込み単位も RAM 予算に従う。混んだ行は格子ごとの点数で分割し、1格子で予算を超える場合は警告して単独で処理。予算に含まれない固定費（ライブラリ、CUDA コンテキスト、樹冠高ラスター等）を README に明記 | `test_budget_below_minimum_is_refused`、`test_ram_budget_multi_pass`、`test_gpu_budget_paths` |

## 検証コード

| 指摘 | 修正 |
|---|---|
| inner join で欠落を見逃す、Inf の誤判定 | `lidarmetricpy.validate.compare_tables`: キーの一意性、外部結合（欠落・余分は不一致）、有限 / NaN / +Inf / −Inf の区別を先に、有限値だけ相対 1e-6。compare.py、compare_other.py、compare_canopylazr.py、backend_compare.py、docs/make_paper_figs.py、tests のすべてで使う。誤判定の再現例は `test_compare_tables_rejects_false_positives` |
| 参照生成シェルが失敗を止めない | `set -euo pipefail`、一時フォルダに作って行数を確かめてから置き換え、バックグラウンドジョブの終了コードも確認。パッケージの版を tests/data/ref/versions.txt に記録。RSCRIPT=false で失敗すること、作り直すと同梱の参照値と同一になることを確認 |
| CPU 版 PyTorch で CUDA テストが有効になる | 実際に装置で確保できるかで判定 |

## README・論文の説明

- 対応範囲: `metrics_set3` の 111 出力のうち 24 指標と、その lidRmetrics の関数・引数を表で明記。強度系は未実装。
- 既定の前処理（zrange、spike filter）と、lidR と一致させる設定 `Options.lidr()` を区別して記載。
- 速度: R 側を、24 指標に必要な lidRmetrics の関数だけを呼ぶ版（validation/ref_lidr_subset.R、出力は全体版と同一）でも計測し、両方を載せた。

## 修正後の再検証（神奈川の実データ、厳しい比較器）

- 小区画 4 区画: 4 つの float64 バックエンドで 33/33 指標、木 4,275/4,275、欠落・余分・NA の食い違い 0、最大相対差 2.4e-12。
- 小区画 + 全図郭 2 枚（2,800 格子）: lidRmetrics 24、木 9、leafR 10、PyForestScan 3、forestr 1、ForestGapR 3 指標がすべて一致。canopyLazR 9 ラスター（12,600 列）も一致。
- 外枠と leafR の扱いを直した参照値スクリプトで作り直した神奈川の参照値は、前の値と完全に同一。
- テスト: 67 件（GPU のない環境では GPU の分が自動で飛ぶ）。
