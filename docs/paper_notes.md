# lidarmetricpy 計算方法論文のための整理ノート

作成 2026-10-27。README（使い方）とは別に、論文の方法・結果・考察の材料をまとめる。
数値はすべて validation/ の実測ファイルから転記した（推定値は「推定」と明記）。

## 1. 位置づけと新規性

- **lidR / lidRmetrics と完全一致する再実装。** 33指標、樹頂点、樹冠面積が、同じ点群に対して浮動小数点の丸め誤差（最大 3e-11）の範囲で一致する。比較は、点密度の違う4区画（400格子）と全図郭2枚（2,400格子、木23,559本）で行った。
- **GPU / CPU / Apple を同じコードで。** バックエンドは CuPy（CUDA）、PyTorch（CUDA / Apple MPS / CPU）、NumPy の5種類。MPS は float64 が使えないので float32 で計算する。このモードでも、木と樹冠は完全一致する。
- **メモリ予算。** GPU 側（`gpu_memory_gb`）と主記憶側（`cpu_memory_gb`）の上限を GiB で指定できる（0.2 までの名前は `max_memory_gb` と `host_memory_gb`。今も使えるが警告が出る）。どちらの上限でも、分割しない場合と結果が同じ。
- **他パッケージの指標の統合。** leafR、PyForestScan、forestr（ENL）、ForestGapR、canopyLazR の指標も、元のパッケージと一致させて同じ計算の流れに入れた。
- 文献の網羅的な調査はまだしていないが、我々の知る限り、「lidR と結果が一致することを保証した GPU 実装」はほかにない。投稿前に要確認。

## 2. 実装の概要

処理の流れ（GPU 経路）は次のとおり。

1. LAZ を CPU で展開する（lazrs）。生の整数座標をチャンクごとに GPU へ送る。
2. 座標 x = X·scale + offset を、掛け算と足し算を別のカーネルで計算する。こうすると FMA がかからず、numpy と同じ丸めになる。
3. DTM による高さの正規化、外れ点除去、格子への振り分け、格子ごとの並べ替えを GPU で行う。
4. 格子ごとの指標を、区間和（1格子 = 1 CUDA ブロック）と並べ替えで計算する。
5. rumple は1 m の最高点を GPU で選び、Delaunay 三角形分割（qhull, `QbB Qt`）を CPU で行う。
6. 樹冠高ラスター（p2r, subcircle 0.15）は、順序を保つ int64 原子操作で最大値を取る。
7. 局所最大の検出は GPU、同点の解決は CPU。Dalponte 法による樹冠の領域成長は numba で逐次実行する。
8. 空画素の補間（knnidw）は、lidR の空間索引の順序と libstdc++ の `std::sort` を移植して再現する。

メモリの扱い:
- 点群全体（1点あたり約48バイト）は GPU に置いたまま、重い計算だけを格子の行ごとのブロックで回す。
- GPU に収まらない場合は、主記憶で行ごとに振り分けて流す。
- それでも主記憶に収まらない場合は、LAZ を複数回読む。

コードは lidarmetricpy/（backend.py, api.py, area.py, trees.py, knnidw.py, extra.py, canopylazr.py）にある。

## 3. 指標の一覧

| 由来 | 指標 | 定義の要点 |
|---|---|---|
| lidRmetrics 0.1.5 `metrics_set3`（zmin = 0.5） | zq20, zcv, CRR, zentropy, zpcum7/9, pz_above_30, L4, Lskew, Lkurt, lad_max/mean/cv, p_single, p_last, ratio_last_first, OpenGapSpace, ClosedGapSpace, Euphotic, Oligophotic, vzsd, vzcv, kde_peaks_count, rumple | パッケージの R コードと同じ定義と境界規則 |
| lidR 4.3.3 単木 | tree_n, tree_h_mean/sd/cv/max/gini/shannon, crown_m2_mean/sd | 樹冠高 p2r(0.15, knnidw()) → lmf（窓 2.5 + 0.15h、3–8 m に制限、hmin 5）→ dalponte2016 |
| forestr 2.0.2 | enl | `calc_enl` を、lidRmetrics の1 mボクセル（z > 0.5）の充填数に適用 |
| leafR 0.3.5 | leafr_lai, leafr_lahv（4変種）, leafr_fhd, leafr_fhd_even, leafr_gs, leafr_gs_even, leafr_gc | 1格子 = leafR の1ピクセル（grain.size = 25, k = 1） |
| PyForestScan 0.3.6 | pfs_pai, pfs_cover, pfs_fhd | 1格子 = 1列（25 × 25 × 1 m）。PAD は k = 1、被覆は k = 0.5、2 m 以上 |
| ForestGapR 0.1.7 | gap_frac, gap_n, gap_area_mean | getForestGaps(閾値 10 m, 面積 1–10⁴ m²) を樹冠高ラスターに適用。格子への集計は本研究の定義 |
| canopyLazR 2.0 | clz_chm, clz_ground, clz_vertical_sd_lad, clz_rugosity, clz_filled_ratio, clz_porosity_ratio, clz_max_lad, clz_max_lad_ht, clz_toc_rugosity | ファイル単位の処理。標高（正規化前）を入力とし、パッケージ自身の格子でラスターを返す |

## 4. 検証の設計

- **データ:** 神奈川県の令和6年度航空レーザ点群（中津川上流）。
  - 点密度の違う4図郭から、中央の 250 × 250 m（25 m 格子で 10 × 10）を切り出した。点密度は 52–133 点/m²。
  - 全図郭2枚（1000 × 750 m、1,200格子）: 9050万点と4830万点。
  - 高さは一度だけ DTM で正規化し（最寄り画素、外れ点除去、−1〜60 m）、Z を地上高にした LAZ に書き出した。lidR とこのパッケージが、まったく同じ点を読むためである。
  - canopyLazR だけは標高が必要なので、正規化していない同じ範囲の区画を使った。
- **参照:**
  - R 4.4.3 で作り、R 4.5.3 に上がったあとも出力が差0で同じことを確かめた。
  - lidR 4.3.3 と lidRmetrics 0.1.5。単一スレッドで実行した（lmf の同点処理を決定的にするため）。
  - leafR 0.3.5、forestr 2.0.2、ForestGapR 0.1.7、canopyLazR 2.0（GitHub）、PyForestScan 0.3.6（PyPI。PDAL に依存しない calculate.py を直接読み込んだ）。
  - leafR はファイルを読む関数なので、格子ごとに LAS を書き出して呼んだ。
- **判定:**
  - 格子ごとに |差| ≤ 10⁻⁶ × max(1, |参照値|) で、NA の位置も一致すれば「一致」とした。
  - 木は、樹頂点の座標・高さ・樹冠面積が完全に一致した本数を数えた。
- **スクリプト:** validation/ 以下。
  - 区画データの作成: make_testdata.py, make_fulltile.py, make_rawclips.py
  - 参照値の作成: ref_lidr.R, ref_other.R, ref_pyforestscan.py, ref_canopylazr.R
  - 照合: compare.py, compare_other.py, compare_canopylazr.py
  - 速度: backend_compare.py, make_backend_figs.py

## 5. 結果: 一致

実測ファイル: validation/compare_*.csv, compare_trees_*.csv, compare_other_*.csv, compare_canopylazr.csv, backend_compare/single.csv。

| 照合対象 | データ | バックエンド | 結果 |
|---|---|---|---|
| lidRmetrics 33指標 | 4区画、400格子 | cupy, torch-cuda, numpy, torch-cpu | 33/33 が全格子で一致（最大差 2.3e-12） |
| lidRmetrics 33指標 | 4区画、400格子 | float32（torch-cuda32） | 29/33 が一致。不一致は zpcum7, zpcum9（最大差 0.029%ポイント）, L4, Lkurt（最大差 1e-5）。順位相関はすべて ≥ 0.99998 |
| lidR 単木（樹頂点・樹高・樹冠面積） | 4区画、4,275本 | 5種類すべて | 全数一致 |
| lidRmetrics 33指標 + 単木 | 全図郭2枚、2,400格子、23,559本 | 5種類すべて | 第6節の表を参照（backend_compare/single.csv の metrics_exact） |
| leafR / forestr ENL / ForestGapR / PyForestScan 17指標 | 4区画、400格子 | 5種類すべて | 17/17 一致（最大差 3.4e-13） |
| 同上 | 全図郭2枚、2,400格子 | 5種類すべて | 17/17 一致（最大差 3.4e-12。両方が +Inf の格子が1つあり、一致とした） |
| canopyLazR 9種類のラスター | 正規化前の4区画 × 解像度2通り（10 m, 5 m）、12,600列 | numpy（CPU のみの実装） | 72/72（指標 × 区画 × 解像度）が一致（最大差 3.1e-11） |

照合の途中で、次の2点が分かった。どちらも計算の誤りではなく、比較の条件と端の扱いの問題だった。
- **区画データに z = −1.000 ちょうどの点が残っていた。** 作るとき −1 m より上で切り、そのあとミリ単位に丸めたため。lidarmetricpy の既定（z > −1）ではこの点が落ちる。照合では範囲の絞り込みを外して、R と同じ全点を使った。これが効くのは、地面の層まで数える leafR の LAD 系だけ。
- **leafR は、地面より上の 1 m 層が1つしかない格子でエラーになる**（行列が1列のベクトルに縮むため）。パッケージに合わせて NA を返すようにした。

## 6. 結果: 速度とメモリ

**計測の条件**
- **機械:** RTX PRO 6000 Blackwell Max-Q（96 GB）、AMD Threadripper PRO 7955WX（16コア32スレッド）、主記憶 251 GB。ほかのセッションと共有している機械で、nice 10 で実行した。
- **コード:** パッケージを固定したコピー（validation/backend_compare/snapshot）で測った。予算超過を直した版（2026-10-27 夜に計測し直し）。
- **指標:** lidRmetrics 相当の33指標と単木（other_packages = False）。点群は全点を使った。
- **時間:** 1図郭の時間は、小区画で準備運転したあとの1回の実測値。
- **出典:** validation/backend_compare/summary.md, single.csv, throughput.csv, 図 fig_backend_single.png, fig_backend_throughput.png。
- lidR の時間は、同じ日に同じ機械の R 4.4.3 で測った（ref_full/*.log, par/lidr_par.log）。

**1図郭・1プロセス**（9050万点の図郭 / 4830万点の図郭）

| バックエンド | 秒 | GPU ピーク GiB | 主記憶ピーク GiB | lidR との一致 |
|---|---|---|---|---|
| CuPy | 5.2 / 3.6 | 25.4 / 16.3 | 2.3 / 2.3 | 33/33、木・樹冠すべて |
| PyTorch CUDA | 5.1 / 3.4 | 24.3 / 14.9 | 2.9 / 2.9 | 33/33、木・樹冠すべて |
| PyTorch CUDA float32（MPS 相当） | 18.2 / 10.9 | 17.8 / 10.9 | 10.4 / 6.4 | 27/33 と 29/33、木・樹冠すべて |
| NumPy（CPU） | 50.6 / 30.1 | – | 25.7 / 15.5 | 33/33、木・樹冠すべて |
| PyTorch CPU | 45.5 / 27.6 | – | 26.7 / 16.3 | 33/33、木・樹冠すべて |
| lidR + lidRmetrics（R、1コア） | 230.9 / 179.4 | – | 9.6 / 5.3 | 参照 |

- float32 で一致しなかった指標は、zpcum7, zpcum9, L4, Lkurt（両図郭）、lad_cv（密な図郭の 0.17% の格子）、kde_peaks_count（密な図郭の1格子）。
  - kde_peaks_count は、同じ float32 のコードで計算しても回によって一致したりしなかったりする（予算超過の修正前の計測では一致）。float32 の GPU 集計は足し合わせの順序で値が揺れるためと考えられるが、確かめていない。
- **GPU 版の内訳**（9050万点、CuPy）: 読み込みと LAZ 展開 2.7秒、指標 0.9秒、樹冠高ラスター 0.3秒、単木 1.5秒、rumple 0.3秒。
  - GPU が担う部分は1秒強で、残りは CPU の処理（LAZ 展開、Dalponte 法、qhull）。
- float32 の経路は、座標の添字を主記憶側（float64）で計算し、GPU の前処理を使わないので遅い。
- 他パッケージの指標を加えると、CuPy で 5.2 → 5.7秒、NumPy で 50.6 → 67.3秒（9050万点）。
- **予算超過の修正の前と後:** 時間はほぼ同じ（CuPy 5.26 → 5.21秒、12ワーカー 1.36 → 1.33秒/図郭）。GPU ピークは下がった（CuPy 29.2 → 25.4 GiB）。修正前の値は validation/backend_compare/before_fix/。

**多数の図郭（処理量）**（2つの図郭を交互に流す。秒／図郭 = 経過時間 ÷ 図郭数。ワーカーの起動時間を含む）

| バックエンド | ワーカー数 | 図郭数 | 秒／図郭 |
|---|---|---|---|
| CuPy | 1 / 4 / 12 | 12 / 24 / 24 | 4.40 / 1.71 / 1.33 |
| PyTorch CUDA | 12 | 24 | 1.48 |
| PyTorch CUDA float32 | 12 | 24 | 4.56 |
| NumPy | 1 / 4 / 8 / 16 | 4 / 8 / 16 / 16 | 39.8 / 16.5 / 13.4 / 9.6 |
| PyTorch CPU | 16 | 16 | 12.4 |
| lidR（R 8プロセス） | 8 | 8 | 30.5 |

**読み取り（実測どうしの比較だけ）**
- **1プロセスどうし:** GPU は lidR の約44倍速い（230.9秒 vs 5.2秒）。CPU 版（NumPy）でも約4.6倍（50.6秒）。
- **並列どうし:** GPU 12ワーカーは、lidR 8プロセスの約23倍の処理量（30.5秒 vs 1.33秒／図郭）。CPU 版も16ワーカーで3.2倍（9.6秒）。
- **lidR を全コアで並列にした場合:** 測っていない。lidR は密な図郭1つに主記憶を約10 GB 使うので、この機械で同時に動かせるのは20本前後と考えられる。この比較は推定にとどまる。
- **メモリ:** GPU 版の主記憶のピークは約2–3 GB で、lidR の約10 GB より小さい。
- **メモリ予算の実測**（予算超過を直した版、9050万点の図郭、準備運転のあと、設定ごとに別プロセス。docs/data/memory_budget.csv、図6。直す前の値は data/memory_budget_before_fix.csv）

  | 設定 | 経路 | ブロック数 | 時間 | GPU ピーク（GiB） | 主記憶ピーク（GiB） |
  |---|---|---|---|---|---|
  | CuPy、制限なし | device | 1 | 5.2秒 | 25.4 | 2.2 |
  | CuPy、GPU 16 GiB | device | 3 | 5.2秒 | 12.6 | 2.2 |
  | CuPy、GPU 8 GiB | device | 10 | 5.3秒 | 6.4 | 2.2 |
  | CuPy、GPU 6 GiB | device | 27 | 5.5秒 | 4.6 | 2.2 |
  | CuPy、GPU 5 GiB | device | 49 | 6.0秒 | 4.4 | 1.9 |
  | CuPy、GPU 4 GiB | host | 11 | 9.5秒 | 3.0 | 4.5 |
  | CuPy、GPU 2 GiB | host | 29 | 9.6秒 | 1.3 | 4.5 |
  | CuPy、GPU 2 GiB・主記憶 4 GiB | host | 29 | 9.6秒 | 1.3 | 4.5 |
  | NumPy、制限なし | host | 1 | 50.7秒 | – | 25.7 |
  | NumPy、主記憶 16 GiB | host | 4 | 47.4秒 | – | 9.2 |
  | NumPy、主記憶 8 GiB | host | 10 | 46.8秒 | – | 5.2 |
  | NumPy、主記憶 4 GiB | host | 30 | 39.7秒 | – | 4.0 |

  - **どの設定でもピークは予算以下。** 5 GiB まで GPU の経路のまま約6秒以内で、それより小さいと host 経路（約9.6秒）に切り替わる。
  - **結果は予算によらない。** どの設定でも、制限なしの結果と全指標で同じ（差0、NA の位置も同じ）。
  - **GPU のピークの測り方:** CuPy のメモリプールの大きさを10ミリ秒ごとに記録した値。プールは使い終わった領域も抱えるので、実際に使っている量の上限にあたる。
  - **NumPy は主記憶の予算を小さくしても遅くならなかった。** 4 GiB では制限なしより速かった（39.7秒 vs 50.7秒）。理由は確かめていない。
  - **12ワーカーの処理量は修正の前後で同じ**（CuPy、24図郭。修正前 1.36秒/図郭、修正後 1.33秒/図郭、修正直後の確認で 1.37秒/図郭）。
  - **予算超過の修正**（2026-10-27）
    - **直す前:** device 経路（点群全体を GPU で前処理する経路）は予算を超えていた。6 GiB の指定でピーク 8.48 GiB、8 GiB の指定で 8.55 GiB。
    - **原因:** 前処理で全点の配列を一時的に二重に持っていた（チャンクの結合と、フィルタの結果を全配列まとめて作る処理）。スパイク除去では全点を一度に並べ替えていた。加えて見積もりが小さすぎた（1点48バイト）。
    - **直し方:**
      - 確保済みの配列にチャンクを直接書き込む。
      - フィルタは配列を1つずつかける。
      - スパイク除去は1 m 行の帯ごとに並べ替える（列の中の順序は全体の並べ替えと同じ）。
      - 格子番号を int32 にする。
      - 見積もりを実測に合わせる（前処理で1点36バイトとチャンク1点100バイト、ブロック中は1点40バイト、プールの余裕として 0.5 GiB）。
    - **直した後の確認:**
      - 直す前の版と24通りで比べた。24通りは、バックエンド（CuPy、PyTorch CUDA）× スパイク除去の有無 × 予算（なし、6 GiB、2 GiB）× DTM の有無（合成した DTM）。
      - CuPy の12通りと、PyTorch CUDA の予算なしと 2 GiB の8通りは、ビット単位で同じだった。
      - PyTorch CUDA の 6 GiB の4通りは、最大 4e-14 の差があった。直す前の版でも、予算によってブロックの分け方が変わると同じ大きさの差が出る（PyTorch の集計の順序がブロックの大きさで変わるため）。CuPy では出ない。
      - 最後の調整（読み込みの区切りの自動縮小と余裕分）の後に、CuPy・スパイク除去あり・DTM ありで 4.5 / 6 / 3 GiB を直す前の制限なしの結果と比べ、ビット単位で同じだった。
      - lidR との一致（小区画）は CuPy と PyTorch CUDA で33/33、木 4,275/4,275。

## 6b. 論文用の図表（docs/figures, docs/tables）

- 作り方: `docs/run_paper_data.sh`（make_paper_data.py と memory_budget.py。数値をすべて計算し直す）→ `docs/make_paper_figs.py`（図は png 300 dpi と pdf、表は csv と md）。
- 値はすべて実測ファイルから読む。推定値は図表に入れていない。

| 番号 | 内容 | 元のデータ |
|---|---|---|
| 図1 fig1_workflow | 処理の流れ（CPU と GPU の分担、出力の指標数） | – |
| 図2 fig2_agreement_scatter | 16指標の散布図、参照パッケージ vs lidarmetricpy（CuPy） | cells_vs_reference.parquet, canopylazr_vs_reference.parquet |
| 図3 fig3_agreement_heatmap | 全指標 × 5バックエンドの最大相対差。× は 1e-6 を超えた格子があるか NA の位置が違う指標 | 同上 |
| 図4 fig4_single_tile | (a) 1図郭の時間（lidR を含む）、(b) 処理段階ごとの内訳 | backend_compare/single.csv, ref_full/*.log |
| 図5 fig5_throughput | ワーカー数と1図郭あたりの秒数（lidR は 8プロセスのみ） | backend_compare/throughput.csv, par/lidr_par.log |
| 図6 fig6_memory | メモリ予算ごとのピークと時間。縦線が指定した GPU 予算 | data/memory_budget.csv |
| 表1 table1_metrics | 指標、元のパッケージと版、定義、使う点 | – |
| 表2 table2_datasets | 検証データ（小区画4、図郭2、生の標高の小区画4） | validation/data/*.json |
| 表3 table3_agreement | パッケージ × バックエンドの一致数と最大相対差 | agreement_per_metric_backend.csv |
| 表4 table4_speed | 時間、ピークメモリ、処理量 | single.csv, throughput.csv |
| 表5 table5_reproduction_details | 完全一致に必要だった細部 | – |

## 7. 再現に必要だった細部（論文の付録の候補）

lidR / lidRmetrics と一致させるには、次の点を一つずつ合わせる必要があった。どれも「定義どおり」に実装しただけでは一致しない。

1. **丸め:**
   - lidR の `round_any` は C の `round()`（0.5 は0から遠い側）を使う。ボクセル座標と、樹冠高ラスターの範囲決めに影響する。
   - R の `round(x, 3)` は、`numpy.round` と約6%の値で結果が違う（R 4 の fround.c は、元の倍精度値に近いほうの10進数を選ぶ）。
2. **区切りの規則:**
   - `seq()` は最後の区切りを `pmin(x, to)` で上限に揃える。zpcum に影響する。
   - `findInterval` は、ちょうど最大値の点を範囲外にする。zentropy に影響する。
3. **ボクセル:**
   - lidRmetrics は z を先に丸めてから z > 0.5 で選ぶので、z = 0.5 ちょうどの点も1 mのボクセルに入る。
   - Lefsky のクラスでボクセルが0個のものは、0%ではなく NA になる（left join の仕様）。
4. **KDE:**
   - `stats::density` の線形の振り分け（BinDist）、2n 点の FFT 畳み込み、`approx` による補間を、そのまま移植した。
5. **単木:**
   - lmf の同点規則: 先に最大と判定された点が窓の中にあると、同じ高さの点は最大にならない。走査はラスター順。
   - Dalponte 法は、同じ回の中で平均高さを更新し、先に割り当てた領域を後の領域が上書きする。
   - knnidw の k 近傍で距離が同じ点が出たときの選び方は、lidR の GridPartition の探索順と libstdc++ の introsort（16要素の閾値、3点の中央値で軸選び）で決まる。
     - C++ の実物と比べ、400回の試行で順序が完全に一致した。
6. **GPU 特有:**
   - CuPy の `scatter_max` / `scatter_min` と `ufunc.at` は、float64 でも内部で float32 の原子操作を使う。順序を保つ int64 に変換して避けた。
   - PyTorch の CUDA は、Python の数値での割り算を逆数の掛け算に置き換える。区切りちょうどの値が動くので、テンソル同士の割り算にした。
   - 座標の計算に FMA がかかると、numpy と最後の1ビットが変わる。別のカーネルに分けて避けた。
7. **他パッケージ:**
   - **leafR:** LAD の層のどこか1つでも NA（下に点がない層）があると、LAHV と FHD は NA になる。FHD の閾値は 1/層数（% 単位）。
   - **PyForestScan:** z の区切りは `arange(0, zmax + 1)` で、zmax が整数ちょうどなら最後の閉区間に入る。2 m 以上の層がないと、PAI と被覆は0になる。
   - **canopyLazR:**
     - 外れ値の範囲は、ファイル全体の型7の四分位から決まる。
     - 層への振り分けは `round()`（0.5 は偶数側）。
     - 範囲外の添字の点は、factor の NA として落ちる。
     - 各柱の最低点を地面とし、地面が0以下の柱は NA。
     - sd は4桁に丸める。3×3窓は外周を使わない。
8. **参照側の注意:**
   - lidR の lmf は OpenMP で並列にすると、同点の処理が実行ごとに変わり得る。参照は1スレッドで作った。
   - rlas と laspy で、座標の最後の1ビットが違う場合があった（標高で最大 2e-13 m）。

**公開前のレビューで見つかった2点**（2026-10-28。合成データのテスト tests/ で見つかった。神奈川のデータは格子どおりに切り出してあり、外枠ちょうどの点がなかったため、それまでの検証では表に出なかった）
- **格子の外枠:** lidR（terra）は、ラスターの右端と下端を閉じた区間として扱う。外枠ちょうどの点は最後の列・行に入る。lidarmetricpy は半開区間で、この点を落としていた。既定を lidR と同じにし、隣の図郭と端を共有する場合のために `Options(closed_edges=False)` を残した。
- **rumple の 1 m 画素:**
  - `decimate_points(highest(1))` は、格子ごとの点の範囲を `st_adjust_bbox` で丸めて画素の枠を作る。右端は `round_any(xmax − 0.5) + 1`（C の round）。
  - 最大 x が負の整数のときは、右端がその点にちょうど重なり、その点は最後の列に入る（下端も同様で、最小 y が正の整数のとき）。
  - lidarmetricpy は全体で共通の floor(x) の画素を使っていたので、この場合だけずれていた。格子ごとにこの規則を入れた（`rumple_pixels`）。神奈川の座標は負なので、実データでも起こりうる。
- 他パッケージの参照値を作る自作スクリプト（ref_other.R, ref_pyforestscan.py）も、外枠の扱いを lidR に合わせた。leafR は、格子の点の広がりがちょうど 25 m だと画素を2つに分けてしまうので、`grain.size = 26` にして1格子1画素にした。leafR の LAD は点数の比だけで決まるので、面積には依存しない。
  - 直したスクリプトで神奈川の小区画の参照値を作り直すと、前の値と完全に同じだった。

## 8. 限界

- **LAZ の展開が下限。**
  - 9050万点の図郭で、LAZ の展開は1スレッドで CPU 23秒、並列版で30秒かかる。これが CPU 時間全体の約85%を占める。
  - GPU の計算をこれ以上速くしても、多数の図郭を流す速さは展開で頭打ちになる。
  - LAZ を一度だけ非圧縮の形式に変える方法は、同じ図郭を繰り返し処理する場合にしか得にならない。
- **Apple MPS は実機で未確認。**
  - float32 の経路は、CUDA 上の float32（torch-cuda32）で検証した。
  - 一致するのは、木、樹冠、他パッケージの指標と、33指標のうち全データを通して27指標（小区画だけなら29、図郭ごとには27と29）。外れたのは zpcum7, zpcum9, L4, Lkurt, lad_cv, kde_peaks_count（図3）。kde_peaks_count は密な図郭の1–2格子だけで、同じコードでも回によって一致したりしなかったりする。float32 の GPU 集計は足し合わせの順序で値が揺れるためと考えられるが、確かめていない。MPS 上の値の差は実測していない。
- **GEDI 相当の指標は未実装。** 理由は3つある。
  - 本来の gediSimulator は C のプログラム（Hancock et al. 2019）で、rGEDI の最新版はその R への移植版である。
  - 観測範囲が、ガウス分布で重みを付けた直径約25 m の円で、格子の集計ではない。隣の格子の点まで要る。
  - RH 値を出すには、波形のノイズ処理と地面の検出が必要。
- **forestr は地上型ライダー（PCL）用。**
  - ENL だけを、lidRmetrics のボクセルに `calc_enl` を当てはめる形で使った。
  - Hardiman の rugosity は、航空機ライダー版の canopyLazR で代える。
- **ForestGapR の格子への集計は本研究の定義**（ギャップの割合、数、平均面積）。ギャップの抽出そのものはパッケージと一致する。
- **canopyLazR はファイル単位で CPU のみ。** 9050万点の図郭で4.0秒（読み込み2.5秒は別）。格子は canopyLazR 自身の格子で、本研究の25 m 格子とは別。
- **検証データの範囲。** 神奈川県の1つの計測（最大7リターン、52–133 点/m²）だけで確かめた。センサーや密度の違うデータ（米国 3DEP など）での照合はこれから。
- **メモリ予算は見積もりに基づく。** 1点あたりのバイト数は、9050万点の図郭で CuPy のプールを実測して決めた。密度や点の属性が大きく違うデータ、PyTorch のキャッシュ方式、Apple の統合メモリでは確かめていない。
- **lidR 側の時間は1プロセスでの実測。** 全コアで並列にした場合の値は測っていない（推定のみ）。

## 9. ライセンスと引用

- **ライセンスは GPL-3.0-or-later。** lidR（GPL-3）の C++ を移植したため。
  - 移植した部分: 局所最大、Dalponte 法、knnidw、GridPartition。
  - leafR（GPL-3）、forestr（GPL-3）、canopyLazR（GPL-2 + file LICENSE）は、定義を再実装した。コードの複製ではない。
  - 公開前に各ライセンスとの両立を確認すること。

引用すべき文献:
- DOI の確認方法は2つある。
  - 「確認済」: パッケージの DESCRIPTION / citation() か、検索で一次資料を確かめたもの。
  - 「要確認」: 記憶にもとづくもの。
- 確認済:
  - Roussel J. et al. 2020. lidR: An R package for analysis of Airborne Laser Scanning (ALS) data. Remote Sensing of Environment 251: 112061. doi:10.1016/j.rse.2020.112061
  - Tompalski P. lidRmetrics: Additional metric functions for the lidR package. R package 0.1.5. https://github.com/ptompalski/lidRmetrics（論文はなし）
  - Almeida D.R.A. et al. 2019. Remote Sensing 11(1): 92. doi:10.3390/rs11010092（leafR の方法）
  - Stark S.C. et al. 2012. Ecology Letters. doi:10.1111/j.1461-0248.2012.01864.x（leafR の LAD）
  - Atkins J.W. et al. 2018. Methods in Ecology and Evolution. doi:10.1111/2041-210X.13061（forestr）
  - Hardiman B.S. et al. 2013. Forests 4(3): 537. doi:10.3390/f4030537
  - Hardiman B.S. et al. 2011. Ecology 92(9): 1818–1827. doi:10.1890/10-2192.1（canopy rugosity）
  - Silva C.A. et al. 2019. Methods in Ecology and Evolution. doi:10.1111/2041-210X.13211（ForestGapR）
  - Kamoske A.G. et al. 2019. Forest Ecology and Management 433: 364–375（canopyLazR）。DOI は要確認
  - PyForestScan. Journal of Open Source Software (2025). doi:10.21105/joss.07314
  - Dalponte M., Coomes D.A. 2016. Methods in Ecology and Evolution 7(10): 1236–1245. doi:10.1111/2041-210X.12575
  - Hancock S. et al. 2019. Earth and Space Science 6(2): 294–310. doi:10.1029/2018EA000506（GEDI simulator。未実装の理由の説明に使う）
- 要確認:
  - Ehbrecht M. et al. 2016. ENL（Forest Ecology and Management）
  - Lefsky M.A. et al. 1999. 光環境ボクセル（Remote Sensing of Environment）
  - MacArthur R.H., Horn H.S. 1969（LAD）
  - Popescu S.C., Wynne R.H. 2004（可変窓の局所最大）
  - Bouvier M. et al. 2015（lidR::LAD）
  - Valbuena R. et al. 2012 / 2017（leafR の GS と GC）
  - Hosking J.R.M. 1990（L モーメント）
  - Barber C.B. et al. 1996（qhull）
