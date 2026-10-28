# 2回目のレビュー（v0.3.0 fe14561）への回答 → v0.3.1

レビューの ZIP は使わず、本文の再現条件から手元で再現してから直した。各項目に回帰テストを付けた
（tests/test_regressions.py、tests/test_oracle.py。R の照合値は tests/make_oracle.R がインストール済みの
lidR 4.3.3 / lidRmetrics 0.1.5 の `metrics_set3` で作る）。

## 数値計算

| 指摘 | 再現 | 修正 | テスト |
|---|---|---|---|
| ① 高さちょうど zmin のセルのボクセル指標が、隣のセルの有無で変わる | 高さ 0.5 m の4点: 単独で Euphotic・enl が NA、植生のある隣と一緒だと 100・1（NumPy、Torch CPU、CuPy） | ボクセル指標と ENL を、生の高さの指標から切り離して常に計算（`_voxel_metrics`）。R も Euphotic = 100 を返すことを確認（`at_zmin_only`） | `test_voxels_of_points_at_zmin_do_not_depend_on_neighbours`、oracle `at_zmin_only` |
| ② DTM の画素境界で、集計グリッドの原点により地盤高が変わる | 0.1 m DTM、x = 0.5: グリッド x0 = 0 で 100.4（zq20 ≈ 10.2）、x0 = 0.1 で 100.5（10.1） | 画素番号を DTM 自身の原点から floor((x − x_dtm) / res) で求め（lidR と同じ）、整数の窓のずれを引く。host・device の両経路。窓の外は端に寄せず NA | `test_dtm_pixel_on_boundaries_independent_of_grid_origin`（解像度 0.1、0.2、0.25 m、全点が画素の辺上、原点 4通り、全バックエンド） |
| ③ zmin < 0 のエントロピー | `[0,1,2,3]` で 1.0、`[-0.2,1,2,3]` で NA | `metrics_dispersion` と同じく z > 0 の点だけでエントロピーを計算 | oracle `entropy_zmin_negative_zero`、`entropy_zmin_negative`（R で 0.63093） |

## メモリー制御と検証コード

| 指摘 | 修正 | テスト |
|---|---|---|
| ④ GPU 経路で LAZ の読み込み単位が RAM 予算に従わない | 読み込み（ホストでの展開）の単位は RAM 予算から、GPU の処理スライスは GPU 予算から、別々に決める | `test_device_path_reads_laz_within_ram_budget`（laspy の chunk_iterator に渡る値が RAM 予算の上限以下で、経路は device のまま） |
| ⑤ プロセス置換のジョブ列挙が失敗しても成功終了 | ジョブ一覧を一時ファイルに書き（失敗で停止）、0件ならエラー。4本とも、壊れた manifest と空の manifest で終了コード 1 を確認 | 手動（壊れた manifest のコピーで実行） |
| ⑥ 参照ファイルの一部欠けで部分検証が全体成功 | 計算の前に必要な参照ファイルをすべて確認し、欠けていれば停止。予定のデータ数と検証済みの数を表示 | compare.py、compare_other.py、compare_canopylazr.py |
| ⑦ NA セルの欠落、文字列、空同士 | 必要な格子を指標の値と独立に決める（`occupied`。LAZ から `occupied_cells` で数える）。必要な格子が欠ければ NA でも不一致。数値でない値は ValueError。空同士は status = "empty"（equal ではない） | `test_compare_tables_rejects_false_positives` に追加 |

## ベンチマーク

R と Python を同じ区間で計測し直した（validation/run_timing_lidr.sh、run_timing_py.sh）。

- 処理時間: 読み込み + 指標 + 単木。Python は準備運転のあとの `compute_tile`、R はスクリプト内の read_s + area_s + tree_s。
- プロセス全体: 新しいプロセスで起動・import（パッケージ読み込み）・処理・保存（格子表、樹頂点、樹冠高ラスター）まで、`/usr/bin/time`。

| 1図郭（9050万点） | 処理時間 | プロセス全体 |
|---|---|---|
| CuPy | 5.5秒 | 6.9秒 |
| NumPy | 53.4秒 | 54.4秒 |
| lidR、24指標の関数だけ（R 4.5.3） | 165.7秒 | 205.3秒 |
| lidR、metrics_set3（R 4.5.3） | 174.9秒 | 214.6秒 |

同じ24指標で、CuPy は lidR の約30倍（どちらの区間でも）。0.3.0 の README の「約38倍」は、R だけ起動と保存を
含めた時間で割っていたため大きかった。速度比較の計測スクリプトは、計算のあとの照合で LAZ を読むため、主記憶の
ピークは計算の直後に記録するようにした（そうしないと CuPy の主記憶ピークが 2.2 GB ではなく 7.9 GB に見えた）。
