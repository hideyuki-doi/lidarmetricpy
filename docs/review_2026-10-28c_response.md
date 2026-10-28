# 最終レビュー（v0.3.1 7d1f2d9）への回答 → v0.3.2

指摘2件とも、レビューの再現条件で再現してから直した。レビュー同梱の回帰テスト
（test_remaining_regressions.py）は 0.3.1 で2件とも失敗し、0.3.2 で2件とも成功する。同じ内容の回帰テストを
tests/test_regressions.py に加えた（0.3.1 では失敗、0.3.2 では成功を確認）。

## A. DTM の右端・下端ちょうどの点

| 項目 | 内容 |
|---|---|
| 再現 | 2×2・1 m の DTM、点 (2, 0.5)、(2, 1.5)、(0.5, 0)、(1.5, 0): 0.3.1 は NumPy・Torch CPU で0行 |
| R での確認 | インストール済みの lidR 4.3.3 の `raster_cell_from_xy` を表示し、`col[x == xmax] <- ncol - 1`、`row[y == ymin] <- nrow - 1` を確認。さらに `normalize_height`（R 4.5.3、terra 1.9.50）を 0.5 × 1 m の非正方形画素の DTM で実行し、右端・下端・右下角・上端・左端・内側の10点の地盤高を得た |
| 修正 | `_dtm_window` が DTM 自身の右端 x、下端 y、幅、高さも返す。画素番号を floor で求め、x == xmax は最終列、y == ymin は最終行に置き換えてから、整数の窓のずれを引く。host・device の両経路 |
| 範囲外の点 | 0.3.2 でも除く（内側に寄せない）。なお lidR の `normalize_height` は、範囲外の点に最近傍の画素を警告付きで使う（上の R 実行で確認）。この違いは README に明記した |
| テスト | `test_dtm_right_and_bottom_edges_take_the_last_pixel`（全バックエンド。R の地盤高と一致、範囲外4点は除外、レビューの例で zq20 = 10.6） |

## B. GPU 経路の最初の転送

| 項目 | 内容 |
|---|---|
| 再現 | device 上限 1000 点・ホスト読込 2000 点で、最初の転送が 2000 点 |
| 修正 | LAZ の読込単位を min(RAM 予算の読込単位, GPU のスライス) にした。メモリー上の点群（dict）も GPU のスライスごとに転送する（0.3.1 は全点を一度に転送していた） |
| テスト | `test_device_path_transfers_within_gpu_chunk`（LAZ と dict。NumPy で動かす device 経路で、読込段階のすべての転送が上限以下であること、結果がスライスの大きさによらないこと） |

## その他

- `compare_tables` の pandas FutureWarning（`fillna` の型の暗黙変換）を解消した。
