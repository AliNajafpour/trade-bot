# Historical news acquisition

Verified 323 eligible publisher metadata records: 183 Guardian and 140 Decrypt. All 17 days have eligible news. Asset tags overlap: BTC 41, ETH 18, SOL 9, MACRO 299; Sonic S has no verified article.

The source universe was fixed by calendar and topic, without reading signals, prices or trade outcomes. Guardian uses 34 daily business/world archives. All business links and fixed macro terms in world URLs were fetched, then article headline/description classified. Decrypt uses the complete 196-row English publisher GraphQL publication inventory for Sep26 through Oct12. This is a bounded publisher sample, not an exhaustive world news archive.

Availability is max(publication, modification). The 185 accepted records revised after publication become visible only at their revision time. Current revisions after replay end, missing revision dates, undated pages and unrelated topics are rejected. The HTTP Last-Modified timestamp is a receipt/server clock, not an article publication clock.

Decrypt API timestamps omit an offset. Although one publisher HTML sample renders the same value with Z, this acquisition uses a conservative +12-hour latest-possible-UTC bound for every naive timestamp. Publisher raw values and the timestamp policy are retained. This delays some genuine news and excludes late Oct12 rows. Date-only timestamps become available at next UTC midnight.

Retrieval took place on Oct23 and is recorded separately. Publisher modification metadata is a reconstructive assumption, not proof of contemporaneous receipt or that a publisher never changed text without updating metadata. Exact downloaded bytes and HTTP receipts are preserved. 184 distinct source files were hashed in the final audit.

Only publisher title/description metadata enters evidence, with a combined title/summary allowance of 25 words. Dynamic price widgets, recommendations, current article-body material and search snippets are excluded.

Sonic S coverage remains a gap. Sonic Labs official RSS and post sitemap show no post in Sep26 through Oct12, and the public Decrypt inventory contains no Sonic title/excerpt. Sonic SVM and generic S references are not treated as Sonic S. The secondary Oct6 Spartan story was not substituted for a missing verifiable official announcement.

Warm-up has 15 eligible articles before the Sep28 09:02UTC replay start. Earliest availability: 2026-09-26T04:02:32+00:00. Latest: 2026-10-12T22:12:48+00:00.

| UTC availability date | Articles | BTC | ETH | SOL | Sonic S | MACRO |
|---|---:|---:|---:|---:|---:|---:|
| 2026-09-26 | 3 | 0 | 0 | 0 | 0 | 3 |
| 2026-09-27 | 9 | 2 | 0 | 0 | 0 | 7 |
| 2026-09-28 | 7 | 1 | 0 | 0 | 0 | 6 |
| 2026-09-29 | 25 | 3 | 2 | 0 | 0 | 24 |
| 2026-09-30 | 25 | 4 | 1 | 1 | 0 | 23 |
| 2026-10-01 | 23 | 3 | 1 | 1 | 0 | 20 |
| 2026-10-02 | 30 | 3 | 1 | 0 | 0 | 29 |
| 2026-10-03 | 13 | 2 | 2 | 0 | 0 | 11 |
| 2026-10-04 | 9 | 0 | 0 | 0 | 0 | 9 |
| 2026-10-05 | 15 | 1 | 0 | 0 | 0 | 15 |
| 2026-10-06 | 30 | 6 | 2 | 2 | 0 | 26 |
| 2026-10-07 | 29 | 3 | 2 | 0 | 0 | 27 |
| 2026-10-08 | 41 | 3 | 2 | 1 | 0 | 40 |
| 2026-10-09 | 30 | 5 | 3 | 3 | 0 | 29 |
| 2026-10-10 | 10 | 2 | 1 | 0 | 0 | 8 |
| 2026-10-11 | 9 | 1 | 0 | 0 | 0 | 9 |
| 2026-10-12 | 15 | 2 | 1 | 1 | 0 | 13 |

Rejection counts: {"current_version_after_replay_end": 22, "outside_fixed_asset_macro_topics": 96}.

Runnable parser check: `python -B scripts/collect_survival_replay_news.py --check`. The collection is resumable from exact successful source receipts. `verification.json` pins the script and evidence hashes and reports the byte and chronology checks.

Files: `evidence.json`, `coverage.json`, `verification.json`, `rejected.json`, `inventory.json`, `decrypt-inventory-receipt.json`, plus raw files and research receipts. `archive-research.md` documents the publisher endpoint discovery and failed alternatives.
