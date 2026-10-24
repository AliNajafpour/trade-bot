# Trading test review, 2026-10-21

The reviewer inspected the frozen-policy backtest, prospective shadow and recorded-observation replay scripts. No material P1 or P2 finding was confirmed in this scope.

The 96 version-two technical-policy ledgers were independently replayed exactly. Input/source hashes, official 30 basis point fees, cash/inventory, fully closed P&L, next-open confirmation timing and the 0.25% planned risk ceiling passed independent checks. The report distinguishes original technical forecasts, a smaller exposure control and the deterministic conditional proxy. It explicitly makes no historical LLM/news fusion profit claim.

The observed replay's four real order-book body hashes and receipt clocks were verified. Context coverage was recomputed causally from saved snapshots. Zero positive technical candidates, zero new inference calls, no executed trades and null historical fusion P&L are reported correctly. Four earlier actual model receipts passed file, event and schema validation and remain separate diagnostic evidence.

The shadow script fetches a fresh quote for each account after any earlier account's inference delay. The core requires a later post-analysis quote before entry. Account marks use fresh bid prices, deduct hypothetical exit fees and distinguish open unrealized P&L from closed realized P&L. Missing quotes produce an unavailable mark rather than invented prices.

The parent completed the bounded shadow with six independent accounts, 24 samples and 144 ticks. It reports zero model evaluations, entries, trades and AI account P&L because technical forecasts did not clear costs. The parent verified 255 raw hashes and all 144 input events. The only partial context flag was the documentation host's TLS failure; trading inputs were available. Separate fixed-15% baseline marks are comparison accounts and are not AI trades. This reviewer did not repeat the completed raw-file sweep or alter its verification receipt.

Fresh focused commands passed with the existing Trade-Bot Python environment:

```powershell
& 'D:\ALI\Programming-AI\Personal Projects and Tests\Trade-Bot\.venv-gpu\Scripts\python.exe' -B -m checks.check_fusion_policy
& 'D:\ALI\Programming-AI\Personal Projects and Tests\Trade-Bot\.venv-gpu\Scripts\python.exe' -B -m checks.check_fusion_shadow
& 'D:\ALI\Programming-AI\Personal Projects and Tests\Trade-Bot\.venv-gpu\Scripts\python.exe' -B -m checks.check_observed_fusion_replay
```

The parent also reports 33 passing software checks. The reviewer performed no new exchange requests, model CLI calls, GPU fitting or unrelated benchmark replay. Current diagnostic fusion decisions do not revise historical technical returns. Short shadow abstention supplies operational evidence, not profitability evidence.

Verdict: the completed testing artifacts are reviewable with their stated paper, friction and retrospective-selection limits.
