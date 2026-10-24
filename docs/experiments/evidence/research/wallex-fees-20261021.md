# Official Wallex fees checked 2026-10-21

The current public order-book fee schedule is available without credentials from [Wallex's structured-fee endpoint](https://api.wallex.ir/v2/structured-fee). It returned HTTP 200 at **2026-10-21 15:04:19 UTC**, with `success=true` and a **30-day** volume window. This is a public conditional schedule, not verification of this user's account tier, promotion, or competition account.

The official [commission page](https://wallex.ir/commission) loads the endpoint through its current public client code. Its table labels the volume as 30-day volume and displays `100 * maker_fee` and `100 * taker_fee` as percent. `/fee` redirects to `/commission`. `/fees` and `/help/fees` returned 404. The web reader did not render the commission table, so the endpoint response and frontend source were downloaded directly with ordinary TLS verification and no account credentials.

## Exact public schedule

Each threshold is the raw API threshold in that quote currency. Upper ranges follow the next published threshold; exact assignment at an equality boundary remains an account/backend detail. These are **per-side trading fees**, not round-trip costs.

| Quote | 30-day volume threshold | Maker percent | Taker percent | Maker bps | Taker bps |
|---|---:|---:|---:|---:|---:|
| TMN | 0 | 0.25% | 0.30% | 25 | 30 |
| TMN | 100,000,000 | 0.20% | 0.25% | 20 | 25 |
| TMN | 1,000,000,000 | 0.17% | 0.20% | 17 | 20 |
| TMN | 3,000,000,000 | 0.14% | 0.17% | 14 | 17 |
| TMN | 10,000,000,000 | 0.12% | 0.15% | 12 | 15 |
| TMN | 20,000,000,000 | 0.10% | 0.13% | 10 | 13 |
| USDT | 0 | 0.10% | 0.13% | 10 | 13 |
| USDT | 500,000 | 0.05% | 0.08% | 5 | 8 |

TMN means toman, not rial: 1 TMN = 10 IRR. The API rates are fractions: `0.003 = 0.30% = 30 bps`. One basis point is `0.0001` as a fraction. The raw response also has a BTC quote category with zero maker/taker rates; that does **not** mean BTC/TMN or BTC/USDT has zero fees. Do not apply the BTC category to markets whose quote is TMN or USDT. Both TMN and USDT include raw `is_fixed=true`; preserve this field without inventing an account-specific interpretation.

Maker adds an order that waits in the book; taker matches existing liquidity immediately. These roles apply to buys and sells. A limit order is not automatically maker: one that crosses the book can take liquidity. The older API reference's Persian descriptions of maker as seller and taker as buyer should not be used to conflate liquidity role with trade direction.

## Account and credit-account limits

The [official API reference](https://api-docs.wallex.ir/) places `GET /v1/account/fee` under API-key protected account services. It returns per-market `makerFeeRate`, `takerFeeRate`, and `recent_days_sum`. Its `0.002` sample is **an example**, not this user's rate. The current commission frontend additionally calls `GET /v2/account/structured-fee` for account-specific structure. Its precise supported authentication scheme was not independently tested. No authenticated account request was made, and no account tier is assigned here. Actual fill reports expose `fee`, `feeAsset`, `feeCoefficient`, and maker/taker coefficients in the [official order-detail reference](https://developers.wallex.ir/spot/singleorderdetail).

Search-indexed official [credit terminology](https://developers.wallex.ir/trade-credit/intro) describes a separate daily credit fee and 30-day packages. Indexed [package details](https://developers.wallex.ir/trade-credit/proppackagedetails) and [package history](https://developers.wallex.ir/trade-credit/subaccounthistory) document daily/final fee and discount fields, with `x-api-key` in their examples. **All three direct document URLs returned 404 during this check**, so indexed text does not establish current package rates. The old `api-docs.wallex.ir` host also failed direct TLS hostname verification; verification was not disabled.

No official public rule fixing the fee for this particular competition/credit account was found. Its trading fee, daily financing fee, discounts and package restrictions remain **unknown**, not zero. Standard spot levels cannot be assumed to override competition terms. Package fees are separate from order execution fees.

## Conflicting marketing copy and other costs

The current official [BTC purchase page](https://wallex.ir/buy-and-sell/btc) and [ETH purchase page](https://wallex.ir/buy-and-sell/eth) still describe generic buy/sell fees of **0.35% down to 0.20% over three months**. That conflicts with the live structured maker/taker table's 30-day schedule. Preserve the discrepancy; use the current dedicated fee endpoint for conditional spot scenarios. A 35-bps scenario may be labeled older published marketing sensitivity, not current public tier 1 or the user's actual rate.

The official commission-page FAQ says instant/OTC buy/sell has zero trading commission. That is a different product: an OTC quote can still include a buy/sell spread. Do not apply OTC's zero commission to order-book spot.

Keep execution cost components separate: order commission, bid/ask spread, market impact, and residual slippage. Withdrawal/network charges and fiat funding charges are not per-trade spot fees. Historical spread/slippage estimates are execution assumptions, not official exchange commissions. A two-sided taker example at public TMN level 1 is approximately 60 bps in commissions before spread/impact, with exact cash/quantity accounting depending on fee asset and fill values.

## Reproducible evidence

- Machine-readable rates: [wallex-official-fees.json](../../../../config/wallex-official-fees.json).
- Raw public response: [public-structured-fee.json](../../../fee-evidence/public-structured-fee.json).
- Receipt: public-structured-fee.receipt.json (`fee-evidence/public-structured-fee.receipt.json`; local supporting artifact).
- Raw response SHA-256: `bb0a2a600d56ccd2f0581a142b6cba2452f04f11acb2437bba6f74688f403e8b`.
- Frontend source defining public/account endpoints: 04h_i7nro62s_.js (`fee-evidence/04h_i7nro62s_.js`; local supporting artifact), with adjacent hash receipt.
- Frontend table labels, coefficient conversion, maker/taker definitions and OTC FAQ: 0k2dbbt03t41s.js (`fee-evidence/0k2dbbt03t41s.js`; local supporting artifact), SHA-256 `d06d0d1f594f45982386c3fe3d501843ef25afa6717cb78ea98ae285ab8cbee6`.

Raw HTML, public scripts and receipts include request URL, capture time, HTTP status or failure, and SHA-256 under `docs/fee-evidence/`. No account API, order, or credential access occurred.
