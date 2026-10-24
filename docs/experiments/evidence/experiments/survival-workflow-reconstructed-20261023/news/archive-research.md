# Publisher archive research for historical replay

Research date: 2026-10-23. Fixed inventory interval: 2026-09-26 00:00 through 2026-10-12 22:58, covering prior news before the replay begins. Source discovery used calendar dates and publisher interfaces, without trading outcomes.

## Working Decrypt inventory

The [publisher article page](https://decrypt.co/379833/uptober-will-bitcoin-live-up-to-it) references its [public application JavaScript](https://decrypt.co/_next/static/chunks/pages/_app-e6ae6b62b87fe901.js). That code specifies `https://gateway.decrypt.co` as its GraphQL endpoint and uses GET requests with `Apollo-Require-Preflight: true`. The published `ArticlePreviews` query accepts `ArticleEntityFilterInput`, `PaginationArg`, and sort strings. A direct public GET with the fixed publication interval returned HTTP 200 and all 196 English published articles in one page (`pageSize:1000`, `pageCount:1`, `total:196`). No article in that response has `modifiedAt` after the replay end.

Canonical URLs reconstruct as `https://decrypt.co/{id}/{slug}`. The exact GET URL, query, variables, HTTP headers/times and response hash are in `research-probes/decrypt_gateway_full_archive.json`; the exact response is `decrypt_gateway_full_archive.raw`. Its SHA-256 is `c7aa8064dcf8f9970a7f6ce55ec0d1edf82afea3a0b3eeb20381329bc613e7ed`.

The selected metadata fields are `id`, `slug`, `title`, `excerpt`, `publishedAt`, `modifiedAt`, `coins.data`, `tags.data`, and `category.data`. `coins` is incomplete: many Bitcoin articles lack BTC metadata, so absence of a coin tag does not establish absence of relevance. No title or excerpt in this interval contains Sonic. Sonic S must not be inferred from Sonic SVM or a generic S symbol.

API timestamps are naive strings. The sample article has JSON-LD `NewsArticle.datePublished` and `dateModified` both `2026-10-01T17:11:44`, and HTML explicitly renders `<time dateTime="2026-10-01T17:11:44Z">`. Thus interpreting this publisher's API timestamp as UTC is supported by its rendered article timestamp, and should be recorded as a publisher-format assumption or checked per article. The article page also embeds the same fields in `__NEXT_DATA__.props.pageProps.post`.

### Inventory counts by publication day

| UTC date | Articles |
|---|---:|
| 2026-09-26 | 4 |
| 2026-09-27 | 3 |
| 2026-09-28 | 16 |
| 2026-09-29 | 15 |
| 2026-09-30 | 16 |
| 2026-10-01 | 16 |
| 2026-10-02 | 11 |
| 2026-10-03 | 4 |
| 2026-10-04 | 4 |
| 2026-10-05 | 17 |
| 2026-10-06 | 17 |
| 2026-10-07 | 16 |
| 2026-10-08 | 20 |
| 2026-10-09 | 13 |
| 2026-10-10 | 4 |
| 2026-10-11 | 3 |
| 2026-10-12 | 17 |

### Decrypt routes that do not supply this inventory

The [robots file](https://decrypt.co/robots.txt) identifies the [sitemap index](https://decrypt.co/sitemap_index.xml). Its newest numbered children [58](https://decrypt.co/post-sitemap58.xml) and [59](https://decrypt.co/post-sitemap59.xml) returned HTTP 200 but empty `urlset` elements (504 bytes each). Their index lastmods were October 8 and October 23 respectively. The [base post sitemap](https://decrypt.co/post-sitemap.xml) returned 1001 URLs: homepage plus 1000 old articles from 2018–2019, with no articles in the target interval. Sitemap child entries contain `loc`, `lastmod`, and sometimes image metadata, not a separate publication field. `https://decrypt.co/sitemap.xml` and the date-filtered `/wp-json/wp/v2/posts` route returned HTTP 404. These failures are preserved separately; the GraphQL result supplies coverage instead.

## Working Guardian chronological archive

The [world day archive](https://www.theguardian.com/world/2026/sep/28/all) and [business day archive](https://www.theguardian.com/business/2026/sep/28/all) both returned HTTP 200. Dated article links can be collected before reading article metadata; the index itself does not contain a `datePublished` field. The parent collector owns the full daily world/business sweep.

A [sample business article](https://www.theguardian.com/business/2026/sep/28/trump-us-economy-midterm-elections) returned HTTP 200. Its JSON-LD is a list containing `@type:NewsArticle`, with `datePublished:2026-09-28T11:00:32.000Z` and `dateModified:2026-09-28T17:03:53.000Z`. Canonical source is `mainEntityOfPage`; headline and publisher/author metadata are also present. Raw pages and SHA-256 receipts are preserved under `research-probes/guardian_{world_day,business_day,article}.{raw,json}`.

The public Content API search using `api-key=test` and `api-key=demo` each returned HTTP 401. No further API-key guesses were made.

## Bitcoin.com and reconstruction limits

`https://news.bitcoin.com/sitemap.xml` returned HTTP 403 in this research. The parent collector separately preserved its WordPress REST HTTP 403. These failures must not become synthetic news. An article supplied by the parent can still be evaluated from its own publisher metadata.

`historical_available_at` should be the maximum of publication and modification timestamps; `retrieved_at` is the current acquisition time and must stay separate. Exclude current article versions whose revision time is after the replay boundary. Date-only records need a conservative next-day availability rule. Publisher metadata retrieved today supports a reconstructed availability assumption, not proof of contemporaneous local receipt or a guarantee no unrecorded intervening revision occurred. Article discovery metadata and summaries do not validate historic execution fills.

## Targeted Sonic official-source gap check

The [official Sonic Labs blog](https://www.soniclabs.com/blog/) and its [RSS feed](https://www.soniclabs.com/blog/rss/) returned HTTP 200. The RSS contains 15 posts, newest publication `Tue, 22 Sep 2026 00:45:54 GMT` (Day 100 update), before the fixed inventory interval. The [official blog post sitemap](https://www.soniclabs.com/blog/sitemap-posts.xml) returned 112 URLs; its newest lastmod is `2026-09-22T17:36:56.000Z`, also for Day 100. Neither supplies a post dated Sep26–Oct12. Sitemap dates are revision dates, not publication dates, so this is a statement about the inspected inventories, not proof that Sonic made no announcement anywhere.

The [former blog sitemap](https://blog.soniclabs.com/sitemap.xml) redirects to the [current blog sitemap index](https://www.soniclabs.com/blog/sitemap.xml), which identifies that post sitemap. The [main site sitemap](https://www.soniclabs.com/sitemap.xml) also returned HTTP 200. Exact raw bytes and HTTP/date/hash receipts are preserved as `research-probes/sonic_{blog,rss,sitemap,old_sitemap,posts_sitemap}.{raw,json}`.

A targeted public search did not yield a directly verifiable official X post for the reported Spartan validator announcement on October 5. Secondary articles and reposts were discovery leads only and were not admitted as official evidence. No timestamp for that event was inferred from them. The requested Sonic S news gap remains explicit; no item was added to normalized evidence by this research task.
