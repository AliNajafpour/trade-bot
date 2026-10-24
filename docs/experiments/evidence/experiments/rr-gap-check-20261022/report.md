# SUSDT native-minute gap verification

The first unresolved minute is 2026-10-12 22:58 UTC. All12minutes through23:09UTC remain absent.

Two existing independently downloaded SUSDT histories contained none of these minutes. Fresh publicWallex HTTP200 responses returned1428 of1440dayminutes and108 of120two-hourminutes. Neither query returned any missing minute. Rawbody and normalizedCSV hashes pass.

No synthetic fill, interpolation, old-data mutation or outcome relabeling occurred. The affected24hourTEST opportunity must remain censored for missing-minute uncertainty.
