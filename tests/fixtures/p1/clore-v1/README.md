# Clore v1 source-shaped fixtures

Provenance: `SOURCE_RECONSTRUCTION_NOT_LIVE`.

These JSON files are derived from the public Clore documentation read on
2026-10-21. They are **not** account captures, provider readbacks, or proof of
live compatibility.

Pinned public pages (SHA256 of the `.md` pages at retrieval time):

- `https://docs.clore.ai/for-hosts/api.md`
  `90563d6154f1871fd134815d670e731dbf7bd6ae0c9774c92bf949ee02d746d7`
- `https://docs.clore.ai/for-renters/order-management.md`
  `07af7262ef0d226d3c3a5830374e655cf9c2ddf251cfcf1b7b0b5ed2cea0fdb4`
- `https://docs.clore.ai/for-renters/fee-structure.md`
  `a727dfa031755da766d3a296e0213e904638a40cfd746017c011d89e74f3caca`

The example values (order IDs, server IDs, spend) are synthetic. `si=112213`
is the owner-selected server ID, not a live order ID. Currency is the
documented `USD-Blockchain` identifier. No credential, header, private key, or
real account data is present.

`marketplace.json` is a contract-shaped reconstruction of the public
`marketplace` endpoint: top-level `code`, `my_servers`, `servers[]`, with each
server carrying `id`, `owner`, `mrl` (hours), `price.on_demand` keyed by exact
currency, `rented`, and `specs`. Its provenance is
`SOURCE_RECONSTRUCTION_FROM_CLORE_PUBLIC_API_DOCS_20261027`, not a live capture.
The owner-forwarded official-response evidence in
`docs/P1-owner-financial-risk-20261026.md` independently records the same
`owner/mrl/rented/price` field family for server 111947. No account or
`api.clore.ai` marketplace request was made by this fixture work.

Unknowns deliberately not encoded as success: there is no documented order
`status` field, no pagination contract, no proof that absence from `my_orders`
or `expired=true` means billing has stopped, and no documented image-digest
support.
