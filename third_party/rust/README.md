# Native Rust dependency licenses

Generated from the locked `flare-cli/Cargo.toml` dependency graph. Python and
Heliopause are preparation/build-only and are not part of this binary release.

This inventory includes runtime dependencies and build/proc-macro tools,
with target-specific roles in `packages.json`. Build-tool notices are included
conservatively; their presence does not mean those tools are shipped.

License expressions are preserved as declared, including legacy slash notation.
Embedded native code may have additional terms in the linked license files.

## (Apache-2.0 OR MIT) AND BSD-3-Clause

| Package | Role | Source | License texts |
| --- | --- | --- | --- |
| encoding_rs 0.8.42 | Runtime | [source](https://crates.io/api/v1/crates/encoding_rs/0.8.42/download) | [COPYRIGHT](encoding_rs-0.8.42/COPYRIGHT), [LICENSE-APACHE](encoding_rs-0.8.42/LICENSE-APACHE), [LICENSE-MIT](encoding_rs-0.8.42/LICENSE-MIT), [LICENSE-WHATWG](encoding_rs-0.8.42/LICENSE-WHATWG) |

## (MIT OR Apache-2.0) AND Unicode-3.0

| Package | Role | Source | License texts |
| --- | --- | --- | --- |
| unicode-ident 1.0.26 | Build/proc-macro only | [source](https://crates.io/api/v1/crates/unicode-ident/1.0.26/download) | [LICENSE-APACHE](unicode-ident-1.0.26/LICENSE-APACHE), [LICENSE-MIT](unicode-ident-1.0.26/LICENSE-MIT), [LICENSE-UNICODE](unicode-ident-1.0.26/LICENSE-UNICODE) |

## 0BSD OR MIT OR Apache-2.0

| Package | Role | Source | License texts |
| --- | --- | --- | --- |
| adler2 2.0.1 | Runtime | [source](https://crates.io/api/v1/crates/adler2/2.0.1/download) | [LICENSE-0BSD](adler2-2.0.1/LICENSE-0BSD), [LICENSE-APACHE](adler2-2.0.1/LICENSE-APACHE), [LICENSE-MIT](adler2-2.0.1/LICENSE-MIT) |

## Apache-2.0

| Package | Role | Source | License texts |
| --- | --- | --- | --- |
| esaxx-rs 0.1.10 | Runtime | [source](https://crates.io/api/v1/crates/esaxx-rs/0.1.10/download) | [LICENSE](esaxx-rs-0.1.10/LICENSE) |
| hf-hub 0.4.3 | Runtime | [source](https://crates.io/api/v1/crates/hf-hub/0.4.3/download) | [LICENSE](hf-hub-0.4.3/LICENSE) |
| hf-hub 0.5.0 | Runtime | [source](https://crates.io/api/v1/crates/hf-hub/0.5.0/download) | [LICENSE](hf-hub-0.5.0/LICENSE) |
| prost 0.14.4 | Runtime | [source](https://crates.io/api/v1/crates/prost/0.14.4/download) | [LICENSE](prost-0.14.4/LICENSE) |
| prost-build 0.14.4 | Build/proc-macro only | [source](https://crates.io/api/v1/crates/prost-build/0.14.4/download) | [LICENSE](prost-build-0.14.4/LICENSE) |
| prost-derive 0.14.4 | Build/proc-macro only | [source](https://crates.io/api/v1/crates/prost-derive/0.14.4/download) | [LICENSE](prost-derive-0.14.4/LICENSE) |
| prost-types 0.14.4 | Build/proc-macro only | [source](https://crates.io/api/v1/crates/prost-types/0.14.4/download) | [LICENSE](prost-types-0.14.4/LICENSE) |
| pulldown-cmark-to-cmark 22.0.3 | Build/proc-macro only | [source](https://crates.io/api/v1/crates/pulldown-cmark-to-cmark/22.0.3/download) | [LICENSE-APACHE](pulldown-cmark-to-cmark-22.0.3/LICENSE-APACHE) |
| spm_precompiled 0.1.4 | Runtime | [source](https://crates.io/api/v1/crates/spm_precompiled/0.1.4/download) | [LICENSE](spm_precompiled-0.1.4/LICENSE) |
| sync_wrapper 1.0.2 | Runtime | [source](https://crates.io/api/v1/crates/sync_wrapper/1.0.2/download) | [LICENSE](sync_wrapper-1.0.2/LICENSE) |
| tokenizers 0.23.2 | Runtime | [source](https://crates.io/api/v1/crates/tokenizers/0.23.2/download) | [LICENSE](tokenizers-0.23.2/LICENSE) |
| unicode-general-category 1.1.0 | Runtime | [source](https://crates.io/api/v1/crates/unicode-general-category/1.1.0/download) | [LICENSE](unicode-general-category-1.1.0/LICENSE) |

## Apache-2.0 / MIT

| Package | Role | Source | License texts |
| --- | --- | --- | --- |
| fnv 1.0.7 | Runtime | [source](https://crates.io/api/v1/crates/fnv/1.0.7/download) | [LICENSE-APACHE](fnv-1.0.7/LICENSE-APACHE), [LICENSE-MIT](fnv-1.0.7/LICENSE-MIT) |

## Apache-2.0 AND ISC

| Package | Role | Source | License texts |
| --- | --- | --- | --- |
| ring 0.17.14 | Runtime | [source](https://crates.io/api/v1/crates/ring/0.17.14/download) | [LICENSE](ring-0.17.14/LICENSE), [LICENSE-BoringSSL](ring-0.17.14/LICENSE-BoringSSL), [LICENSE-other-bits](ring-0.17.14/LICENSE-other-bits), [LICENSE-APACHE](ring-0.17.14/src/polyfill/once_cell/LICENSE-APACHE), [LICENSE-MIT](ring-0.17.14/src/polyfill/once_cell/LICENSE-MIT), [LICENSE](ring-0.17.14/third_party/fiat/LICENSE) |

## Apache-2.0 OR BSL-1.0

| Package | Role | Source | License texts |
| --- | --- | --- | --- |
| ryu 1.0.23 | Runtime | [source](https://crates.io/api/v1/crates/ryu/1.0.23/download) | [LICENSE-APACHE](ryu-1.0.23/LICENSE-APACHE), [LICENSE-BOOST](ryu-1.0.23/LICENSE-BOOST) |

## Apache-2.0 OR ISC OR MIT

| Package | Role | Source | License texts |
| --- | --- | --- | --- |
| rustls 0.23.46 | Runtime | [source](https://crates.io/api/v1/crates/rustls/0.23.46/download) | [LICENSE-APACHE](rustls-0.23.46/LICENSE-APACHE), [LICENSE-ISC](rustls-0.23.46/LICENSE-ISC), [LICENSE-MIT](rustls-0.23.46/LICENSE-MIT) |

## Apache-2.0 OR MIT

| Package | Role | Source | License texts |
| --- | --- | --- | --- |
| atomic-waker 1.1.2 | Runtime | [source](https://crates.io/api/v1/crates/atomic-waker/1.1.2/download) | [LICENSE-APACHE](atomic-waker-1.1.2/LICENSE-APACHE), [LICENSE-MIT](atomic-waker-1.1.2/LICENSE-MIT), [LICENSE-THIRD-PARTY](atomic-waker-1.1.2/LICENSE-THIRD-PARTY) |
| autocfg 1.5.1 | Build/proc-macro only | [source](https://crates.io/api/v1/crates/autocfg/1.5.1/download) | [LICENSE-APACHE](autocfg-1.5.1/LICENSE-APACHE), [LICENSE-MIT](autocfg-1.5.1/LICENSE-MIT) |
| bit-set 0.8.0 | Runtime | [source](https://crates.io/api/v1/crates/bit-set/0.8.0/download) | [LICENSE-APACHE](bit-set-0.8.0/LICENSE-APACHE), [LICENSE-MIT](bit-set-0.8.0/LICENSE-MIT) |
| bit-vec 0.8.0 | Runtime | [source](https://crates.io/api/v1/crates/bit-vec/0.8.0/download) | [LICENSE-APACHE](bit-vec-0.8.0/LICENSE-APACHE), [LICENSE-MIT](bit-vec-0.8.0/LICENSE-MIT) |
| encode_unicode 1.0.0 | Runtime | [source](https://crates.io/api/v1/crates/encode_unicode/1.0.0/download) | [LICENSE-APACHE](encode_unicode-1.0.0/LICENSE-APACHE), [LICENSE-MIT](encode_unicode-1.0.0/LICENSE-MIT) |
| equivalent 1.0.2 | Runtime | [source](https://crates.io/api/v1/crates/equivalent/1.0.2/download) | [LICENSE-APACHE](equivalent-1.0.2/LICENSE-APACHE), [LICENSE-MIT](equivalent-1.0.2/LICENSE-MIT) |
| fastrand 2.5.0 | Runtime | [source](https://crates.io/api/v1/crates/fastrand/2.5.0/download) | [LICENSE-APACHE](fastrand-2.5.0/LICENSE-APACHE), [LICENSE-MIT](fastrand-2.5.0/LICENSE-MIT) |
| idna_adapter 1.2.2 | Runtime | [source](https://crates.io/api/v1/crates/idna_adapter/1.2.2/download) | [LICENSE-APACHE](idna_adapter-1.2.2/LICENSE-APACHE), [LICENSE-MIT](idna_adapter-1.2.2/LICENSE-MIT) |
| indexmap 2.14.2 | Runtime | [source](https://crates.io/api/v1/crates/indexmap/2.14.2/download) | [LICENSE-APACHE](indexmap-2.14.2/LICENSE-APACHE), [LICENSE-MIT](indexmap-2.14.2/LICENSE-MIT) |
| multiversion_no_op 1.0.0 | Build/proc-macro only | [source](https://crates.io/api/v1/crates/multiversion_no_op/1.0.0/download) | [COPYRIGHT](multiversion_no_op-1.0.0/COPYRIGHT), [LICENSE-APACHE](multiversion_no_op-1.0.0/LICENSE-APACHE), [LICENSE-MIT](multiversion_no_op-1.0.0/LICENSE-MIT) |
| pin-project 1.1.13 | Runtime | [source](https://crates.io/api/v1/crates/pin-project/1.1.13/download) | [LICENSE-APACHE](pin-project-1.1.13/LICENSE-APACHE), [LICENSE-MIT](pin-project-1.1.13/LICENSE-MIT) |
| pin-project-internal 1.1.13 | Build/proc-macro only | [source](https://crates.io/api/v1/crates/pin-project-internal/1.1.13/download) | [LICENSE-APACHE](pin-project-internal-1.1.13/LICENSE-APACHE), [LICENSE-MIT](pin-project-internal-1.1.13/LICENSE-MIT) |
| pin-project-lite 0.2.17 | Runtime | [source](https://crates.io/api/v1/crates/pin-project-lite/0.2.17/download) | [LICENSE-APACHE](pin-project-lite-0.2.17/LICENSE-APACHE), [LICENSE-MIT](pin-project-lite-0.2.17/LICENSE-MIT) |
| portable-atomic 1.15.0 | Runtime | [source](https://crates.io/api/v1/crates/portable-atomic/1.15.0/download) | [LICENSE-APACHE](portable-atomic-1.15.0/LICENSE-APACHE), [LICENSE-MIT](portable-atomic-1.15.0/LICENSE-MIT) |
| utf8_iter 1.0.4 | Runtime | [source](https://crates.io/api/v1/crates/utf8_iter/1.0.4/download) | [COPYRIGHT](utf8_iter-1.0.4/COPYRIGHT), [LICENSE-APACHE](utf8_iter-1.0.4/LICENSE-APACHE), [LICENSE-MIT](utf8_iter-1.0.4/LICENSE-MIT) |
| utf8parse 0.2.2 | Runtime | [source](https://crates.io/api/v1/crates/utf8parse/0.2.2/download) | [LICENSE-APACHE](utf8parse-0.2.2/LICENSE-APACHE), [LICENSE-MIT](utf8parse-0.2.2/LICENSE-MIT) |
| uuid 1.26.1 | Runtime | [source](https://crates.io/api/v1/crates/uuid/1.26.1/download) | [LICENSE-APACHE](uuid-1.26.1/LICENSE-APACHE), [LICENSE-MIT](uuid-1.26.1/LICENSE-MIT) |
| zeroize 1.9.1 | Runtime | [source](https://crates.io/api/v1/crates/zeroize/1.9.1/download) | [LICENSE-APACHE](zeroize-1.9.1/LICENSE-APACHE), [LICENSE-MIT](zeroize-1.9.1/LICENSE-MIT) |

## Apache-2.0 OR MIT OR Zlib

| Package | Role | Source | License texts |
| --- | --- | --- | --- |
| macro_rules_attribute 0.2.3 | Runtime | [source](https://crates.io/api/v1/crates/macro_rules_attribute/0.2.3/download) | [LICENSE-APACHE](macro_rules_attribute-0.2.3/LICENSE-APACHE), [LICENSE-MIT](macro_rules_attribute-0.2.3/LICENSE-MIT), [LICENSE-ZLIB](macro_rules_attribute-0.2.3/LICENSE-ZLIB) |
| macro_rules_attribute-proc_macro 0.2.3 | Build/proc-macro only | [source](https://crates.io/api/v1/crates/macro_rules_attribute-proc_macro/0.2.3/download) | [LICENSE-APACHE](macro_rules_attribute-proc_macro-0.2.3/LICENSE-APACHE), [LICENSE-MIT](macro_rules_attribute-proc_macro-0.2.3/LICENSE-MIT), [LICENSE-ZLIB](macro_rules_attribute-proc_macro-0.2.3/LICENSE-ZLIB) |

## Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT

| Package | Role | Source | License texts |
| --- | --- | --- | --- |
| linux-raw-sys 0.12.1 | Runtime | [source](https://crates.io/api/v1/crates/linux-raw-sys/0.12.1/download) | [COPYRIGHT](linux-raw-sys-0.12.1/COPYRIGHT), [LICENSE-APACHE](linux-raw-sys-0.12.1/LICENSE-APACHE), [LICENSE-Apache-2.0_WITH_LLVM-exception](linux-raw-sys-0.12.1/LICENSE-Apache-2.0_WITH_LLVM-exception), [LICENSE-MIT](linux-raw-sys-0.12.1/LICENSE-MIT) |
| linux-raw-sys 0.4.15 | Runtime | [source](https://crates.io/api/v1/crates/linux-raw-sys/0.4.15/download) | [COPYRIGHT](linux-raw-sys-0.4.15/COPYRIGHT), [LICENSE-APACHE](linux-raw-sys-0.4.15/LICENSE-APACHE), [LICENSE-Apache-2.0_WITH_LLVM-exception](linux-raw-sys-0.4.15/LICENSE-Apache-2.0_WITH_LLVM-exception), [LICENSE-MIT](linux-raw-sys-0.4.15/LICENSE-MIT) |
| rustix 0.38.44 | Runtime | [source](https://crates.io/api/v1/crates/rustix/0.38.44/download) | [COPYRIGHT](rustix-0.38.44/COPYRIGHT), [LICENSE-APACHE](rustix-0.38.44/LICENSE-APACHE), [LICENSE-Apache-2.0_WITH_LLVM-exception](rustix-0.38.44/LICENSE-Apache-2.0_WITH_LLVM-exception), [LICENSE-MIT](rustix-0.38.44/LICENSE-MIT) |
| rustix 1.1.5 | Runtime | [source](https://crates.io/api/v1/crates/rustix/1.1.5/download) | [COPYRIGHT](rustix-1.1.5/COPYRIGHT), [LICENSE-APACHE](rustix-1.1.5/LICENSE-APACHE), [LICENSE-Apache-2.0_WITH_LLVM-exception](rustix-1.1.5/LICENSE-Apache-2.0_WITH_LLVM-exception), [LICENSE-MIT](rustix-1.1.5/LICENSE-MIT) |

## Apache-2.0/MIT

| Package | Role | Source | License texts |
| --- | --- | --- | --- |
| bytecount 0.6.9 | Runtime | [source](https://crates.io/api/v1/crates/bytecount/0.6.9/download) | [LICENSE.Apache2](bytecount-0.6.9/LICENSE.Apache2), [LICENSE.MIT](bytecount-0.6.9/LICENSE.MIT) |
| rayon-cond 0.4.0 | Runtime | [source](https://crates.io/api/v1/crates/rayon-cond/0.4.0/download) | [LICENSE-APACHE](rayon-cond-0.4.0/LICENSE-APACHE), [LICENSE-MIT](rayon-cond-0.4.0/LICENSE-MIT) |
| signal-hook 0.3.18 | Runtime | [source](https://crates.io/api/v1/crates/signal-hook/0.3.18/download) | [LICENSE-APACHE](signal-hook-0.3.18/LICENSE-APACHE), [LICENSE-MIT](signal-hook-0.3.18/LICENSE-MIT) |

## BSD-2-Clause OR Apache-2.0 OR MIT

| Package | Role | Source | License texts |
| --- | --- | --- | --- |
| zerocopy 0.8.62 | Runtime | [source](https://crates.io/api/v1/crates/zerocopy/0.8.62/download) | [LICENSE-APACHE](zerocopy-0.8.62/LICENSE-APACHE), [LICENSE-BSD](zerocopy-0.8.62/LICENSE-BSD), [LICENSE-MIT](zerocopy-0.8.62/LICENSE-MIT) |

## BSD-3-Clause

| Package | Role | Source | License texts |
| --- | --- | --- | --- |
| alloc-no-stdlib 2.0.4 | Runtime | [source](https://crates.io/api/v1/crates/alloc-no-stdlib/2.0.4/download) | [LICENSE](alloc-no-stdlib-2.0.4/LICENSE) |
| alloc-stdlib 0.2.4 | Runtime | [source](https://crates.io/api/v1/crates/alloc-stdlib/0.2.4/download) | [LICENSE](alloc-stdlib-0.2.4/upstream/LICENSE) |
| subtle 2.6.1 | Runtime | [source](https://crates.io/api/v1/crates/subtle/2.6.1/download) | [LICENSE](subtle-2.6.1/LICENSE) |
| zstd-safe 7.3.0 | Runtime | [source](https://crates.io/api/v1/crates/zstd-safe/7.3.0/download) | [LICENSE](zstd-safe-7.3.0/LICENSE) |
| zstd-sys 2.1.1+zstd.1.5.7 | Runtime | [source](https://crates.io/api/v1/crates/zstd-sys/2.1.1+zstd.1.5.7/download) | [LICENSE](zstd-sys-2.1.1+zstd.1.5.7/LICENSE), [LICENSE.BSD-3-Clause](zstd-sys-2.1.1+zstd.1.5.7/LICENSE.BSD-3-Clause), [COPYING](zstd-sys-2.1.1+zstd.1.5.7/zstd/COPYING), [LICENSE](zstd-sys-2.1.1+zstd.1.5.7/zstd/LICENSE) |

## BSD-3-Clause AND MIT

| Package | Role | Source | License texts |
| --- | --- | --- | --- |
| brotli 8.0.4 | Runtime | [source](https://crates.io/api/v1/crates/brotli/8.0.4/download) | [LICENSE.BSD-3-Clause](brotli-8.0.4/LICENSE.BSD-3-Clause), [LICENSE.MIT](brotli-8.0.4/LICENSE.MIT) |

## BSD-3-Clause/MIT

| Package | Role | Source | License texts |
| --- | --- | --- | --- |
| brotli-decompressor 5.0.3 | Runtime | [source](https://crates.io/api/v1/crates/brotli-decompressor/5.0.3/download) | [LICENSE](brotli-decompressor-5.0.3/LICENSE) |

## BSL-1.0

| Package | Role | Source | License texts |
| --- | --- | --- | --- |
| xxhash-rust 0.8.19 | Runtime | [source](https://crates.io/api/v1/crates/xxhash-rust/0.8.19/download) | [LICENSE](xxhash-rust-0.8.19/LICENSE) |

## CDLA-Permissive-2.0

| Package | Role | Source | License texts |
| --- | --- | --- | --- |
| webpki-roots 0.26.11 | Runtime | [source](https://crates.io/api/v1/crates/webpki-roots/0.26.11/download) | [LICENSE](webpki-roots-0.26.11/LICENSE) |
| webpki-roots 1.0.9 | Runtime | [source](https://crates.io/api/v1/crates/webpki-roots/1.0.9/download) | [LICENSE](webpki-roots-1.0.9/LICENSE) |

## ISC

| Package | Role | Source | License texts |
| --- | --- | --- | --- |
| libloading 0.8.9 | Runtime | [source](https://crates.io/api/v1/crates/libloading/0.8.9/download) | [LICENSE](libloading-0.8.9/LICENSE) |
| rustls-webpki 0.103.15 | Runtime | [source](https://crates.io/api/v1/crates/rustls-webpki/0.103.15/download) | [LICENSE](rustls-webpki-0.103.15/LICENSE) |
| untrusted 0.9.0 | Runtime | [source](https://crates.io/api/v1/crates/untrusted/0.9.0/download) | [LICENSE.txt](untrusted-0.9.0/LICENSE.txt) |

## MIT

| Package | Role | Source | License texts |
| --- | --- | --- | --- |
| async-stream 0.3.6 | Runtime | [source](https://crates.io/api/v1/crates/async-stream/0.3.6/download) | [LICENSE](async-stream-0.3.6/LICENSE) |
| async-stream-impl 0.3.6 | Build/proc-macro only | [source](https://crates.io/api/v1/crates/async-stream-impl/0.3.6/download) | [LICENSE](async-stream-impl-0.3.6/LICENSE) |
| axum 0.8.9 | Runtime | [source](https://crates.io/api/v1/crates/axum/0.8.9/download) | [LICENSE](axum-0.8.9/LICENSE) |
| axum-core 0.5.6 | Runtime | [source](https://crates.io/api/v1/crates/axum-core/0.5.6/download) | [LICENSE](axum-core-0.5.6/LICENSE) |
| bytes 1.12.1 | Runtime | [source](https://crates.io/api/v1/crates/bytes/1.12.1/download) | [LICENSE](bytes-1.12.1/LICENSE) |
| castaway 0.2.4 | Runtime | [source](https://crates.io/api/v1/crates/castaway/0.2.4/download) | [LICENSE](castaway-0.2.4/LICENSE) |
| compact_str 0.8.2 | Runtime | [source](https://crates.io/api/v1/crates/compact_str/0.8.2/download) | [LICENSE](compact_str-0.8.2/LICENSE) |
| compact_str 0.9.1 | Runtime | [source](https://crates.io/api/v1/crates/compact_str/0.9.1/download) | [LICENSE](compact_str-0.9.1/LICENSE) |
| console 0.15.11 | Runtime | [source](https://crates.io/api/v1/crates/console/0.15.11/download) | [LICENSE](console-0.15.11/LICENSE) |
| console 0.16.6 | Runtime | [source](https://crates.io/api/v1/crates/console/0.16.6/download) | [LICENSE](console-0.16.6/LICENSE) |
| crossterm 0.28.1 | Runtime | [source](https://crates.io/api/v1/crates/crossterm/0.28.1/download) | [LICENSE](crossterm-0.28.1/LICENSE) |
| crossterm_winapi 0.9.1 | Runtime | [source](https://crates.io/api/v1/crates/crossterm_winapi/0.9.1/download) | [LICENSE](crossterm_winapi-0.9.1/LICENSE) |
| darling 0.20.11 | Build/proc-macro only | [source](https://crates.io/api/v1/crates/darling/0.20.11/download) | [LICENSE](darling-0.20.11/LICENSE) |
| darling_core 0.20.11 | Build/proc-macro only | [source](https://crates.io/api/v1/crates/darling_core/0.20.11/download) | [LICENSE](darling_core-0.20.11/LICENSE) |
| darling_macro 0.20.11 | Build/proc-macro only | [source](https://crates.io/api/v1/crates/darling_macro/0.20.11/download) | [LICENSE](darling_macro-0.20.11/LICENSE) |
| data-encoding 2.11.1 | Runtime | [source](https://crates.io/api/v1/crates/data-encoding/2.11.1/download) | [LICENSE](data-encoding-2.11.1/LICENSE) |
| deepseek-recipe 0.2.0 | Runtime | [source](https://github.com/local-inference-lab/deepseek-recipe/tree/6d09e9be68dbf4ae488f6b22ee9114dcf9ea6270) | [LICENSE](deepseek-recipe-0.2.0/LICENSE) |
| deepseek-recipe-core 0.2.0 | Runtime | [source](https://github.com/local-inference-lab/deepseek-recipe/tree/6d09e9be68dbf4ae488f6b22ee9114dcf9ea6270) | [LICENSE](deepseek-recipe-core-0.2.0/LICENSE) |
| deepseek-recipe-encoding 0.2.0 | Runtime | [source](https://github.com/local-inference-lab/deepseek-recipe/tree/6d09e9be68dbf4ae488f6b22ee9114dcf9ea6270) | [LICENSE](deepseek-recipe-encoding-0.2.0/LICENSE) |
| deepseek-recipe-macros 0.2.0 | Build/proc-macro only | [source](https://github.com/local-inference-lab/deepseek-recipe/tree/6d09e9be68dbf4ae488f6b22ee9114dcf9ea6270) | [LICENSE](deepseek-recipe-macros-0.2.0/LICENSE) |
| derivre 0.3.13 | Runtime | [source](https://crates.io/api/v1/crates/derivre/0.3.13/download) | [LICENSE](derivre-0.3.13/LICENSE) |
| email_address 0.2.9 | Runtime | [source](https://crates.io/api/v1/crates/email_address/0.2.9/download) | [LICENSE](email_address-0.2.9/LICENSE) |
| fancy-regex 0.14.0 | Runtime | [source](https://crates.io/api/v1/crates/fancy-regex/0.14.0/download) | [LICENSE](fancy-regex-0.14.0/LICENSE) |
| fancy-regex 0.19.2 | Runtime | [source](https://crates.io/api/v1/crates/fancy-regex/0.19.2/download) | [LICENSE](fancy-regex-0.19.2/LICENSE) |
| fluent-uri 0.3.2 | Runtime | [source](https://crates.io/api/v1/crates/fluent-uri/0.3.2/download) | [LICENSE](fluent-uri-0.3.2/LICENSE) |
| fluent-uri 0.4.1 | Runtime | [source](https://crates.io/api/v1/crates/fluent-uri/0.4.1/download) | [LICENSE](fluent-uri-0.4.1/LICENSE) |
| generic-array 0.14.7 | Runtime | [source](https://crates.io/api/v1/crates/generic-array/0.14.7/download) | [LICENSE](generic-array-0.14.7/LICENSE) |
| h2 0.4.20 | Runtime | [source](https://crates.io/api/v1/crates/h2/0.4.20/download) | [LICENSE](h2-0.4.20/LICENSE) |
| http-body 1.1.0 | Runtime | [source](https://crates.io/api/v1/crates/http-body/1.1.0/download) | [LICENSE](http-body-1.1.0/LICENSE) |
| http-body-util 0.1.5 | Runtime | [source](https://crates.io/api/v1/crates/http-body-util/0.1.5/download) | [LICENSE](http-body-util-0.1.5/LICENSE) |
| hyper 1.11.1 | Runtime | [source](https://crates.io/api/v1/crates/hyper/1.11.1/download) | [LICENSE](hyper-1.11.1/LICENSE) |
| hyper-util 0.1.20 | Runtime | [source](https://crates.io/api/v1/crates/hyper-util/0.1.20/download) | [LICENSE](hyper-util-0.1.20/LICENSE) |
| indicatif 0.17.11 | Runtime | [source](https://crates.io/api/v1/crates/indicatif/0.17.11/download) | [LICENSE](indicatif-0.17.11/LICENSE) |
| indicatif 0.18.6 | Runtime | [source](https://crates.io/api/v1/crates/indicatif/0.18.6/download) | [LICENSE](indicatif-0.18.6/LICENSE) |
| instability 0.3.10 | Build/proc-macro only | [source](https://crates.io/api/v1/crates/instability/0.3.10/download) | [LICENSE.md](instability-0.3.10/LICENSE.md) |
| jsonschema 0.29.1 | Runtime | [source](https://crates.io/api/v1/crates/jsonschema/0.29.1/download) | [LICENSE](jsonschema-0.29.1/LICENSE) |
| jsonschema 0.58.6 | Runtime | [source](https://crates.io/api/v1/crates/jsonschema/0.58.6/download) | [LICENSE](jsonschema-0.58.6/LICENSE) |
| jsonschema-regex 0.58.6 | Runtime | [source](https://crates.io/api/v1/crates/jsonschema-regex/0.58.6/download) | [LICENSE](jsonschema-regex-0.58.6/upstream/LICENSE) |
| jsonschema-value 0.58.6 | Runtime | [source](https://crates.io/api/v1/crates/jsonschema-value/0.58.6/download) | [LICENSE](jsonschema-value-0.58.6/upstream/LICENSE) |
| llguidance 1.9.1 | Runtime | [source](https://crates.io/api/v1/crates/llguidance/1.9.1/download) | [LICENSE](llguidance-1.9.1/LICENSE) |
| lru 0.12.5 | Runtime | [source](https://crates.io/api/v1/crates/lru/0.12.5/download) | [LICENSE](lru-0.12.5/LICENSE) |
| micromap 0.3.0 | Runtime | [source](https://crates.io/api/v1/crates/micromap/0.3.0/download) | [copyrights.yml](micromap-0.3.0/.github/workflows/copyrights.yml), [LICENSE.txt](micromap-0.3.0/LICENSE.txt) |
| mio 1.2.3 | Runtime | [source](https://crates.io/api/v1/crates/mio/1.2.3/download) | [LICENSE](mio-1.2.3/LICENSE) |
| multer 3.1.0 | Runtime | [source](https://crates.io/api/v1/crates/multer/3.1.0/download) | [LICENSE](multer-3.1.0/LICENSE) |
| nom 7.1.3 | Runtime | [source](https://crates.io/api/v1/crates/nom/7.1.3/download) | [LICENSE](nom-7.1.3/LICENSE) |
| number_prefix 0.4.0 | Runtime | [source](https://crates.io/api/v1/crates/number_prefix/0.4.0/download) | [LICENCE](number_prefix-0.4.0/upstream/LICENCE) |
| onig 6.5.3 | Runtime | [source](https://crates.io/api/v1/crates/onig/6.5.3/download) | [LICENSE.md](onig-6.5.3/LICENSE.md) |
| onig_sys 69.9.3 | Runtime | [source](https://crates.io/api/v1/crates/onig_sys/69.9.3/download) | [LICENSE.md](onig_sys-69.9.3/LICENSE.md), [COPYING](onig_sys-69.9.3/oniguruma/COPYING) |
| outref 0.5.2 | Runtime | [source](https://crates.io/api/v1/crates/outref/0.5.2/download) | [LICENSE](outref-0.5.2/LICENSE) |
| pulldown-cmark 0.13.5 | Build/proc-macro only | [source](https://crates.io/api/v1/crates/pulldown-cmark/0.13.5/download) | [LICENSE](pulldown-cmark-0.13.5/LICENSE) |
| ratatui 0.29.0 | Runtime | [source](https://crates.io/api/v1/crates/ratatui/0.29.0/download) | [LICENSE](ratatui-0.29.0/LICENSE) |
| referencing 0.29.1 | Runtime | [source](https://crates.io/api/v1/crates/referencing/0.29.1/download) | [LICENSE](referencing-0.29.1/LICENSE) |
| referencing 0.58.6 | Runtime | [source](https://crates.io/api/v1/crates/referencing/0.58.6/download) | [LICENSE](referencing-0.58.6/LICENSE) |
| sha2-asm 0.6.4 | Runtime | [source](https://crates.io/api/v1/crates/sha2-asm/0.6.4/download) | [LICENSE](sha2-asm-0.6.4/LICENSE) |
| simd-adler32 0.3.10 | Runtime | [source](https://crates.io/api/v1/crates/simd-adler32/0.3.10/download) | [LICENSE.md](simd-adler32-0.3.10/LICENSE.md) |
| slab 0.4.12 | Runtime | [source](https://crates.io/api/v1/crates/slab/0.4.12/download) | [LICENSE](slab-0.4.12/LICENSE) |
| spin 0.9.9 | Runtime | [source](https://crates.io/api/v1/crates/spin/0.9.9/download) | [LICENSE](spin-0.9.9/LICENSE) |
| strsim 0.11.1 | Runtime | [source](https://crates.io/api/v1/crates/strsim/0.11.1/download) | [LICENSE](strsim-0.11.1/LICENSE) |
| strum 0.26.3 | Runtime | [source](https://crates.io/api/v1/crates/strum/0.26.3/download) | [LICENSE](strum-0.26.3/LICENSE) |
| strum 0.28.0 | Runtime | [source](https://crates.io/api/v1/crates/strum/0.28.0/download) | [LICENSE](strum-0.28.0/LICENSE) |
| strum_macros 0.26.4 | Build/proc-macro only | [source](https://crates.io/api/v1/crates/strum_macros/0.26.4/download) | [LICENSE](strum_macros-0.26.4/LICENSE) |
| strum_macros 0.28.0 | Build/proc-macro only | [source](https://crates.io/api/v1/crates/strum_macros/0.28.0/download) | [LICENSE](strum_macros-0.28.0/LICENSE) |
| synstructure 0.14.0 | Build/proc-macro only | [source](https://crates.io/api/v1/crates/synstructure/0.14.0/download) | [LICENSE](synstructure-0.14.0/LICENSE) |
| tokio 1.53.1 | Runtime | [source](https://crates.io/api/v1/crates/tokio/1.53.1/download) | [LICENSE](tokio-1.53.1/LICENSE) |
| tokio-macros 2.7.2 | Build/proc-macro only | [source](https://crates.io/api/v1/crates/tokio-macros/2.7.2/download) | [LICENSE](tokio-macros-2.7.2/LICENSE) |
| tokio-stream 0.1.19 | Runtime | [source](https://crates.io/api/v1/crates/tokio-stream/0.1.19/download) | [LICENSE](tokio-stream-0.1.19/LICENSE) |
| tokio-util 0.7.20 | Runtime | [source](https://crates.io/api/v1/crates/tokio-util/0.7.20/download) | [LICENSE](tokio-util-0.7.20/LICENSE) |
| toktrie 1.9.1 | Runtime | [source](https://crates.io/api/v1/crates/toktrie/1.9.1/download) | [LICENSE](toktrie-1.9.1/LICENSE) |
| tonic 0.14.6 | Runtime | [source](https://crates.io/api/v1/crates/tonic/0.14.6/download) | [LICENSE](tonic-0.14.6/LICENSE) |
| tonic-build 0.14.6 | Build/proc-macro only | [source](https://crates.io/api/v1/crates/tonic-build/0.14.6/download) | [LICENSE](tonic-build-0.14.6/LICENSE) |
| tonic-prost 0.14.6 | Runtime | [source](https://crates.io/api/v1/crates/tonic-prost/0.14.6/download) | [LICENSE](tonic-prost-0.14.6/upstream/LICENSE) |
| tonic-prost-build 0.14.6 | Build/proc-macro only | [source](https://crates.io/api/v1/crates/tonic-prost-build/0.14.6/download) | [LICENSE](tonic-prost-build-0.14.6/upstream/LICENSE) |
| tower 0.5.3 | Runtime | [source](https://crates.io/api/v1/crates/tower/0.5.3/download) | [LICENSE](tower-0.5.3/LICENSE) |
| tower-layer 0.3.3 | Runtime | [source](https://crates.io/api/v1/crates/tower-layer/0.3.3/download) | [LICENSE](tower-layer-0.3.3/LICENSE) |
| tower-service 0.3.3 | Runtime | [source](https://crates.io/api/v1/crates/tower-service/0.3.3/download) | [LICENSE](tower-service-0.3.3/LICENSE) |
| tracing 0.1.44 | Runtime | [source](https://crates.io/api/v1/crates/tracing/0.1.44/download) | [LICENSE](tracing-0.1.44/LICENSE) |
| tracing-attributes 0.1.31 | Build/proc-macro only | [source](https://crates.io/api/v1/crates/tracing-attributes/0.1.31/download) | [LICENSE](tracing-attributes-0.1.31/LICENSE) |
| tracing-core 0.1.36 | Runtime | [source](https://crates.io/api/v1/crates/tracing-core/0.1.36/download) | [LICENSE](tracing-core-0.1.36/LICENSE), [LICENSE](tracing-core-0.1.36/src/spin/LICENSE) |
| try-lock 0.2.5 | Runtime | [source](https://crates.io/api/v1/crates/try-lock/0.2.5/download) | [LICENSE](try-lock-0.2.5/LICENSE) |
| twox-hash 2.1.5 | Runtime | [source](https://crates.io/api/v1/crates/twox-hash/2.1.5/download) | [LICENSE.txt](twox-hash-2.1.5/LICENSE.txt) |
| unit-prefix 0.5.2 | Runtime | [source](https://crates.io/api/v1/crates/unit-prefix/0.5.2/download) | [LICENSE](unit-prefix-0.5.2/LICENSE) |
| uuid-simd 0.8.0 | Runtime | [source](https://crates.io/api/v1/crates/uuid-simd/0.8.0/download) | [LICENSE](uuid-simd-0.8.0/upstream/LICENSE) |
| vsimd 0.8.0 | Runtime | [source](https://crates.io/api/v1/crates/vsimd/0.8.0/download) | [LICENSE](vsimd-0.8.0/upstream/LICENSE) |
| want 0.3.2 | Runtime | [source](https://crates.io/api/v1/crates/want/0.3.2/download) | [LICENSE](want-0.3.2/LICENSE) |
| winnow 0.7.15 | Runtime | [source](https://crates.io/api/v1/crates/winnow/0.7.15/download) | [LICENSE-MIT](winnow-0.7.15/LICENSE-MIT) |
| zmij 1.0.23 | Runtime | [source](https://crates.io/api/v1/crates/zmij/1.0.23/download) | [LICENSE-MIT](zmij-1.0.23/LICENSE-MIT) |
| zstd 0.13.3 | Runtime | [source](https://crates.io/api/v1/crates/zstd/0.13.3/download) | [LICENSE](zstd-0.13.3/LICENSE) |

## MIT / Apache-2.0

| Package | Role | Source | License texts |
| --- | --- | --- | --- |
| cassowary 0.3.0 | Runtime | [source](https://crates.io/api/v1/crates/cassowary/0.3.0/download) | [LICENSE-APACHE](cassowary-0.3.0/LICENSE-APACHE), [LICENSE-MIT](cassowary-0.3.0/LICENSE-MIT) |

## MIT AND BSD-3-Clause

| Package | Role | Source | License texts |
| --- | --- | --- | --- |
| matchit 0.8.4 | Runtime | [source](https://crates.io/api/v1/crates/matchit/0.8.4/download) | [LICENSE](matchit-0.8.4/LICENSE), [LICENSE.httprouter](matchit-0.8.4/LICENSE.httprouter) |

## MIT OR Apache-2.0

| Package | Role | Source | License texts |
| --- | --- | --- | --- |
| ahash 0.8.12 | Runtime | [source](https://crates.io/api/v1/crates/ahash/0.8.12/download) | [LICENSE-APACHE](ahash-0.8.12/LICENSE-APACHE), [LICENSE-MIT](ahash-0.8.12/LICENSE-MIT) |
| allocator-api2 0.2.21 | Runtime | [source](https://crates.io/api/v1/crates/allocator-api2/0.2.21/download) | [LICENSE-APACHE](allocator-api2-0.2.21/LICENSE-APACHE), [LICENSE-MIT](allocator-api2-0.2.21/LICENSE-MIT) |
| anstream 1.0.0 | Runtime | [source](https://crates.io/api/v1/crates/anstream/1.0.0/download) | [LICENSE-APACHE](anstream-1.0.0/LICENSE-APACHE), [LICENSE-MIT](anstream-1.0.0/LICENSE-MIT) |
| anstyle 1.0.14 | Runtime | [source](https://crates.io/api/v1/crates/anstyle/1.0.14/download) | [LICENSE-APACHE](anstyle-1.0.14/LICENSE-APACHE), [LICENSE-MIT](anstyle-1.0.14/LICENSE-MIT) |
| anstyle-parse 1.0.0 | Runtime | [source](https://crates.io/api/v1/crates/anstyle-parse/1.0.0/download) | [LICENSE-APACHE](anstyle-parse-1.0.0/LICENSE-APACHE), [LICENSE-MIT](anstyle-parse-1.0.0/LICENSE-MIT) |
| anstyle-query 1.1.5 | Runtime | [source](https://crates.io/api/v1/crates/anstyle-query/1.1.5/download) | [LICENSE-APACHE](anstyle-query-1.1.5/LICENSE-APACHE), [LICENSE-MIT](anstyle-query-1.1.5/LICENSE-MIT) |
| anstyle-wincon 3.0.11 | Runtime | [source](https://crates.io/api/v1/crates/anstyle-wincon/3.0.11/download) | [LICENSE-APACHE](anstyle-wincon-3.0.11/LICENSE-APACHE), [LICENSE-MIT](anstyle-wincon-3.0.11/LICENSE-MIT) |
| anyhow 1.0.104 | Runtime | [source](https://crates.io/api/v1/crates/anyhow/1.0.104/download) | [LICENSE-APACHE](anyhow-1.0.104/LICENSE-APACHE), [LICENSE-MIT](anyhow-1.0.104/LICENSE-MIT) |
| async-trait 0.1.92 | Build/proc-macro only | [source](https://crates.io/api/v1/crates/async-trait/0.1.92/download) | [LICENSE-APACHE](async-trait-0.1.92/LICENSE-APACHE), [LICENSE-MIT](async-trait-0.1.92/LICENSE-MIT) |
| base64 0.22.1 | Runtime | [source](https://crates.io/api/v1/crates/base64/0.22.1/download) | [LICENSE-APACHE](base64-0.22.1/LICENSE-APACHE), [LICENSE-MIT](base64-0.22.1/LICENSE-MIT) |
| base64 0.23.1 | Runtime | [source](https://crates.io/api/v1/crates/base64/0.23.1/download) | [LICENSE-APACHE](base64-0.23.1/LICENSE-APACHE), [LICENSE-MIT](base64-0.23.1/LICENSE-MIT) |
| bitflags 2.13.2 | Runtime | [source](https://crates.io/api/v1/crates/bitflags/2.13.2/download) | [LICENSE-APACHE](bitflags-2.13.2/LICENSE-APACHE), [LICENSE-MIT](bitflags-2.13.2/LICENSE-MIT) |
| blake2 0.10.6 | Runtime | [source](https://crates.io/api/v1/crates/blake2/0.10.6/download) | [LICENSE-APACHE](blake2-0.10.6/LICENSE-APACHE), [LICENSE-MIT](blake2-0.10.6/LICENSE-MIT) |
| block-buffer 0.10.4 | Runtime | [source](https://crates.io/api/v1/crates/block-buffer/0.10.4/download) | [LICENSE-APACHE](block-buffer-0.10.4/LICENSE-APACHE), [LICENSE-MIT](block-buffer-0.10.4/LICENSE-MIT) |
| cc 1.6.0 | Build/proc-macro only | [source](https://crates.io/api/v1/crates/cc/1.6.0/download) | [LICENSE-APACHE](cc-1.6.0/LICENSE-APACHE), [LICENSE-MIT](cc-1.6.0/LICENSE-MIT) |
| cfg-if 1.0.5 | Runtime | [source](https://crates.io/api/v1/crates/cfg-if/1.0.5/download) | [LICENSE-APACHE](cfg-if-1.0.5/LICENSE-APACHE), [LICENSE-MIT](cfg-if-1.0.5/LICENSE-MIT) |
| clap 4.6.7 | Runtime | [source](https://crates.io/api/v1/crates/clap/4.6.7/download) | [LICENSE-APACHE](clap-4.6.7/LICENSE-APACHE), [LICENSE-MIT](clap-4.6.7/LICENSE-MIT) |
| clap_builder 4.6.7 | Runtime | [source](https://crates.io/api/v1/crates/clap_builder/4.6.7/download) | [LICENSE-APACHE](clap_builder-4.6.7/LICENSE-APACHE), [LICENSE-MIT](clap_builder-4.6.7/LICENSE-MIT) |
| clap_derive 4.6.7 | Build/proc-macro only | [source](https://crates.io/api/v1/crates/clap_derive/4.6.7/download) | [LICENSE-APACHE](clap_derive-4.6.7/LICENSE-APACHE), [LICENSE-MIT](clap_derive-4.6.7/LICENSE-MIT) |
| clap_lex 1.1.1 | Runtime | [source](https://crates.io/api/v1/crates/clap_lex/1.1.1/download) | [LICENSE-APACHE](clap_lex-1.1.1/LICENSE-APACHE), [LICENSE-MIT](clap_lex-1.1.1/LICENSE-MIT) |
| cmake 0.1.58 | Build/proc-macro only | [source](https://crates.io/api/v1/crates/cmake/0.1.58/download) | [LICENSE-APACHE](cmake-0.1.58/LICENSE-APACHE), [LICENSE-MIT](cmake-0.1.58/LICENSE-MIT) |
| colorchoice 1.0.5 | Runtime | [source](https://crates.io/api/v1/crates/colorchoice/1.0.5/download) | [LICENSE-APACHE](colorchoice-1.0.5/LICENSE-APACHE), [LICENSE-MIT](colorchoice-1.0.5/LICENSE-MIT) |
| cookie 0.18.2 | Runtime | [source](https://crates.io/api/v1/crates/cookie/0.18.2/download) | [LICENSE-APACHE](cookie-0.18.2/LICENSE-APACHE), [LICENSE-MIT](cookie-0.18.2/LICENSE-MIT) |
| cookie_store 0.22.0 | Runtime | [source](https://crates.io/api/v1/crates/cookie_store/0.22.0/download) | [LICENSE-APACHE](cookie_store-0.22.0/LICENSE-APACHE), [LICENSE-MIT](cookie_store-0.22.0/LICENSE-MIT) |
| cpufeatures 0.2.17 | Runtime | [source](https://crates.io/api/v1/crates/cpufeatures/0.2.17/download) | [LICENSE-APACHE](cpufeatures-0.2.17/LICENSE-APACHE), [LICENSE-MIT](cpufeatures-0.2.17/LICENSE-MIT) |
| crc32fast 1.5.2 | Runtime | [source](https://crates.io/api/v1/crates/crc32fast/1.5.2/download) | [LICENSE-APACHE](crc32fast-1.5.2/LICENSE-APACHE), [LICENSE-MIT](crc32fast-1.5.2/LICENSE-MIT) |
| crossbeam-deque 0.8.8 | Runtime | [source](https://crates.io/api/v1/crates/crossbeam-deque/0.8.8/download) | [LICENSE-APACHE](crossbeam-deque-0.8.8/LICENSE-APACHE), [LICENSE-MIT](crossbeam-deque-0.8.8/LICENSE-MIT) |
| crossbeam-epoch 0.9.21 | Runtime | [source](https://crates.io/api/v1/crates/crossbeam-epoch/0.9.21/download) | [LICENSE-APACHE](crossbeam-epoch-0.9.21/LICENSE-APACHE), [LICENSE-MIT](crossbeam-epoch-0.9.21/LICENSE-MIT) |
| crossbeam-utils 0.8.23 | Runtime | [source](https://crates.io/api/v1/crates/crossbeam-utils/0.8.23/download) | [LICENSE-APACHE](crossbeam-utils-0.8.23/LICENSE-APACHE), [LICENSE-MIT](crossbeam-utils-0.8.23/LICENSE-MIT) |
| crypto-common 0.1.7 | Runtime | [source](https://crates.io/api/v1/crates/crypto-common/0.1.7/download) | [LICENSE-APACHE](crypto-common-0.1.7/LICENSE-APACHE), [LICENSE-MIT](crypto-common-0.1.7/LICENSE-MIT) |
| daachorse 3.0.3 | Runtime | [source](https://crates.io/api/v1/crates/daachorse/3.0.3/download) | [LICENSE-APACHE](daachorse-3.0.3/LICENSE-APACHE), [LICENSE-MIT](daachorse-3.0.3/LICENSE-MIT) |
| dary_heap 0.3.9 | Runtime | [source](https://crates.io/api/v1/crates/dary_heap/0.3.9/download) | [LICENSE-APACHE](dary_heap-0.3.9/LICENSE-APACHE), [LICENSE-MIT](dary_heap-0.3.9/LICENSE-MIT) |
| data-url 0.3.2 | Runtime | [source](https://crates.io/api/v1/crates/data-url/0.3.2/download) | [LICENSE-APACHE](data-url-0.3.2/LICENSE-APACHE), [LICENSE-MIT](data-url-0.3.2/LICENSE-MIT) |
| deranged 0.5.9 | Runtime | [source](https://crates.io/api/v1/crates/deranged/0.5.9/download) | [LICENSE-Apache](deranged-0.5.9/LICENSE-Apache), [LICENSE-MIT](deranged-0.5.9/LICENSE-MIT) |
| derive_builder 0.20.2 | Runtime | [source](https://crates.io/api/v1/crates/derive_builder/0.20.2/download) | [LICENSE-APACHE](derive_builder-0.20.2/LICENSE-APACHE), [LICENSE-MIT](derive_builder-0.20.2/LICENSE-MIT) |
| derive_builder_core 0.20.2 | Build/proc-macro only | [source](https://crates.io/api/v1/crates/derive_builder_core/0.20.2/download) | [LICENSE-APACHE](derive_builder_core-0.20.2/LICENSE-APACHE), [LICENSE-MIT](derive_builder_core-0.20.2/LICENSE-MIT) |
| derive_builder_macro 0.20.2 | Build/proc-macro only | [source](https://crates.io/api/v1/crates/derive_builder_macro/0.20.2/download) | [LICENSE-APACHE](derive_builder_macro-0.20.2/LICENSE-APACHE), [LICENSE-MIT](derive_builder_macro-0.20.2/LICENSE-MIT) |
| digest 0.10.7 | Runtime | [source](https://crates.io/api/v1/crates/digest/0.10.7/download) | [LICENSE-APACHE](digest-0.10.7/LICENSE-APACHE), [LICENSE-MIT](digest-0.10.7/LICENSE-MIT) |
| dirs 6.0.0 | Runtime | [source](https://crates.io/api/v1/crates/dirs/6.0.0/download) | [LICENSE-APACHE](dirs-6.0.0/LICENSE-APACHE), [LICENSE-MIT](dirs-6.0.0/LICENSE-MIT) |
| dirs-sys 0.5.0 | Runtime | [source](https://crates.io/api/v1/crates/dirs-sys/0.5.0/download) | [LICENSE-APACHE](dirs-sys-0.5.0/LICENSE-APACHE), [LICENSE-MIT](dirs-sys-0.5.0/LICENSE-MIT) |
| displaydoc 0.2.7 | Build/proc-macro only | [source](https://crates.io/api/v1/crates/displaydoc/0.2.7/download) | [LICENSE-APACHE](displaydoc-0.2.7/LICENSE-APACHE), [LICENSE-MIT](displaydoc-0.2.7/LICENSE-MIT) |
| document-features 0.2.12 | Build/proc-macro only | [source](https://crates.io/api/v1/crates/document-features/0.2.12/download) | [LICENSE-APACHE](document-features-0.2.12/LICENSE-APACHE), [LICENSE-MIT](document-features-0.2.12/LICENSE-MIT) |
| either 1.19.0 | Runtime | [source](https://crates.io/api/v1/crates/either/1.19.0/download) | [LICENSE-APACHE](either-1.19.0/LICENSE-APACHE), [LICENSE-MIT](either-1.19.0/LICENSE-MIT) |
| errno 0.3.14 | Runtime | [source](https://crates.io/api/v1/crates/errno/0.3.14/download) | [LICENSE-APACHE](errno-0.3.14/LICENSE-APACHE), [LICENSE-MIT](errno-0.3.14/LICENSE-MIT) |
| fdeflate 0.3.7 | Runtime | [source](https://crates.io/api/v1/crates/fdeflate/0.3.7/download) | [LICENSE-APACHE](fdeflate-0.3.7/LICENSE-APACHE), [LICENSE-MIT](fdeflate-0.3.7/LICENSE-MIT) |
| find-msvc-tools 0.1.14 | Build/proc-macro only | [source](https://crates.io/api/v1/crates/find-msvc-tools/0.1.14/download) | [LICENSE-APACHE](find-msvc-tools-0.1.14/LICENSE-APACHE), [LICENSE-MIT](find-msvc-tools-0.1.14/LICENSE-MIT) |
| fixedbitset 0.5.7 | Build/proc-macro only | [source](https://crates.io/api/v1/crates/fixedbitset/0.5.7/download) | [LICENSE-APACHE](fixedbitset-0.5.7/LICENSE-APACHE), [LICENSE-MIT](fixedbitset-0.5.7/LICENSE-MIT) |
| flate2 1.1.10 | Runtime | [source](https://crates.io/api/v1/crates/flate2/1.1.10/download) | [LICENSE-APACHE](flate2-1.1.10/LICENSE-APACHE), [LICENSE-MIT](flate2-1.1.10/LICENSE-MIT) |
| form_urlencoded 1.2.2 | Runtime | [source](https://crates.io/api/v1/crates/form_urlencoded/1.2.2/download) | [LICENSE-APACHE](form_urlencoded-1.2.2/LICENSE-APACHE), [LICENSE-MIT](form_urlencoded-1.2.2/LICENSE-MIT) |
| fraction 0.15.4 | Runtime | [source](https://crates.io/api/v1/crates/fraction/0.15.4/download) | [LICENSE-APACHE](fraction-0.15.4/LICENSE-APACHE), [LICENSE-MIT](fraction-0.15.4/LICENSE-MIT) |
| fraction 0.17.0 | Runtime | [source](https://crates.io/api/v1/crates/fraction/0.17.0/download) | [LICENSE-APACHE](fraction-0.17.0/LICENSE-APACHE), [LICENSE-MIT](fraction-0.17.0/LICENSE-MIT) |
| futures-channel 0.3.34 | Runtime | [source](https://crates.io/api/v1/crates/futures-channel/0.3.34/download) | [LICENSE-APACHE](futures-channel-0.3.34/LICENSE-APACHE), [LICENSE-MIT](futures-channel-0.3.34/LICENSE-MIT) |
| futures-core 0.3.34 | Runtime | [source](https://crates.io/api/v1/crates/futures-core/0.3.34/download) | [LICENSE-APACHE](futures-core-0.3.34/LICENSE-APACHE), [LICENSE-MIT](futures-core-0.3.34/LICENSE-MIT) |
| futures-macro 0.3.34 | Build/proc-macro only | [source](https://crates.io/api/v1/crates/futures-macro/0.3.34/download) | [LICENSE-APACHE](futures-macro-0.3.34/LICENSE-APACHE), [LICENSE-MIT](futures-macro-0.3.34/LICENSE-MIT) |
| futures-sink 0.3.34 | Runtime | [source](https://crates.io/api/v1/crates/futures-sink/0.3.34/download) | [LICENSE-APACHE](futures-sink-0.3.34/LICENSE-APACHE), [LICENSE-MIT](futures-sink-0.3.34/LICENSE-MIT) |
| futures-task 0.3.34 | Runtime | [source](https://crates.io/api/v1/crates/futures-task/0.3.34/download) | [LICENSE-APACHE](futures-task-0.3.34/LICENSE-APACHE), [LICENSE-MIT](futures-task-0.3.34/LICENSE-MIT) |
| futures-util 0.3.34 | Runtime | [source](https://crates.io/api/v1/crates/futures-util/0.3.34/download) | [LICENSE-APACHE](futures-util-0.3.34/LICENSE-APACHE), [LICENSE-MIT](futures-util-0.3.34/LICENSE-MIT) |
| getrandom 0.2.17 | Runtime | [source](https://crates.io/api/v1/crates/getrandom/0.2.17/download) | [LICENSE-APACHE](getrandom-0.2.17/LICENSE-APACHE), [LICENSE-MIT](getrandom-0.2.17/LICENSE-MIT) |
| getrandom 0.3.4 | Runtime | [source](https://crates.io/api/v1/crates/getrandom/0.3.4/download) | [LICENSE-APACHE](getrandom-0.3.4/LICENSE-APACHE), [LICENSE-MIT](getrandom-0.3.4/LICENSE-MIT) |
| getrandom 0.4.3 | Runtime | [source](https://crates.io/api/v1/crates/getrandom/0.4.3/download) | [LICENSE-APACHE](getrandom-0.4.3/LICENSE-APACHE), [LICENSE-MIT](getrandom-0.4.3/LICENSE-MIT) |
| hashbrown 0.15.5 | Runtime | [source](https://crates.io/api/v1/crates/hashbrown/0.15.5/download) | [LICENSE-APACHE](hashbrown-0.15.5/LICENSE-APACHE), [LICENSE-MIT](hashbrown-0.15.5/LICENSE-MIT) |
| hashbrown 0.17.1 | Runtime | [source](https://crates.io/api/v1/crates/hashbrown/0.17.1/download) | [LICENSE-APACHE](hashbrown-0.17.1/LICENSE-APACHE), [LICENSE-MIT](hashbrown-0.17.1/LICENSE-MIT) |
| heck 0.5.0 | Build/proc-macro only | [source](https://crates.io/api/v1/crates/heck/0.5.0/download) | [LICENSE-APACHE](heck-0.5.0/LICENSE-APACHE), [LICENSE-MIT](heck-0.5.0/LICENSE-MIT) |
| http 1.5.0 | Runtime | [source](https://crates.io/api/v1/crates/http/1.5.0/download) | [LICENSE-APACHE](http-1.5.0/LICENSE-APACHE), [LICENSE-MIT](http-1.5.0/LICENSE-MIT) |
| httparse 1.10.1 | Runtime | [source](https://crates.io/api/v1/crates/httparse/1.10.1/download) | [LICENSE-APACHE](httparse-1.10.1/LICENSE-APACHE), [LICENSE-MIT](httparse-1.10.1/LICENSE-MIT) |
| httpdate 1.0.3 | Runtime | [source](https://crates.io/api/v1/crates/httpdate/1.0.3/download) | [LICENSE-APACHE](httpdate-1.0.3/LICENSE-APACHE), [LICENSE-MIT](httpdate-1.0.3/LICENSE-MIT) |
| hyper-timeout 0.5.2 | Runtime | [source](https://crates.io/api/v1/crates/hyper-timeout/0.5.2/download) | [LICENSE-APACHE](hyper-timeout-0.5.2/LICENSE-APACHE), [LICENSE-MIT](hyper-timeout-0.5.2/LICENSE-MIT) |
| idna 1.1.0 | Runtime | [source](https://crates.io/api/v1/crates/idna/1.1.0/download) | [LICENSE-APACHE](idna-1.1.0/LICENSE-APACHE), [LICENSE-MIT](idna-1.1.0/LICENSE-MIT) |
| image-webp 0.2.4 | Runtime | [source](https://crates.io/api/v1/crates/image-webp/0.2.4/download) | [LICENSE-APACHE](image-webp-0.2.4/LICENSE-APACHE), [LICENSE-MIT](image-webp-0.2.4/LICENSE-MIT) |
| indoc 2.0.8 | Build/proc-macro only | [source](https://crates.io/api/v1/crates/indoc/2.0.8/download) | [LICENSE-APACHE](indoc-2.0.8/LICENSE-APACHE), [LICENSE-MIT](indoc-2.0.8/LICENSE-MIT) |
| io-uring 0.7.15 | Runtime | [source](https://crates.io/api/v1/crates/io-uring/0.7.15/download) | [LICENSE-APACHE](io-uring-0.7.15/LICENSE-APACHE), [LICENSE-MIT](io-uring-0.7.15/LICENSE-MIT) |
| is_terminal_polyfill 1.70.2 | Runtime | [source](https://crates.io/api/v1/crates/is_terminal_polyfill/1.70.2/download) | [LICENSE-APACHE](is_terminal_polyfill-1.70.2/LICENSE-APACHE), [LICENSE-MIT](is_terminal_polyfill-1.70.2/LICENSE-MIT) |
| itertools 0.13.0 | Runtime | [source](https://crates.io/api/v1/crates/itertools/0.13.0/download) | [LICENSE-APACHE](itertools-0.13.0/LICENSE-APACHE), [LICENSE-MIT](itertools-0.13.0/LICENSE-MIT) |
| itertools 0.14.0 | Runtime | [source](https://crates.io/api/v1/crates/itertools/0.14.0/download) | [LICENSE-APACHE](itertools-0.14.0/LICENSE-APACHE), [LICENSE-MIT](itertools-0.14.0/LICENSE-MIT) |
| itoa 1.0.18 | Runtime | [source](https://crates.io/api/v1/crates/itoa/1.0.18/download) | [LICENSE-APACHE](itoa-1.0.18/LICENSE-APACHE), [LICENSE-MIT](itoa-1.0.18/LICENSE-MIT) |
| jobserver 0.1.35 | Build/proc-macro only | [source](https://crates.io/api/v1/crates/jobserver/0.1.35/download) | [LICENSE-APACHE](jobserver-0.1.35/LICENSE-APACHE), [LICENSE-MIT](jobserver-0.1.35/LICENSE-MIT) |
| lazy_static 1.5.1 | Runtime | [source](https://crates.io/api/v1/crates/lazy_static/1.5.1/download) | [LICENSE-APACHE](lazy_static-1.5.1/LICENSE-APACHE), [LICENSE-MIT](lazy_static-1.5.1/LICENSE-MIT) |
| libc 0.2.189 | Runtime | [source](https://crates.io/api/v1/crates/libc/0.2.189/download) | [LICENSE-APACHE](libc-0.2.189/LICENSE-APACHE), [LICENSE-MIT](libc-0.2.189/LICENSE-MIT) |
| litrs 1.0.1 | Build/proc-macro only | [source](https://crates.io/api/v1/crates/litrs/1.0.1/download) | [LICENSE-APACHE](litrs-1.0.1/LICENSE-APACHE), [LICENSE-MIT](litrs-1.0.1/LICENSE-MIT) |
| lock_api 0.4.14 | Runtime | [source](https://crates.io/api/v1/crates/lock_api/0.4.14/download) | [LICENSE-APACHE](lock_api-0.4.14/LICENSE-APACHE), [LICENSE-MIT](lock_api-0.4.14/LICENSE-MIT) |
| log 0.4.34 | Runtime | [source](https://crates.io/api/v1/crates/log/0.4.34/download) | [LICENSE-APACHE](log-0.4.34/LICENSE-APACHE), [LICENSE-MIT](log-0.4.34/LICENSE-MIT) |
| mime 0.3.17 | Runtime | [source](https://crates.io/api/v1/crates/mime/0.3.17/download) | [LICENSE-APACHE](mime-0.3.17/LICENSE-APACHE), [LICENSE-MIT](mime-0.3.17/LICENSE-MIT) |
| monostate 0.1.18 | Runtime | [source](https://crates.io/api/v1/crates/monostate/0.1.18/download) | [LICENSE-APACHE](monostate-0.1.18/LICENSE-APACHE), [LICENSE-MIT](monostate-0.1.18/LICENSE-MIT) |
| monostate-impl 0.1.18 | Build/proc-macro only | [source](https://crates.io/api/v1/crates/monostate-impl/0.1.18/download) | [LICENSE-APACHE](monostate-impl-0.1.18/LICENSE-APACHE), [LICENSE-MIT](monostate-impl-0.1.18/LICENSE-MIT) |
| multimap 0.10.1 | Build/proc-macro only | [source](https://crates.io/api/v1/crates/multimap/0.10.1/download) | [LICENSE-APACHE](multimap-0.10.1/LICENSE-APACHE), [LICENSE-MIT](multimap-0.10.1/LICENSE-MIT) |
| num 0.4.3 | Runtime | [source](https://crates.io/api/v1/crates/num/0.4.3/download) | [LICENSE-APACHE](num-0.4.3/LICENSE-APACHE), [LICENSE-MIT](num-0.4.3/LICENSE-MIT) |
| num-bigint 0.4.8 | Runtime | [source](https://crates.io/api/v1/crates/num-bigint/0.4.8/download) | [LICENSE-APACHE](num-bigint-0.4.8/LICENSE-APACHE), [LICENSE-MIT](num-bigint-0.4.8/LICENSE-MIT) |
| num-complex 0.4.6 | Runtime | [source](https://crates.io/api/v1/crates/num-complex/0.4.6/download) | [LICENSE-APACHE](num-complex-0.4.6/LICENSE-APACHE), [LICENSE-MIT](num-complex-0.4.6/LICENSE-MIT) |
| num-conv 0.1.0 | Runtime | [source](https://crates.io/api/v1/crates/num-conv/0.1.0/download) | [LICENSE-Apache](num-conv-0.1.0/LICENSE-Apache), [LICENSE-MIT](num-conv-0.1.0/LICENSE-MIT) |
| num-integer 0.1.47 | Runtime | [source](https://crates.io/api/v1/crates/num-integer/0.1.47/download) | [LICENSE-APACHE](num-integer-0.1.47/LICENSE-APACHE), [LICENSE-MIT](num-integer-0.1.47/LICENSE-MIT) |
| num-iter 0.1.46 | Runtime | [source](https://crates.io/api/v1/crates/num-iter/0.1.46/download) | [LICENSE-APACHE](num-iter-0.1.46/LICENSE-APACHE), [LICENSE-MIT](num-iter-0.1.46/LICENSE-MIT) |
| num-rational 0.4.2 | Runtime | [source](https://crates.io/api/v1/crates/num-rational/0.4.2/download) | [LICENSE-APACHE](num-rational-0.4.2/LICENSE-APACHE), [LICENSE-MIT](num-rational-0.4.2/LICENSE-MIT) |
| num-traits 0.2.19 | Runtime | [source](https://crates.io/api/v1/crates/num-traits/0.2.19/download) | [LICENSE-APACHE](num-traits-0.2.19/LICENSE-APACHE), [LICENSE-MIT](num-traits-0.2.19/LICENSE-MIT) |
| once_cell 1.21.4 | Runtime | [source](https://crates.io/api/v1/crates/once_cell/1.21.4/download) | [LICENSE-APACHE](once_cell-1.21.4/LICENSE-APACHE), [LICENSE-MIT](once_cell-1.21.4/LICENSE-MIT) |
| once_cell_polyfill 1.70.2 | Runtime | [source](https://crates.io/api/v1/crates/once_cell_polyfill/1.70.2/download) | [LICENSE-APACHE](once_cell_polyfill-1.70.2/LICENSE-APACHE), [LICENSE-MIT](once_cell_polyfill-1.70.2/LICENSE-MIT) |
| parking_lot 0.12.5 | Runtime | [source](https://crates.io/api/v1/crates/parking_lot/0.12.5/download) | [LICENSE-APACHE](parking_lot-0.12.5/LICENSE-APACHE), [LICENSE-MIT](parking_lot-0.12.5/LICENSE-MIT) |
| parking_lot_core 0.9.12 | Runtime | [source](https://crates.io/api/v1/crates/parking_lot_core/0.9.12/download) | [LICENSE-APACHE](parking_lot_core-0.9.12/LICENSE-APACHE), [LICENSE-MIT](parking_lot_core-0.9.12/LICENSE-MIT) |
| paste 1.0.15 | Build/proc-macro only | [source](https://crates.io/api/v1/crates/paste/1.0.15/download) | [LICENSE-APACHE](paste-1.0.15/LICENSE-APACHE), [LICENSE-MIT](paste-1.0.15/LICENSE-MIT) |
| pastey 0.2.3 | Build/proc-macro only | [source](https://crates.io/api/v1/crates/pastey/0.2.3/download) | [LICENSE-APACHE](pastey-0.2.3/LICENSE-APACHE), [LICENSE-MIT](pastey-0.2.3/LICENSE-MIT) |
| percent-encoding 2.3.2 | Runtime | [source](https://crates.io/api/v1/crates/percent-encoding/2.3.2/download) | [LICENSE-APACHE](percent-encoding-2.3.2/LICENSE-APACHE), [LICENSE-MIT](percent-encoding-2.3.2/LICENSE-MIT) |
| petgraph 0.8.3 | Build/proc-macro only | [source](https://crates.io/api/v1/crates/petgraph/0.8.3/download) | [LICENSE-APACHE](petgraph-0.8.3/LICENSE-APACHE), [LICENSE-MIT](petgraph-0.8.3/LICENSE-MIT), [LICENSE.md](petgraph-0.8.3/assets/images/LICENSE.md) |
| pkg-config 0.3.34 | Build/proc-macro only | [source](https://crates.io/api/v1/crates/pkg-config/0.3.34/download) | [LICENSE-APACHE](pkg-config-0.3.34/LICENSE-APACHE), [LICENSE-MIT](pkg-config-0.3.34/LICENSE-MIT) |
| png 0.18.1 | Runtime | [source](https://crates.io/api/v1/crates/png/0.18.1/download) | [LICENSE-APACHE](png-0.18.1/LICENSE-APACHE), [LICENSE-MIT](png-0.18.1/LICENSE-MIT) |
| powerfmt 0.2.1 | Runtime | [source](https://crates.io/api/v1/crates/powerfmt/0.2.1/download) | [LICENSE-Apache](powerfmt-0.2.1/LICENSE-Apache), [LICENSE-MIT](powerfmt-0.2.1/LICENSE-MIT) |
| ppv-lite86 0.2.21 | Runtime | [source](https://crates.io/api/v1/crates/ppv-lite86/0.2.21/download) | [LICENSE-APACHE](ppv-lite86-0.2.21/LICENSE-APACHE), [LICENSE-MIT](ppv-lite86-0.2.21/LICENSE-MIT) |
| prettyplease 0.2.37 | Build/proc-macro only | [source](https://crates.io/api/v1/crates/prettyplease/0.2.37/download) | [LICENSE-APACHE](prettyplease-0.2.37/LICENSE-APACHE), [LICENSE-MIT](prettyplease-0.2.37/LICENSE-MIT) |
| proc-macro2 1.0.107 | Build/proc-macro only | [source](https://crates.io/api/v1/crates/proc-macro2/1.0.107/download) | [LICENSE-APACHE](proc-macro2-1.0.107/LICENSE-APACHE), [LICENSE-MIT](proc-macro2-1.0.107/LICENSE-MIT) |
| quote 1.0.47 | Build/proc-macro only | [source](https://crates.io/api/v1/crates/quote/1.0.47/download) | [LICENSE-APACHE](quote-1.0.47/LICENSE-APACHE), [LICENSE-MIT](quote-1.0.47/LICENSE-MIT) |
| rand 0.9.5 | Runtime | [source](https://crates.io/api/v1/crates/rand/0.9.5/download) | [COPYRIGHT](rand-0.9.5/COPYRIGHT), [LICENSE-APACHE](rand-0.9.5/LICENSE-APACHE), [LICENSE-MIT](rand-0.9.5/LICENSE-MIT) |
| rand_chacha 0.9.0 | Runtime | [source](https://crates.io/api/v1/crates/rand_chacha/0.9.0/download) | [COPYRIGHT](rand_chacha-0.9.0/COPYRIGHT), [LICENSE-APACHE](rand_chacha-0.9.0/LICENSE-APACHE), [LICENSE-MIT](rand_chacha-0.9.0/LICENSE-MIT) |
| rand_core 0.9.5 | Runtime | [source](https://crates.io/api/v1/crates/rand_core/0.9.5/download) | [COPYRIGHT](rand_core-0.9.5/COPYRIGHT), [LICENSE-APACHE](rand_core-0.9.5/LICENSE-APACHE), [LICENSE-MIT](rand_core-0.9.5/LICENSE-MIT) |
| rayon 1.12.0 | Runtime | [source](https://crates.io/api/v1/crates/rayon/1.12.0/download) | [LICENSE-APACHE](rayon-1.12.0/LICENSE-APACHE), [LICENSE-MIT](rayon-1.12.0/LICENSE-MIT) |
| rayon-core 1.13.0 | Runtime | [source](https://crates.io/api/v1/crates/rayon-core/1.13.0/download) | [LICENSE-APACHE](rayon-core-1.13.0/LICENSE-APACHE), [LICENSE-MIT](rayon-core-1.13.0/LICENSE-MIT) |
| ref-cast 1.0.27 | Runtime | [source](https://crates.io/api/v1/crates/ref-cast/1.0.27/download) | [LICENSE-APACHE](ref-cast-1.0.27/LICENSE-APACHE), [LICENSE-MIT](ref-cast-1.0.27/LICENSE-MIT) |
| ref-cast-impl 1.0.27 | Build/proc-macro only | [source](https://crates.io/api/v1/crates/ref-cast-impl/1.0.27/download) | [LICENSE-APACHE](ref-cast-impl-1.0.27/LICENSE-APACHE), [LICENSE-MIT](ref-cast-impl-1.0.27/LICENSE-MIT) |
| regex 1.13.1 | Runtime | [source](https://crates.io/api/v1/crates/regex/1.13.1/download) | [LICENSE-APACHE](regex-1.13.1/LICENSE-APACHE), [LICENSE-MIT](regex-1.13.1/LICENSE-MIT) |
| regex-automata 0.4.18 | Runtime | [source](https://crates.io/api/v1/crates/regex-automata/0.4.18/download) | [LICENSE-APACHE](regex-automata-0.4.18/LICENSE-APACHE), [LICENSE-MIT](regex-automata-0.4.18/LICENSE-MIT) |
| regex-syntax 0.8.11 | Runtime | [source](https://crates.io/api/v1/crates/regex-syntax/0.8.11/download) | [LICENSE-APACHE](regex-syntax-0.8.11/LICENSE-APACHE), [LICENSE-MIT](regex-syntax-0.8.11/LICENSE-MIT), [LICENSE-UNICODE](regex-syntax-0.8.11/src/unicode_tables/LICENSE-UNICODE) |
| rustls-pki-types 1.15.1 | Runtime | [source](https://crates.io/api/v1/crates/rustls-pki-types/1.15.1/download) | [LICENSE-APACHE](rustls-pki-types-1.15.1/LICENSE-APACHE), [LICENSE-MIT](rustls-pki-types-1.15.1/LICENSE-MIT) |
| rustversion 1.0.23 | Build/proc-macro only | [source](https://crates.io/api/v1/crates/rustversion/1.0.23/download) | [LICENSE-APACHE](rustversion-1.0.23/LICENSE-APACHE), [LICENSE-MIT](rustversion-1.0.23/LICENSE-MIT) |
| scopeguard 1.2.0 | Runtime | [source](https://crates.io/api/v1/crates/scopeguard/1.2.0/download) | [LICENSE-APACHE](scopeguard-1.2.0/LICENSE-APACHE), [LICENSE-MIT](scopeguard-1.2.0/LICENSE-MIT) |
| serde 1.0.229 | Runtime | [source](https://crates.io/api/v1/crates/serde/1.0.229/download) | [LICENSE-APACHE](serde-1.0.229/LICENSE-APACHE), [LICENSE-MIT](serde-1.0.229/LICENSE-MIT) |
| serde_core 1.0.229 | Runtime | [source](https://crates.io/api/v1/crates/serde_core/1.0.229/download) | [LICENSE-APACHE](serde_core-1.0.229/LICENSE-APACHE), [LICENSE-MIT](serde_core-1.0.229/LICENSE-MIT) |
| serde_derive 1.0.229 | Build/proc-macro only | [source](https://crates.io/api/v1/crates/serde_derive/1.0.229/download) | [LICENSE-APACHE](serde_derive-1.0.229/LICENSE-APACHE), [LICENSE-MIT](serde_derive-1.0.229/LICENSE-MIT) |
| serde_json 1.0.151 | Runtime | [source](https://crates.io/api/v1/crates/serde_json/1.0.151/download) | [LICENSE-APACHE](serde_json-1.0.151/LICENSE-APACHE), [LICENSE-MIT](serde_json-1.0.151/LICENSE-MIT) |
| serde_path_to_error 0.1.20 | Runtime | [source](https://crates.io/api/v1/crates/serde_path_to_error/0.1.20/download) | [LICENSE-APACHE](serde_path_to_error-0.1.20/LICENSE-APACHE), [LICENSE-MIT](serde_path_to_error-0.1.20/LICENSE-MIT) |
| serde_spanned 0.6.9 | Runtime | [source](https://crates.io/api/v1/crates/serde_spanned/0.6.9/download) | [LICENSE-APACHE](serde_spanned-0.6.9/LICENSE-APACHE), [LICENSE-MIT](serde_spanned-0.6.9/LICENSE-MIT) |
| sha2 0.10.9 | Runtime | [source](https://crates.io/api/v1/crates/sha2/0.10.9/download) | [LICENSE-APACHE](sha2-0.10.9/LICENSE-APACHE), [LICENSE-MIT](sha2-0.10.9/LICENSE-MIT) |
| shlex 2.0.1 | Build/proc-macro only | [source](https://crates.io/api/v1/crates/shlex/2.0.1/download) | [LICENSE-APACHE](shlex-2.0.1/LICENSE-APACHE), [LICENSE-MIT](shlex-2.0.1/LICENSE-MIT) |
| signal-hook-mio 0.2.5 | Runtime | [source](https://crates.io/api/v1/crates/signal-hook-mio/0.2.5/download) | [LICENSE-APACHE](signal-hook-mio-0.2.5/LICENSE-APACHE), [LICENSE-MIT](signal-hook-mio-0.2.5/LICENSE-MIT) |
| signal-hook-registry 1.4.8 | Runtime | [source](https://crates.io/api/v1/crates/signal-hook-registry/1.4.8/download) | [LICENSE-APACHE](signal-hook-registry-1.4.8/LICENSE-APACHE), [LICENSE-MIT](signal-hook-registry-1.4.8/LICENSE-MIT) |
| simdutf8 0.1.5 | Runtime | [source](https://crates.io/api/v1/crates/simdutf8/0.1.5/download) | [LICENSE-Apache](simdutf8-0.1.5/LICENSE-Apache), [LICENSE-MIT](simdutf8-0.1.5/LICENSE-MIT) |
| smallvec 1.16.2 | Runtime | [source](https://crates.io/api/v1/crates/smallvec/1.16.2/download) | [LICENSE-APACHE](smallvec-1.16.2/LICENSE-APACHE), [LICENSE-MIT](smallvec-1.16.2/LICENSE-MIT) |
| socket2 0.6.5 | Runtime | [source](https://crates.io/api/v1/crates/socket2/0.6.5/download) | [LICENSE-APACHE](socket2-0.6.5/LICENSE-APACHE), [LICENSE-MIT](socket2-0.6.5/LICENSE-MIT) |
| stable_deref_trait 1.2.1 | Runtime | [source](https://crates.io/api/v1/crates/stable_deref_trait/1.2.1/download) | [LICENSE-APACHE](stable_deref_trait-1.2.1/LICENSE-APACHE), [LICENSE-MIT](stable_deref_trait-1.2.1/LICENSE-MIT) |
| static_assertions 1.1.0 | Runtime | [source](https://crates.io/api/v1/crates/static_assertions/1.1.0/download) | [LICENSE-APACHE](static_assertions-1.1.0/LICENSE-APACHE), [LICENSE-MIT](static_assertions-1.1.0/LICENSE-MIT) |
| syn 2.0.119 | Build/proc-macro only | [source](https://crates.io/api/v1/crates/syn/2.0.119/download) | [LICENSE-APACHE](syn-2.0.119/LICENSE-APACHE), [LICENSE-MIT](syn-2.0.119/LICENSE-MIT) |
| syn 3.0.6 | Build/proc-macro only | [source](https://crates.io/api/v1/crates/syn/3.0.6/download) | [LICENSE-APACHE](syn-3.0.6/LICENSE-APACHE), [LICENSE-MIT](syn-3.0.6/LICENSE-MIT) |
| tempfile 3.27.0 | Runtime | [source](https://crates.io/api/v1/crates/tempfile/3.27.0/download) | [LICENSE-APACHE](tempfile-3.27.0/LICENSE-APACHE), [LICENSE-MIT](tempfile-3.27.0/LICENSE-MIT) |
| thiserror 1.0.69 | Runtime | [source](https://crates.io/api/v1/crates/thiserror/1.0.69/download) | [LICENSE-APACHE](thiserror-1.0.69/LICENSE-APACHE), [LICENSE-MIT](thiserror-1.0.69/LICENSE-MIT) |
| thiserror 2.0.21 | Runtime | [source](https://crates.io/api/v1/crates/thiserror/2.0.21/download) | [LICENSE-APACHE](thiserror-2.0.21/LICENSE-APACHE), [LICENSE-MIT](thiserror-2.0.21/LICENSE-MIT) |
| thiserror-impl 1.0.69 | Build/proc-macro only | [source](https://crates.io/api/v1/crates/thiserror-impl/1.0.69/download) | [LICENSE-APACHE](thiserror-impl-1.0.69/LICENSE-APACHE), [LICENSE-MIT](thiserror-impl-1.0.69/LICENSE-MIT) |
| thiserror-impl 2.0.21 | Build/proc-macro only | [source](https://crates.io/api/v1/crates/thiserror-impl/2.0.21/download) | [LICENSE-APACHE](thiserror-impl-2.0.21/LICENSE-APACHE), [LICENSE-MIT](thiserror-impl-2.0.21/LICENSE-MIT) |
| time 0.3.45 | Runtime | [source](https://crates.io/api/v1/crates/time/0.3.45/download) | [LICENSE-Apache](time-0.3.45/LICENSE-Apache), [LICENSE-MIT](time-0.3.45/LICENSE-MIT) |
| time-core 0.1.7 | Runtime | [source](https://crates.io/api/v1/crates/time-core/0.1.7/download) | [LICENSE-Apache](time-core-0.1.7/LICENSE-Apache), [LICENSE-MIT](time-core-0.1.7/LICENSE-MIT) |
| time-macros 0.2.25 | Build/proc-macro only | [source](https://crates.io/api/v1/crates/time-macros/0.2.25/download) | [LICENSE-Apache](time-macros-0.2.25/LICENSE-Apache), [LICENSE-MIT](time-macros-0.2.25/LICENSE-MIT) |
| tokio-rustls 0.26.6 | Runtime | [source](https://crates.io/api/v1/crates/tokio-rustls/0.26.6/download) | [LICENSE-APACHE](tokio-rustls-0.26.6/LICENSE-APACHE), [LICENSE-MIT](tokio-rustls-0.26.6/LICENSE-MIT) |
| toml 0.8.23 | Runtime | [source](https://crates.io/api/v1/crates/toml/0.8.23/download) | [LICENSE-APACHE](toml-0.8.23/LICENSE-APACHE), [LICENSE-MIT](toml-0.8.23/LICENSE-MIT) |
| toml_datetime 0.6.11 | Runtime | [source](https://crates.io/api/v1/crates/toml_datetime/0.6.11/download) | [LICENSE-APACHE](toml_datetime-0.6.11/LICENSE-APACHE), [LICENSE-MIT](toml_datetime-0.6.11/LICENSE-MIT) |
| toml_edit 0.22.27 | Runtime | [source](https://crates.io/api/v1/crates/toml_edit/0.22.27/download) | [LICENSE-APACHE](toml_edit-0.22.27/LICENSE-APACHE), [LICENSE-MIT](toml_edit-0.22.27/LICENSE-MIT) |
| toml_write 0.1.2 | Runtime | [source](https://crates.io/api/v1/crates/toml_write/0.1.2/download) | [LICENSE-APACHE](toml_write-0.1.2/LICENSE-APACHE), [LICENSE-MIT](toml_write-0.1.2/LICENSE-MIT) |
| typenum 1.20.1 | Runtime | [source](https://crates.io/api/v1/crates/typenum/1.20.1/download) | [LICENSE](typenum-1.20.1/LICENSE), [LICENSE-APACHE](typenum-1.20.1/LICENSE-APACHE), [LICENSE-MIT](typenum-1.20.1/LICENSE-MIT) |
| unicase 2.10.0 | Build/proc-macro only | [source](https://crates.io/api/v1/crates/unicase/2.10.0/download) | [LICENSE-APACHE](unicase-2.10.0/LICENSE-APACHE), [LICENSE-MIT](unicase-2.10.0/LICENSE-MIT) |
| unicode-segmentation 1.13.3 | Runtime | [source](https://crates.io/api/v1/crates/unicode-segmentation/1.13.3/download) | [COPYRIGHT](unicode-segmentation-1.13.3/COPYRIGHT), [LICENSE-APACHE](unicode-segmentation-1.13.3/LICENSE-APACHE), [LICENSE-MIT](unicode-segmentation-1.13.3/LICENSE-MIT) |
| unicode-truncate 1.1.0 | Runtime | [source](https://crates.io/api/v1/crates/unicode-truncate/1.1.0/download) | [COPYRIGHT](unicode-truncate-1.1.0/COPYRIGHT), [LICENSE-APACHE](unicode-truncate-1.1.0/LICENSE-APACHE), [LICENSE-MIT](unicode-truncate-1.1.0/LICENSE-MIT) |
| unicode-width 0.1.14 | Runtime | [source](https://crates.io/api/v1/crates/unicode-width/0.1.14/download) | [COPYRIGHT](unicode-width-0.1.14/COPYRIGHT), [LICENSE-APACHE](unicode-width-0.1.14/LICENSE-APACHE), [LICENSE-MIT](unicode-width-0.1.14/LICENSE-MIT) |
| unicode-width 0.2.0 | Runtime | [source](https://crates.io/api/v1/crates/unicode-width/0.2.0/download) | [COPYRIGHT](unicode-width-0.2.0/COPYRIGHT), [LICENSE-APACHE](unicode-width-0.2.0/LICENSE-APACHE), [LICENSE-MIT](unicode-width-0.2.0/LICENSE-MIT) |
| unicode_categories 0.1.1 | Runtime | [source](https://crates.io/api/v1/crates/unicode_categories/0.1.1/download) | [LICENSE-APACHE](unicode_categories-0.1.1/LICENSE-APACHE), [LICENSE-MIT](unicode_categories-0.1.1/LICENSE-MIT) |
| ureq 2.12.1 | Runtime | [source](https://crates.io/api/v1/crates/ureq/2.12.1/download) | [LICENSE-APACHE](ureq-2.12.1/LICENSE-APACHE), [LICENSE-MIT](ureq-2.12.1/LICENSE-MIT), [LICENSE-APACHE](ureq-2.12.1/src/chunked/LICENSE-APACHE), [LICENSE-MIT](ureq-2.12.1/src/chunked/LICENSE-MIT) |
| ureq 3.4.2 | Runtime | [source](https://crates.io/api/v1/crates/ureq/3.4.2/download) | [LICENSE-APACHE](ureq-3.4.2/LICENSE-APACHE), [LICENSE-MIT](ureq-3.4.2/LICENSE-MIT) |
| ureq-proto 0.6.4 | Runtime | [source](https://crates.io/api/v1/crates/ureq-proto/0.6.4/download) | [LICENSE-APACHE.txt](ureq-proto-0.6.4/LICENSE-APACHE.txt), [LICENSE-MIT.txt](ureq-proto-0.6.4/LICENSE-MIT.txt) |
| url 2.5.8 | Runtime | [source](https://crates.io/api/v1/crates/url/2.5.8/download) | [LICENSE-APACHE](url-2.5.8/LICENSE-APACHE), [LICENSE-MIT](url-2.5.8/LICENSE-MIT) |
| utf8-zero 0.8.1 | Runtime | [source](https://crates.io/api/v1/crates/utf8-zero/0.8.1/download) | [LICENSE-APACHE](utf8-zero-0.8.1/LICENSE-APACHE), [LICENSE-MIT](utf8-zero-0.8.1/LICENSE-MIT) |
| windows-link 0.2.1 | Runtime | [source](https://crates.io/api/v1/crates/windows-link/0.2.1/download) | [license-apache-2.0](windows-link-0.2.1/license-apache-2.0), [license-mit](windows-link-0.2.1/license-mit) |
| windows-sys 0.59.0 | Runtime | [source](https://crates.io/api/v1/crates/windows-sys/0.59.0/download) | [license-apache-2.0](windows-sys-0.59.0/license-apache-2.0), [license-mit](windows-sys-0.59.0/license-mit) |
| windows-sys 0.60.2 | Runtime | [source](https://crates.io/api/v1/crates/windows-sys/0.60.2/download) | [license-apache-2.0](windows-sys-0.60.2/license-apache-2.0), [license-mit](windows-sys-0.60.2/license-mit) |
| windows-sys 0.61.2 | Runtime | [source](https://crates.io/api/v1/crates/windows-sys/0.61.2/download) | [license-apache-2.0](windows-sys-0.61.2/license-apache-2.0), [license-mit](windows-sys-0.61.2/license-mit) |
| windows-targets 0.52.6 | Runtime | [source](https://crates.io/api/v1/crates/windows-targets/0.52.6/download) | [license-apache-2.0](windows-targets-0.52.6/license-apache-2.0), [license-mit](windows-targets-0.52.6/license-mit) |
| windows-targets 0.53.5 | Runtime | [source](https://crates.io/api/v1/crates/windows-targets/0.53.5/download) | [license-apache-2.0](windows-targets-0.53.5/license-apache-2.0), [license-mit](windows-targets-0.53.5/license-mit) |
| windows_x86_64_msvc 0.52.6 | Runtime | [source](https://crates.io/api/v1/crates/windows_x86_64_msvc/0.52.6/download) | [license-apache-2.0](windows_x86_64_msvc-0.52.6/license-apache-2.0), [license-mit](windows_x86_64_msvc-0.52.6/license-mit) |
| windows_x86_64_msvc 0.53.1 | Runtime | [source](https://crates.io/api/v1/crates/windows_x86_64_msvc/0.53.1/download) | [license-apache-2.0](windows_x86_64_msvc-0.53.1/license-apache-2.0), [license-mit](windows_x86_64_msvc-0.53.1/license-mit) |

## MIT OR Zlib OR Apache-2.0

| Package | Role | Source | License texts |
| --- | --- | --- | --- |
| miniz_oxide 0.8.9 | Runtime | [source](https://crates.io/api/v1/crates/miniz_oxide/0.8.9/download) | [LICENSE](miniz_oxide-0.8.9/LICENSE), [LICENSE-APACHE.md](miniz_oxide-0.8.9/LICENSE-APACHE.md), [LICENSE-MIT.md](miniz_oxide-0.8.9/LICENSE-MIT.md), [LICENSE-ZLIB.md](miniz_oxide-0.8.9/LICENSE-ZLIB.md) |
| miniz_oxide 0.9.1 | Runtime | [source](https://crates.io/api/v1/crates/miniz_oxide/0.9.1/download) | [LICENSE](miniz_oxide-0.9.1/LICENSE), [LICENSE-APACHE.md](miniz_oxide-0.9.1/LICENSE-APACHE.md), [LICENSE-MIT.md](miniz_oxide-0.9.1/LICENSE-MIT.md), [LICENSE-ZLIB.md](miniz_oxide-0.9.1/LICENSE-ZLIB.md) |

## MIT-0

| Package | Role | Source | License texts |
| --- | --- | --- | --- |
| borrow-or-share 0.2.4 | Runtime | [source](https://crates.io/api/v1/crates/borrow-or-share/0.2.4/download) | [LICENSE](borrow-or-share-0.2.4/LICENSE) |

## MIT/Apache-2.0

| Package | Role | Source | License texts |
| --- | --- | --- | --- |
| base64 0.13.1 | Runtime | [source](https://crates.io/api/v1/crates/base64/0.13.1/download) | [LICENSE-APACHE](base64-0.13.1/LICENSE-APACHE), [LICENSE-MIT](base64-0.13.1/LICENSE-MIT) |
| core_detect 1.0.0 | Runtime | [source](https://crates.io/api/v1/crates/core_detect/1.0.0/download) | [LICENSE-APACHE](core_detect-1.0.0/LICENSE-APACHE), [LICENSE-MIT](core_detect-1.0.0/LICENSE-MIT) |
| gcd 2.3.0 | Runtime | [source](https://crates.io/api/v1/crates/gcd/2.3.0/download) | [LICENSE-APACHE](gcd-2.3.0/LICENSE-APACHE), [LICENSE-MIT](gcd-2.3.0/LICENSE-MIT) |
| ident_case 1.0.1 | Build/proc-macro only | [source](https://crates.io/api/v1/crates/ident_case/1.0.1/download) | [LICENSE](ident_case-1.0.1/LICENSE) |
| minimal-lexical 0.2.1 | Runtime | [source](https://crates.io/api/v1/crates/minimal-lexical/0.2.1/download) | [LICENSE-APACHE](minimal-lexical-0.2.1/LICENSE-APACHE), [LICENSE-MIT](minimal-lexical-0.2.1/LICENSE-MIT), [LICENSE.md](minimal-lexical-0.2.1/LICENSE.md) |
| num-cmp 0.1.0 | Runtime | [source](https://crates.io/api/v1/crates/num-cmp/0.1.0/download) | [LICENSE.txt](num-cmp-0.1.0/LICENSE.txt) |
| quick-error 2.0.1 | Runtime | [source](https://crates.io/api/v1/crates/quick-error/2.0.1/download) | [LICENSE-APACHE](quick-error-2.0.1/LICENSE-APACHE), [LICENSE-MIT](quick-error-2.0.1/LICENSE-MIT) |
| socks 0.3.4 | Runtime | [source](https://crates.io/api/v1/crates/socks/0.3.4/download) | [LICENSE-APACHE](socks-0.3.4/LICENSE-APACHE), [LICENSE-MIT](socks-0.3.4/LICENSE-MIT) |
| unicode-normalization-alignments 0.1.12 | Runtime | [source](https://crates.io/api/v1/crates/unicode-normalization-alignments/0.1.12/download) | [COPYRIGHT](unicode-normalization-alignments-0.1.12/COPYRIGHT), [LICENSE-APACHE](unicode-normalization-alignments-0.1.12/LICENSE-APACHE), [LICENSE-MIT](unicode-normalization-alignments-0.1.12/LICENSE-MIT) |
| version_check 0.9.5 | Build/proc-macro only | [source](https://crates.io/api/v1/crates/version_check/0.9.5/download) | [LICENSE-APACHE](version_check-0.9.5/LICENSE-APACHE), [LICENSE-MIT](version_check-0.9.5/LICENSE-MIT) |
| winapi 0.3.9 | Runtime | [source](https://crates.io/api/v1/crates/winapi/0.3.9/download) | [LICENSE-APACHE](winapi-0.3.9/LICENSE-APACHE), [LICENSE-MIT](winapi-0.3.9/LICENSE-MIT) |

## MPL-2.0

| Package | Role | Source | License texts |
| --- | --- | --- | --- |
| option-ext 0.2.0 | Runtime | [source](https://crates.io/api/v1/crates/option-ext/0.2.0/download) | [LICENSE.txt](option-ext-0.2.0/LICENSE.txt) |

## Unicode-3.0

| Package | Role | Source | License texts |
| --- | --- | --- | --- |
| icu_collections 2.2.0 | Runtime | [source](https://crates.io/api/v1/crates/icu_collections/2.2.0/download) | [LICENSE](icu_collections-2.2.0/LICENSE) |
| icu_locale_core 2.2.0 | Runtime | [source](https://crates.io/api/v1/crates/icu_locale_core/2.2.0/download) | [LICENSE](icu_locale_core-2.2.0/LICENSE) |
| icu_normalizer 2.2.0 | Runtime | [source](https://crates.io/api/v1/crates/icu_normalizer/2.2.0/download) | [LICENSE](icu_normalizer-2.2.0/LICENSE) |
| icu_normalizer_data 2.2.0 | Runtime | [source](https://crates.io/api/v1/crates/icu_normalizer_data/2.2.0/download) | [LICENSE](icu_normalizer_data-2.2.0/LICENSE) |
| icu_properties 2.2.0 | Runtime | [source](https://crates.io/api/v1/crates/icu_properties/2.2.0/download) | [LICENSE](icu_properties-2.2.0/LICENSE) |
| icu_properties_data 2.2.0 | Runtime | [source](https://crates.io/api/v1/crates/icu_properties_data/2.2.0/download) | [LICENSE](icu_properties_data-2.2.0/LICENSE) |
| icu_provider 2.2.0 | Runtime | [source](https://crates.io/api/v1/crates/icu_provider/2.2.0/download) | [LICENSE](icu_provider-2.2.0/LICENSE) |
| litemap 0.8.3 | Runtime | [source](https://crates.io/api/v1/crates/litemap/0.8.3/download) | [LICENSE](litemap-0.8.3/LICENSE) |
| potential_utf 0.1.6 | Runtime | [source](https://crates.io/api/v1/crates/potential_utf/0.1.6/download) | [LICENSE](potential_utf-0.1.6/LICENSE) |
| tinystr 0.8.4 | Runtime | [source](https://crates.io/api/v1/crates/tinystr/0.8.4/download) | [LICENSE](tinystr-0.8.4/LICENSE) |
| writeable 0.6.4 | Runtime | [source](https://crates.io/api/v1/crates/writeable/0.6.4/download) | [LICENSE](writeable-0.6.4/LICENSE) |
| yoke 0.8.3 | Runtime | [source](https://crates.io/api/v1/crates/yoke/0.8.3/download) | [LICENSE](yoke-0.8.3/LICENSE) |
| yoke-derive 0.8.4 | Build/proc-macro only | [source](https://crates.io/api/v1/crates/yoke-derive/0.8.4/download) | [LICENSE](yoke-derive-0.8.4/LICENSE) |
| zerofrom 0.1.8 | Runtime | [source](https://crates.io/api/v1/crates/zerofrom/0.1.8/download) | [LICENSE](zerofrom-0.1.8/LICENSE) |
| zerofrom-derive 0.1.8 | Build/proc-macro only | [source](https://crates.io/api/v1/crates/zerofrom-derive/0.1.8/download) | [LICENSE](zerofrom-derive-0.1.8/LICENSE) |
| zerotrie 0.2.5 | Runtime | [source](https://crates.io/api/v1/crates/zerotrie/0.2.5/download) | [LICENSE](zerotrie-0.2.5/LICENSE) |
| zerovec 0.11.8 | Runtime | [source](https://crates.io/api/v1/crates/zerovec/0.11.8/download) | [LICENSE](zerovec-0.11.8/LICENSE) |
| zerovec-derive 0.11.6 | Build/proc-macro only | [source](https://crates.io/api/v1/crates/zerovec-derive/0.11.6/download) | [LICENSE](zerovec-derive-0.11.6/LICENSE) |

## Unlicense OR MIT

| Package | Role | Source | License texts |
| --- | --- | --- | --- |
| aho-corasick 1.1.5 | Runtime | [source](https://crates.io/api/v1/crates/aho-corasick/1.1.5/download) | [COPYING](aho-corasick-1.1.5/COPYING), [LICENSE-MIT](aho-corasick-1.1.5/LICENSE-MIT) |
| byteorder 1.5.0 | Runtime | [source](https://crates.io/api/v1/crates/byteorder/1.5.0/download) | [COPYING](byteorder-1.5.0/COPYING), [LICENSE-MIT](byteorder-1.5.0/LICENSE-MIT) |
| byteorder-lite 0.1.0 | Runtime | [source](https://crates.io/api/v1/crates/byteorder-lite/0.1.0/download) | [LICENSE-MIT](byteorder-lite-0.1.0/LICENSE-MIT) |
| memchr 2.8.3 | Runtime | [source](https://crates.io/api/v1/crates/memchr/2.8.3/download) | [COPYING](memchr-2.8.3/COPYING), [LICENSE-MIT](memchr-2.8.3/LICENSE-MIT) |
| turbojpeg 1.5.1 | Runtime | [source](https://crates.io/api/v1/crates/turbojpeg/1.5.1/download) | [LICENSE](turbojpeg-1.5.1/LICENSE) |
| turbojpeg-sys 1.2.0 | Runtime | [source](https://crates.io/api/v1/crates/turbojpeg-sys/1.2.0/download) | [LICENSE.md](turbojpeg-sys-1.2.0/libjpeg-turbo/LICENSE.md), [README.ijg](turbojpeg-sys-1.2.0/libjpeg-turbo/README.ijg), [License.rtf](turbojpeg-sys-1.2.0/libjpeg-turbo/release/License.rtf), [LICENSE](turbojpeg-sys-1.2.0/upstream/LICENSE) |

## Zlib

| Package | Role | Source | License texts |
| --- | --- | --- | --- |
| foldhash 0.1.5 | Runtime | [source](https://crates.io/api/v1/crates/foldhash/0.1.5/download) | [LICENSE](foldhash-0.1.5/LICENSE) |
| foldhash 0.2.0 | Runtime | [source](https://crates.io/api/v1/crates/foldhash/0.2.0/download) | [LICENSE](foldhash-0.2.0/LICENSE) |
| zlib-rs 0.6.8 | Runtime | [source](https://crates.io/api/v1/crates/zlib-rs/0.6.8/download) | [LICENSE](zlib-rs-0.6.8/LICENSE) |

## Zlib OR Apache-2.0 OR MIT

| Package | Role | Source | License texts |
| --- | --- | --- | --- |
| bytemuck 1.25.2 | Runtime | [source](https://crates.io/api/v1/crates/bytemuck/1.25.2/download) | [LICENSE-APACHE](bytemuck-1.25.2/LICENSE-APACHE), [LICENSE-MIT](bytemuck-1.25.2/LICENSE-MIT), [LICENSE-ZLIB](bytemuck-1.25.2/LICENSE-ZLIB) |
| bytemuck_derive 1.12.1 | Build/proc-macro only | [source](https://crates.io/api/v1/crates/bytemuck_derive/1.12.1/download) | [LICENSE-APACHE](bytemuck_derive-1.12.1/LICENSE-APACHE), [LICENSE-MIT](bytemuck_derive-1.12.1/LICENSE-MIT), [LICENSE-ZLIB](bytemuck_derive-1.12.1/LICENSE-ZLIB) |
