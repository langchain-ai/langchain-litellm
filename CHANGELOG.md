# Changelog

## [0.8.0](https://github.com/langchain-ai/langchain-litellm/compare/langchain-litellm==0.7.2...langchain-litellm==0.8.0) (2026-09-23)


### ⚠ BREAKING CHANGES

* **chat_models:** `.stream()` and `.astream()` now stream incrementally on an instance that did not pass `streaming=True`. They previously returned the whole response as a single chunk, because the default `streaming=False` was recorded as an explicit opt-out. Code that assumed one chunk per call, or that expected `astream_events` to emit a single `on_chat_model_stream`, now receives one per token. Pass `streaming=False` explicitly to keep the previous behaviour.
* **embeddings:** `LiteLLMEmbeddings` rejects unknown constructor kwargs rather than discarding them. `timeout`, `max_tokens`, `client` and `streaming` were accepted and ignored; use the declared `request_timeout`, or pass provider values through `model_kwargs`.
* **chat_models:** `ChatLiteLLMRouter` no longer copies litellm's router metadata into `response_metadata`. Read `response_cost` instead of `hidden_params.response_cost`, and `model_id` instead of `hidden_params.model_id`; both are now present on the streaming paths too, where none of these keys ever appeared. `api_base`, `attempted_fallbacks`, `attempted_retries`, `caching_groups`, `deployment`, `deployment_model_name`, `hidden_params`, `max_retries`, `model_group`, `model_group_alias`, `model_group_size`, `model_info` and `original_model_group` are gone with no replacement.

### Features

* **chat_models:** surface the response cost in response_metadata ([#281](https://github.com/langchain-ai/langchain-litellm/issues/281)) ([5d04bb8](https://github.com/langchain-ai/langchain-litellm/commit/5d04bb85cb5598b692d55cd7500c74eb25eebc58))
* make .stream() stream, honour embeddings retries, and stop dropping credentials ([#272](https://github.com/langchain-ai/langchain-litellm/issues/272)) ([08cc7be](https://github.com/langchain-ai/langchain-litellm/commit/08cc7bec40f1ccdc1028a39da33d23467697b0a6))


### Bug Fixes

* accept `base_url` as an alias for `api_base` in `ChatLiteLLM` ([#200](https://github.com/langchain-ai/langchain-litellm/issues/200)) ([14d8c07](https://github.com/langchain-ai/langchain-litellm/commit/14d8c072dc4f4ad2bad95be5d7bd5ed91397b03a))
* accept base_url for LiteLLM embeddings ([#203](https://github.com/langchain-ai/langchain-litellm/issues/203)) ([e5b2e5e](https://github.com/langchain-ai/langchain-litellm/commit/e5b2e5e5b765ffca5f63db4e4346688060b36315))
* **chat_models:** forward provider-specific api_key fields to litellm ([#261](https://github.com/langchain-ai/langchain-litellm/issues/261)) ([b57c1e4](https://github.com/langchain-ai/langchain-litellm/commit/b57c1e409c42ca03f08872f9055872e35506e391))
* **chat_models:** honor per-call model override in _get_ls_params ([#248](https://github.com/langchain-ai/langchain-litellm/issues/248)) ([c2d4fec](https://github.com/langchain-ai/langchain-litellm/commit/c2d4fec22aac0e04e9a1a338f2085412e8983f14))
* **chat_models:** include top_p and top_k in _default_params ([#233](https://github.com/langchain-ai/langchain-litellm/issues/233)) ([a03a841](https://github.com/langchain-ai/langchain-litellm/commit/a03a8415584b6014170ca5fb14077858b390f465))
* **chat_models:** name a streamed cost once ([#284](https://github.com/langchain-ai/langchain-litellm/issues/284)) ([919944d](https://github.com/langchain-ai/langchain-litellm/commit/919944d1a04a867eec278b1b21e1eb118cc46dc1))
* **chat_models:** omit top_p and top_k when the caller left them unset ([#278](https://github.com/langchain-ai/langchain-litellm/issues/278)) ([e0c222e](https://github.com/langchain-ai/langchain-litellm/commit/e0c222e0865fc59e3fb6666bbc998209446c2930))
* **chat_models:** report unparsable tool-call arguments as invalid ([#260](https://github.com/langchain-ai/langchain-litellm/issues/260)) ([c93299f](https://github.com/langchain-ai/langchain-litellm/commit/c93299f9dc5f951cad3a1a94d7b540fd8c2d7e12))
* **chat_models:** stop echoing an unparsable tool call back to the provider ([#279](https://github.com/langchain-ai/langchain-litellm/issues/279)) ([a3bf8ec](https://github.com/langchain-ai/langchain-litellm/commit/a3bf8ec8d6cbbf799779aa21c2d3c84c3b71c53d))
* **chat_models:** stop injecting thinking blocks into AIMessage.content ([#244](https://github.com/langchain-ai/langchain-litellm/issues/244)) ([a47db29](https://github.com/langchain-ai/langchain-litellm/commit/a47db29cb7dfeeaac13ba3e2401f86474ec256e4))
* **chat_models:** stop republishing litellm's router bookkeeping ([#283](https://github.com/langchain-ai/langchain-litellm/issues/283)) ([c568593](https://github.com/langchain-ai/langchain-litellm/commit/c56859392ae97cd06f04651e2c2d6d2fe31546f7))
* **chat_models:** surface finish_reason in streamed response_metadata ([#241](https://github.com/langchain-ai/langchain-litellm/issues/241)) ([b0ca0c3](https://github.com/langchain-ai/langchain-litellm/commit/b0ca0c30644f34236288298dd8f88285d5c91b23))
* **deps:** declare pydantic and typing-extensions ([#280](https://github.com/langchain-ai/langchain-litellm/issues/280)) ([57ae70e](https://github.com/langchain-ai/langchain-litellm/commit/57ae70ea2aaf62619902b8e050201cfcd50fa8cc))
* **router:** honour max_retries in ChatLiteLLMRouter ([#262](https://github.com/langchain-ai/langchain-litellm/issues/262)) ([dd8b4f1](https://github.com/langchain-ai/langchain-litellm/commit/dd8b4f1c61c6c8ff7794eec21d1899be486a8a3b))

## [0.7.2](https://github.com/langchain-ai/langchain-litellm/compare/langchain-litellm==0.7.1...langchain-litellm==0.7.2) (2026-09-16)


### Bug Fixes

* **deps:** drop the unused cryptography dependency ([#267](https://github.com/langchain-ai/langchain-litellm/issues/267)) ([8aa4fc6](https://github.com/langchain-ai/langchain-litellm/commit/8aa4fc6664a1f790082093a2cc8b2d3dc05bd2dd))

## [0.7.1](https://github.com/langchain-ai/langchain-litellm/compare/langchain-litellm==0.7.0...langchain-litellm==0.7.1) (2026-08-31)


### Bug Fixes

* **litellm:** bump langchain-core ([#245](https://github.com/langchain-ai/langchain-litellm/issues/245)) ([060fe0c](https://github.com/langchain-ai/langchain-litellm/commit/060fe0c623b3d8ad53bd2b0c43877621065f201c))

## [0.7.0](https://github.com/langchain-ai/langchain-litellm/compare/langchain-litellm==0.6.6...langchain-litellm==0.7.0) (2026-06-15)


### Features

* **langchain-litellm:** record package version in model metadata ([#192](https://github.com/langchain-ai/langchain-litellm/issues/192)) ([1c0dcf3](https://github.com/langchain-ai/langchain-litellm/commit/1c0dcf3c0a451b43be39129106b7a8c272d89064))

## [0.6.6](https://github.com/langchain-ai/langchain-litellm/compare/langchain-litellm==0.6.5...langchain-litellm==0.6.6) (2026-05-21)


### Bug Fixes

* bug when api_key is provided but not passed and is overridden by environment variables ([#134](https://github.com/langchain-ai/langchain-litellm/issues/134)) ([1b12aeb](https://github.com/langchain-ai/langchain-litellm/commit/1b12aeb7ecbdc576f4428282f6b27c0122eb6905))
* populate `model_provider` in `response_metadata` and `ls_provider` in `_get_ls_params` ([#152](https://github.com/langchain-ai/langchain-litellm/issues/152)) ([#158](https://github.com/langchain-ai/langchain-litellm/issues/158)) ([91004e2](https://github.com/langchain-ai/langchain-litellm/commit/91004e23b8641f41f78ed00e970cb2dd77286369))
* remove global litellm module mutations from _client_params ([#132](https://github.com/langchain-ai/langchain-litellm/issues/132)) ([#161](https://github.com/langchain-ai/langchain-litellm/issues/161)) ([a7ca120](https://github.com/langchain-ai/langchain-litellm/commit/a7ca12035814c6a88cc606be8cb1e3d6277e7eba))
* strip thinking/redacted_thinking blocks from messages sent to non-Anthropic providers ([#159](https://github.com/langchain-ai/langchain-litellm/issues/159)) ([ce809af](https://github.com/langchain-ai/langchain-litellm/commit/ce809af72b5390705cf0b2563107790f05e28143))

## [0.6.5](https://github.com/langchain-ai/langchain-litellm/compare/langchain-litellm==0.6.4...langchain-litellm==0.6.5) (2026-05-07)


### Bug Fixes

* bump litellm floor to 1.83.14 to clear downstream CVE alerts ([a1370a6](https://github.com/langchain-ai/langchain-litellm/commit/a1370a6f6bd4a88eb2361aea9640883160a56674)), closes [#141](https://github.com/langchain-ai/langchain-litellm/issues/141)
* patch 3 security alerts (critical+high severity) in litellm ([#137](https://github.com/langchain-ai/langchain-litellm/issues/137)) ([b170dcc](https://github.com/langchain-ai/langchain-litellm/commit/b170dcc6278dd741565402d7dbdcb409d1756643))

## [0.6.4](https://github.com/langchain-ai/langchain-litellm/compare/langchain-litellm==0.6.3...langchain-litellm==0.6.4) (2026-04-03)


### Bug Fixes

* downgrade `tool_choice` to auto when thinking is enabled on Claude ([#126](https://github.com/langchain-ai/langchain-litellm/issues/126)) ([8912d08](https://github.com/langchain-ai/langchain-litellm/commit/8912d087d5f12933fedb7421abc09e54e562f13c))
* extract reasoning tokens and handle pydantic usage in metadata ([#121](https://github.com/langchain-ai/langchain-litellm/issues/121)) ([ae18705](https://github.com/langchain-ai/langchain-litellm/commit/ae18705f6ad8d171a60db57353bcfeab29b76a0f))
* filter `tool_use` content blocks from AI message dicts ([#125](https://github.com/langchain-ai/langchain-litellm/issues/125)) ([06c893d](https://github.com/langchain-ai/langchain-litellm/commit/06c893dead994e989a8c74a157804cac6fe875b6))
* populate `model_name` in `response_metadata` for streaming and router paths ([#124](https://github.com/langchain-ai/langchain-litellm/issues/124)) ([df1216c](https://github.com/langchain-ai/langchain-litellm/commit/df1216cf4dc94d10d3ea51d4b7ee98c61a6c3d37))
* test ([f066e3d](https://github.com/langchain-ai/langchain-litellm/commit/f066e3ded2d2297cc4501bce2ca114e30e2a9a99))

## [0.6.3](https://github.com/langchain-ai/langchain-litellm/compare/v0.6.2...v0.6.3) (2026-04-01)


### Bug Fixes

* set usage_metadata on AIMessage in _create_chat_result ([#102](https://github.com/langchain-ai/langchain-litellm/issues/102)) ([75766a0](https://github.com/langchain-ai/langchain-litellm/commit/75766a05027efcff732f5ff48dabc69a8a7af7a7))

## [0.6.2](https://github.com/langchain-ai/langchain-litellm/compare/v0.6.1...v0.6.2) (2026-03-24)


### Bug Fixes

* exclude compromised litellm versions from deps ([#103](https://github.com/langchain-ai/langchain-litellm/issues/103)) ([701e152](https://github.com/langchain-ai/langchain-litellm/commit/701e1523eeeefa8826a298d3dc275d3ca7ff106e))

## [0.6.1](https://github.com/langchain-ai/langchain-litellm/compare/v0.6.0...v0.6.1) (2026-03-01)


### Bug Fixes

* clean up build artifacts to unblock PyPI publication ([#95](https://github.com/langchain-ai/langchain-litellm/issues/95)) ([02e7456](https://github.com/langchain-ai/langchain-litellm/commit/02e74567ea3a22e18bf011a52e51aae7b2f2f5a5))

## [0.6.0](https://github.com/langchain-ai/langchain-litellm/compare/v0.5.1...v0.6.0) (2026-03-01)


### Features

* add LiteLLMEmbeddings and LiteLLMEmbeddingsRouter ([#88](https://github.com/langchain-ai/langchain-litellm/issues/88)) ([2bace91](https://github.com/langchain-ai/langchain-litellm/commit/2bace9185918964a5e6047190ef86e9495ff7e64))
