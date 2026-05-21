# Changelog

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
