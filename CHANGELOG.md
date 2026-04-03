# Changelog

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
