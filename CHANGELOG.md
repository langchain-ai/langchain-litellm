# Changelog

## [0.7.0](https://github.com/langchain-ai/langchain-litellm/compare/langchain-litellm==0.6.3...langchain-litellm==0.7.0) (2026-04-01)


### Features

* add configurable timeout and retry logic to LiteLLMOCRLoader ([#68](https://github.com/langchain-ai/langchain-litellm/issues/68)) ([1c079a5](https://github.com/langchain-ai/langchain-litellm/commit/1c079a5f64cea85146247da577f2602da45f8b6e))
* add LiteLLMEmbeddings and LiteLLMEmbeddingsRouter ([#88](https://github.com/langchain-ai/langchain-litellm/issues/88)) ([2bace91](https://github.com/langchain-ai/langchain-litellm/commit/2bace9185918964a5e6047190ef86e9495ff7e64))
* add structured output support with schema validation in ChatLiteLLM ([#36](https://github.com/langchain-ai/langchain-litellm/issues/36)) ([3e6ef60](https://github.com/langchain-ai/langchain-litellm/commit/3e6ef609d57f403dc150b5209cb3109e25c796db))
* add support for extra headers in ChatLiteLLM ([#35](https://github.com/langchain-ai/langchain-litellm/issues/35)) ([1846a85](https://github.com/langchain-ai/langchain-litellm/commit/1846a85f80ef5d174894a390c2dfe7229e47a7db))
* added support for ChatLiteLLMRouter ([#1](https://github.com/langchain-ai/langchain-litellm/issues/1)) ([5cb4bb2](https://github.com/langchain-ai/langchain-litellm/commit/5cb4bb2489f86b8e3aab7412a96d826ae5343189))


### Bug Fixes

* ainvoke and astream openai completion api ([#24](https://github.com/langchain-ai/langchain-litellm/issues/24)) ([7675be8](https://github.com/langchain-ai/langchain-litellm/commit/7675be857da88c399b1f1dfc162eb7eb2f05fbc5))
* allow reasoning_content and function_call to coexist in message chunks ([#55](https://github.com/langchain-ai/langchain-litellm/issues/55)) ([338672d](https://github.com/langchain-ai/langchain-litellm/commit/338672d5b695da2288537318b4dae0a76e8856e3))
* clean up build artifacts to unblock PyPI publication ([#95](https://github.com/langchain-ai/langchain-litellm/issues/95)) ([02e7456](https://github.com/langchain-ai/langchain-litellm/commit/02e74567ea3a22e18bf011a52e51aae7b2f2f5a5))
* exclude compromised litellm versions from deps ([#103](https://github.com/langchain-ai/langchain-litellm/issues/103)) ([701e152](https://github.com/langchain-ai/langchain-litellm/commit/701e1523eeeefa8826a298d3dc275d3ca7ff106e))
* Expose cache tokens in streaming responses ([#53](https://github.com/langchain-ai/langchain-litellm/issues/53)) ([cb02c23](https://github.com/langchain-ai/langchain-litellm/commit/cb02c23be6cd81795e9af0662b742fdf0208a96f)), closes [#52](https://github.com/langchain-ai/langchain-litellm/issues/52)
* filter out None values from params in ChatLiteLLMRouter methods ([#37](https://github.com/langchain-ai/langchain-litellm/issues/37)) ([6579a2b](https://github.com/langchain-ai/langchain-litellm/commit/6579a2b0137729b606c238535682d6c9226bec05))
* fixed streaming with tool calls ([#5](https://github.com/langchain-ai/langchain-litellm/issues/5)) ([#6](https://github.com/langchain-ai/langchain-litellm/issues/6)) ([0ddac98](https://github.com/langchain-ai/langchain-litellm/commit/0ddac986f3b489a62278571c9256eb9463c9778f))
* set usage_metadata on AIMessage in _create_chat_result ([#102](https://github.com/langchain-ai/langchain-litellm/issues/102)) ([75766a0](https://github.com/langchain-ai/langchain-litellm/commit/75766a05027efcff732f5ff48dabc69a8a7af7a7))
* streaming delta conversion to handle dict types ([#14](https://github.com/langchain-ai/langchain-litellm/issues/14)) ([8073686](https://github.com/langchain-ai/langchain-litellm/commit/807368662f854fae27e9b8c927f9e1fd80ebf3fd))

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
