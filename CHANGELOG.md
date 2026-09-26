# Changelog

## [0.10.0](https://github.com/langchain-ai/langchain-litellm/compare/langchain-litellm==0.9.0...langchain-litellm==0.10.0) (2026-09-26)


### ⚠ BREAKING CHANGES

* **chat_models:** a tool_choice string other than auto, none, required or any is read as a tool name and must name a bound function tool, or bind_tools raises ValueError. On mistral/, codestral/ and Vertex AI Mistral models a tool name no longer forces a tool call, since litellm drops the function choice there; pass "required" to keep forcing. On Claude with thinking enabled other than through model_kwargs["thinking"], the named tool is now forced, which Anthropic rejects alongside thinking.
* **chat_models:** `use_responses_api=True`, which `ChatLiteLLM` ignored, now sends calls to the Responses API, where litellm drops Chat Completions-only params such as `stop`, `n` and `seed`, and raises `ValueError` for a model litellm cannot bridge; `ChatLiteLLMRouter` raises `ValueError` for it. Drop the flag to stay on Chat Completions.

### Features

* **chat_models:** route calls through the Responses API with use_responses_api ([#301](https://github.com/langchain-ai/langchain-litellm/issues/301)) ([2a7a4a2](https://github.com/langchain-ai/langchain-litellm/commit/2a7a4a29fb7325827e47f654a7e1b127f634ecb3))


### Bug Fixes

* **chat_models:** downgrade a forced tool_choice wherever thinking is manual ([#313](https://github.com/langchain-ai/langchain-litellm/issues/313)) ([079aa08](https://github.com/langchain-ai/langchain-litellm/commit/079aa08c2322e71211292e1c2af2a763d30e8b6d))
* **chat_models:** force the tool a string tool_choice names ([#311](https://github.com/langchain-ai/langchain-litellm/issues/311)) ([cad44b6](https://github.com/langchain-ai/langchain-litellm/commit/cad44b69dcedf38f19db77c883d16f0cb350b80d))
* **chat_models:** rejoin a reply litellm splits across choices ([#307](https://github.com/langchain-ai/langchain-litellm/issues/307)) ([a6fe164](https://github.com/langchain-ai/langchain-litellm/commit/a6fe164375cc1651bd34247816733a6dfe66560d))
* **chat_models:** replay signed thinking blocks to the endpoint that signed them ([#300](https://github.com/langchain-ai/langchain-litellm/issues/300)) ([5cfd986](https://github.com/langchain-ai/langchain-litellm/commit/5cfd986709e7b183a6f7188051a15227624cb6fc))
* **chat_models:** stop bind_tools raising KeyError on a dict tool_choice ([#310](https://github.com/langchain-ai/langchain-litellm/issues/310)) ([9db3498](https://github.com/langchain-ai/langchain-litellm/commit/9db349870ccb4017d38f450c8f93b3eb0a781042))

## [0.9.0](https://github.com/langchain-ai/langchain-litellm/compare/langchain-litellm==0.8.1...langchain-litellm==0.9.0) (2026-09-24)


### ⚠ BREAKING CHANGES

* langchain-litellm now requires Python 3.11 or later. Stay on langchain-litellm 0.8.x to keep using Python 3.10.

### Features

* drop support for Python 3.10 ([3d75c58](https://github.com/langchain-ai/langchain-litellm/commit/3d75c58b64c0f283156a63baff3293a64eb73b7c))

## [0.8.1](https://github.com/langchain-ai/langchain-litellm/compare/langchain-litellm==0.8.0...langchain-litellm==0.8.1) (2026-09-24)


### Bug Fixes

* **chat_models:** preserve file metadata after normalization ([#289](https://github.com/langchain-ai/langchain-litellm/issues/289)) ([46abd65](https://github.com/langchain-ai/langchain-litellm/commit/46abd6586b269967fa5f726658485fadb3db9185))
* **deps:** exclude litellm releases that fail to import on Python 3.10 ([#290](https://github.com/langchain-ai/langchain-litellm/issues/290)) ([092b466](https://github.com/langchain-ai/langchain-litellm/commit/092b466468d32f1144e8f5487d7c09be0eb92f90))

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

*Never published to PyPI. 0.6.1, released the same day, is the first published version to include these changes.*


### Features

* add LiteLLMEmbeddings and LiteLLMEmbeddingsRouter ([#88](https://github.com/langchain-ai/langchain-litellm/issues/88)) ([2bace91](https://github.com/langchain-ai/langchain-litellm/commit/2bace9185918964a5e6047190ef86e9495ff7e64))

*Versions 0.1.0 through 0.5.1 predate automated changelog generation and were reconstructed from git history. Dates are PyPI publication dates. Documentation, CI and release-tooling commits are omitted, as they are for later versions.*

## [0.5.1](https://github.com/langchain-ai/langchain-litellm/compare/v0.5.0...v0.5.1) (2026-02-11)


### Features

* add configurable timeout and retry logic to LiteLLMOCRLoader ([#68](https://github.com/langchain-ai/langchain-litellm/issues/68)) ([1c079a5](https://github.com/langchain-ai/langchain-litellm/commit/1c079a5f64cea85146247da577f2602da45f8b6e))

**Contributors:** [@Akshay-Dongare](https://github.com/Akshay-Dongare)

## [0.5.0](https://github.com/langchain-ai/langchain-litellm/compare/v0.4.0...v0.5.0) (2026-02-09)


### Features

* Initial LiteLLM OCR Loader ([#65](https://github.com/langchain-ai/langchain-litellm/issues/65)) ([ad733f1](https://github.com/langchain-ai/langchain-litellm/commit/ad733f13d9325be4ac1ed392fb75a046067673ae))

**Contributors:** [@Bschim](https://github.com/Bschim)

## [0.4.0](https://github.com/langchain-ai/langchain-litellm/compare/v0.3.5...v0.4.0) (2026-02-02)


### Features

* Add support for base_model parameter to fix fine-tuned Vertex AI models ([#63](https://github.com/langchain-ai/langchain-litellm/issues/63)) ([ed37959](https://github.com/langchain-ai/langchain-litellm/commit/ed37959860b7207450c3a96c590870f0c5702704))
* Add support for gemini-2.5-pro by updating litellm dependency ([#60](https://github.com/langchain-ai/langchain-litellm/issues/60)) ([132d71c](https://github.com/langchain-ai/langchain-litellm/commit/132d71c528451bb6a45171891bdaffaa9f8af7cd))
* Add support for num_ctx parameter and improve documentation ([#57](https://github.com/langchain-ai/langchain-litellm/issues/57)) ([3cbbb43](https://github.com/langchain-ai/langchain-litellm/commit/3cbbb439d923929f3444101956506c1a6a8b502e))


### Bug Fixes

* allow reasoning_content and function_call to coexist in message chunks ([#55](https://github.com/langchain-ai/langchain-litellm/issues/55)) ([338672d](https://github.com/langchain-ai/langchain-litellm/commit/338672d5b695da2288537318b4dae0a76e8856e3))
* Expose cache tokens in streaming responses ([#53](https://github.com/langchain-ai/langchain-litellm/issues/53)) ([cb02c23](https://github.com/langchain-ai/langchain-litellm/commit/cb02c23be6cd81795e9af0662b742fdf0208a96f))
* Fix AttributeError in streaming when chunks lack role or function_call attributes ([#59](https://github.com/langchain-ai/langchain-litellm/issues/59)) ([62488a0](https://github.com/langchain-ai/langchain-litellm/commit/62488a01277e978020674a645dcdd98797761ef0))
* Fix missing usage metadata in streaming responses ([#56](https://github.com/langchain-ai/langchain-litellm/issues/56)) ([ae8b263](https://github.com/langchain-ai/langchain-litellm/commit/ae8b263b3365a93f50b4f1d69b1a1e04b985fe45))
* Fix multimodal content handling and preserve LiteLLM native format ([#61](https://github.com/langchain-ai/langchain-litellm/issues/61)) ([5e2eed1](https://github.com/langchain-ai/langchain-litellm/commit/5e2eed19795e5b9a0f50e6f4ac972c25d2a9ba71))
* Fix request_timeout parameter not being respected ([#58](https://github.com/langchain-ai/langchain-litellm/issues/58)) ([f231574](https://github.com/langchain-ai/langchain-litellm/commit/f23157461dff779e515ef1446188aa2d4e498cb2))
* Fix streaming crash on Bedrock while preserving OpenAI usage stats ([#64](https://github.com/langchain-ai/langchain-litellm/issues/64)) ([28edfda](https://github.com/langchain-ai/langchain-litellm/commit/28edfda96af300e8bebc1d195d870a2825eb61e4))
* Fix structured output crashes: handle tool_choice and Dict arguments ([#62](https://github.com/langchain-ai/langchain-litellm/issues/62)) ([7793f17](https://github.com/langchain-ai/langchain-litellm/commit/7793f172e331d03fe08acc0c941bed62edf56cb2))

**Contributors:** [@Akshay-Dongare](https://github.com/Akshay-Dongare)

## [0.3.5](https://github.com/langchain-ai/langchain-litellm/compare/v0.3.4...v0.3.5) (2025-12-13)


### Bug Fixes

* Inject Root Metadata into Delta ([#47](https://github.com/langchain-ai/langchain-litellm/issues/47)) ([7c239b9](https://github.com/langchain-ai/langchain-litellm/commit/7c239b9e43e29d0abc3ace50d6a7f84b1cecd737))

**Contributors:** [@Akshay-Dongare](https://github.com/Akshay-Dongare)

## [0.3.4](https://github.com/langchain-ai/langchain-litellm/compare/v0.3.3...v0.3.4) (2025-12-13)


### Features

* Add compatibility for Vertex AI specific grounding metadata field in Litellm chat model. ([#45](https://github.com/langchain-ai/langchain-litellm/issues/45)) ([6368a4d](https://github.com/langchain-ai/langchain-litellm/commit/6368a4dd97b1bc9ce6c457d9c472d26256e2eb43))
* Added the handling of 'vertex_ai_grounding_metadata' for provider_specific_fields in LiteLLMRouter. ([#45](https://github.com/langchain-ai/langchain-litellm/issues/45)) ([0ec4970](https://github.com/langchain-ai/langchain-litellm/commit/0ec4970cd46eee16ec2caedabaccccc46b983902))

**Contributors:** [@Akshay-Dongare](https://github.com/Akshay-Dongare)

## [0.3.3](https://github.com/langchain-ai/langchain-litellm/compare/v0.3.2...v0.3.3) (2025-12-11)


### Features

* Add "provider_specific_fields" support in chat models and router ([#43](https://github.com/langchain-ai/langchain-litellm/issues/43)) ([c56a352](https://github.com/langchain-ai/langchain-litellm/commit/c56a3527704613002bdf15eaa63fbec0ef8c7aa7))


### Bug Fixes

* Fix bug where reason content is not output in invoke ([#25](https://github.com/langchain-ai/langchain-litellm/issues/25)) ([dbe0134](https://github.com/langchain-ai/langchain-litellm/commit/dbe01341c2a184933e28bae2eca39067b99020f2))

**Contributors:** [@TBice123123](https://github.com/TBice123123), [@Akshay-Dongare](https://github.com/Akshay-Dongare)

## [0.3.2](https://github.com/langchain-ai/langchain-litellm/compare/v0.3.1...v0.3.2) (2025-11-20)


### Features

* Add logprobs to ChatResult for ChatLiteLLM ([#40](https://github.com/langchain-ai/langchain-litellm/issues/40)) ([c9e60c7](https://github.com/langchain-ai/langchain-litellm/commit/c9e60c7095142e86af1abdf45a2437b4d92ef22d))
* add structured output support with schema validation in ChatLiteLLM ([#36](https://github.com/langchain-ai/langchain-litellm/issues/36)) ([3e6ef60](https://github.com/langchain-ai/langchain-litellm/commit/3e6ef609d57f403dc150b5209cb3109e25c796db))
* add support for extra headers in ChatLiteLLM ([#35](https://github.com/langchain-ai/langchain-litellm/issues/35)) ([1846a85](https://github.com/langchain-ai/langchain-litellm/commit/1846a85f80ef5d174894a390c2dfe7229e47a7db))


### Bug Fixes

* filter out None values from params in ChatLiteLLMRouter methods ([#37](https://github.com/langchain-ai/langchain-litellm/issues/37)) ([6579a2b](https://github.com/langchain-ai/langchain-litellm/commit/6579a2b0137729b606c238535682d6c9226bec05))

**Contributors:** [@allen-cook](https://github.com/allen-cook), [@SamMaggioli](https://github.com/SamMaggioli), [@k4han](https://github.com/k4han)

## [0.3.1](https://github.com/langchain-ai/langchain-litellm/compare/v0.3.0...v0.3.1) (2025-11-20)


### Features

* Add support for LangChain version 1.0 ([#38](https://github.com/langchain-ai/langchain-litellm/issues/38)) ([03439b1](https://github.com/langchain-ai/langchain-litellm/commit/03439b1627971df24177569e91519a0398859912))

**Contributors:** [@quinlanjager](https://github.com/quinlanjager)

## [0.3.0](https://github.com/langchain-ai/langchain-litellm/compare/v0.2.3...v0.3.0) (2025-10-20)


### Features

* Add Usage Metadata in Streaming Responses ([#34](https://github.com/langchain-ai/langchain-litellm/issues/34)) ([a9abf29](https://github.com/langchain-ai/langchain-litellm/commit/a9abf297eb394880d2243a74e6a42735401fd0dc))

**Contributors:** [@RheagalFire](https://github.com/RheagalFire)

## [0.2.3](https://github.com/langchain-ai/langchain-litellm/compare/v0.2.2...v0.2.3) (2025-09-25)


### Bug Fixes

* ainvoke and astream openai completion api ([#24](https://github.com/langchain-ai/langchain-litellm/issues/24)) ([7675be8](https://github.com/langchain-ai/langchain-litellm/commit/7675be857da88c399b1f1dfc162eb7eb2f05fbc5))

**Contributors:** [@maxence-oden](https://github.com/maxence-oden)

## [0.2.2](https://github.com/langchain-ai/langchain-litellm/compare/v0.2.1...v0.2.2) (2025-07-11)


### Bug Fixes

* Fix for the Issue:_OPENAI_MODELS is out of date #7 ([#12](https://github.com/langchain-ai/langchain-litellm/issues/12)) ([64cbad1](https://github.com/langchain-ai/langchain-litellm/commit/64cbad1eda7b759866d6f9846994ef795090f6b6))
* Fix tool message conversion ([#11](https://github.com/langchain-ai/langchain-litellm/issues/11)) ([9fefd6d](https://github.com/langchain-ai/langchain-litellm/commit/9fefd6d64e48558b5f3d81d71838a08150ba17af))
* Solves issue #7 ([43b6adc](https://github.com/langchain-ai/langchain-litellm/commit/43b6adc71dffc2e6b8132563868ddc6d9700e744))
* streaming delta conversion to handle dict types ([#14](https://github.com/langchain-ai/langchain-litellm/issues/14)) ([8073686](https://github.com/langchain-ai/langchain-litellm/commit/807368662f854fae27e9b8c927f9e1fd80ebf3fd))
* Update temperature range ([846d3d2](https://github.com/langchain-ai/langchain-litellm/commit/846d3d2397cc1e844ffdc6c516c92d8e00761149))

**Contributors:** [@Akshay-Dongare](https://github.com/Akshay-Dongare), [@AshutoshDongare](https://github.com/AshutoshDongare), [@jeonsworld](https://github.com/jeonsworld)

## [0.2.1](https://github.com/langchain-ai/langchain-litellm/compare/v0.2.0...v0.2.1) (2025-05-12)


### Bug Fixes

* fixed streaming with tool calls ([#6](https://github.com/langchain-ai/langchain-litellm/issues/6)) ([0ddac98](https://github.com/langchain-ai/langchain-litellm/commit/0ddac986f3b489a62278571c9256eb9463c9778f))

**Contributors:** [@florianchappaz](https://github.com/florianchappaz)

## [0.2.0](https://github.com/langchain-ai/langchain-litellm/compare/v0.1.4...v0.2.0) (2025-04-26)


### Features

* added support for ChatLiteLLMRouter ([#2](https://github.com/langchain-ai/langchain-litellm/issues/2)) ([5cb4bb2](https://github.com/langchain-ai/langchain-litellm/commit/5cb4bb2489f86b8e3aab7412a96d826ae5343189))

**Contributors:** [@florianchappaz](https://github.com/florianchappaz)

## [0.1.4](https://github.com/langchain-ai/langchain-litellm/compare/0.1.3...v0.1.4) (2025-04-05)


Release tooling and documentation only, with no functional changes.

## [0.1.3](https://github.com/langchain-ai/langchain-litellm/compare/0.1.2...0.1.3) (2025-04-05)


### Features

* exporting ChatLiteLLM from \_\_init\_\_.py so you can do from langchain_litellm import ChatLiteLLM ([746e14c](https://github.com/langchain-ai/langchain-litellm/commit/746e14c82b54dc54f6135928a02e3afb7226992e))

**Contributors:** [@Akshay-Dongare](https://github.com/Akshay-Dongare)

## [0.1.2](https://github.com/langchain-ai/langchain-litellm/compare/bce995d...0.1.2) (2025-04-04)


Release tooling and documentation only, with no functional changes.

## [0.1.1](https://github.com/langchain-ai/langchain-litellm/compare/9ce92c0...bce995d) (2025-04-03)


### Bug Fixes

* solved poetry dependency issues and rewrote integration tests ([4e92b40](https://github.com/langchain-ai/langchain-litellm/commit/4e92b40d2c145bfd44e7a51de5cf7c2a3e425b51))

**Contributors:** [@Akshay-Dongare](https://github.com/Akshay-Dongare)

## [0.1.0](https://github.com/langchain-ai/langchain-litellm/tree/9ce92c0cd9e0a432f431e6a04aea11018e4e1729) (2025-04-03)


### Features

* added chat_model code and docs ([479fd92](https://github.com/langchain-ai/langchain-litellm/commit/479fd92ffa57b62372778bca25f69fd56ef82e46))

**Contributors:** [@Akshay-Dongare](https://github.com/Akshay-Dongare)
