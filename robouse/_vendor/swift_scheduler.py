# Copyright (c) ModelScope Contributors. All rights reserved.
# Extracted from ms-swift 4.2.1. Only imports are decoupled from vLLM.
from __future__ import annotations
import asyncio
from abc import ABC
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union
from .swift_protocol import (
    RolloutOutput,
    RolloutInferRequest,
    RequestConfig,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
)
from .swift_utils import remove_response


class RolloutScheduler(ABC):
    def __init__(
        self,
        infer_engine: Optional[GRPOVllmEngine] = None,
        max_turns: Optional[int] = None,
        *args,
        **kwargs,
    ):
        self.infer_engine = infer_engine
        self._tokenizer = kwargs.get("tokenizer", None)
        self.max_turns = max_turns

    async def async_infer(
        self,
        infer_requests: List[Union["RolloutInferRequest", Dict[str, Any]]],
        request_config: "RequestConfig",
        *,
        use_tqdm: Optional[bool] = None,
        **kwargs,
    ) -> List["RolloutOutput"]:
        """
        Perform asynchronous batched inference for multiple rollout requests.

        This method serves as the main entry point for multi-round training inference.
        It executes the `run` method for each inference request concurrently and
        aggregates the results into a single flattened list.

        Each inference request can be either a `RolloutInferRequest` instance or a
        dictionary that can be converted into one. The results from all requests are
        collected asynchronously using the underlying inference engine.

        Args:
            infer_requests (List[Union[RolloutInferRequest, Dict[str, Any]]]):
                A list of inference requests. Each request can be either:
                    - A `RolloutInferRequest` object.
                    - A dictionary containing the fields required to initialize a
                    `RolloutInferRequest`.
            request_config (RequestConfig):
                Configuration object specifying inference settings. Must satisfy
                `request_config.n == 1`, as only single-response generation is supported.
            use_tqdm (Optional[bool], optional):
                Whether to display a progress bar during batch inference.
                If `None`, it defaults to `True` when there are multiple requests,
                otherwise `False`.
            **kwargs:
                Additional arguments forwarded to the underlying `run` method.

        Returns:
            List[RolloutOutput]:
                A list of RolloutOutput objects corresponding to the provided inference requests.

        Raises:
            AssertionError:
                If `request_config.n` is not equal to `1`.

        Notes:
            - Internally, this method converts dict-based requests into
            `RolloutInferRequest` instances.
            - Uses `infer_engine._batch_infer_stream` to perform concurrent execution.
            - The returned list is guaranteed to be flattened, even if individual
            tasks return lists of responses.
        """
        assert request_config.n == 1

        async def _infer_async_single(
            infer_request: Union["RolloutInferRequest", Dict[str, Any]],
            request_config: "RequestConfig",
            **kwargs,
        ):
            if isinstance(infer_request, Dict):
                infer_request = RolloutInferRequest(**infer_request)
            return await self.run(infer_request, request_config, **kwargs)

        tasks = [
            _infer_async_single(infer_request, request_config, **kwargs)
            for infer_request in infer_requests
        ]
        if use_tqdm is None:
            use_tqdm = len(infer_requests) > 1
        results = await self.infer_engine._batch_infer_stream(
            tasks, request_config.stream, use_tqdm, None
        )
        flattened_results = []
        for result in results:
            if isinstance(result, list):
                flattened_results.extend(result)
            else:
                flattened_results.append(result)
        return flattened_results

    async def run(
        self,
        infer_request: "RolloutInferRequest",
        request_config: "RequestConfig",
        **kwargs,
    ) -> "RolloutOutput":
        response: "ChatCompletionResponse" = await self.infer_engine.infer_async(
            infer_request, request_config, **kwargs
        )
        response_token_ids = response.choices[0].token_ids
        response_loss_mask = [1] * len(response_token_ids)
        return RolloutOutput(
            response=response,
            messages=infer_request.messages,
            response_token_ids=[response_token_ids],
            response_loss_mask=[response_loss_mask],
            rollout_infos={"num_turns": 1},
        )

    def __getattr__(self, key: str):
        try:
            return object.__getattribute__(self, key)
        except AttributeError:
            pass
        try:
            infer_engine = object.__getattribute__(self, "infer_engine")
            if hasattr(infer_engine, key):
                return getattr(infer_engine, key)
            if hasattr(infer_engine.engine, key):
                return getattr(infer_engine.engine, key)
        except AttributeError:
            raise AttributeError(f"{type(self).__name__} object has no attribute {key}")

    @property
    def engine(self):
        return self.infer_engine

    @property
    def tokenizer(self):
        """Get tokenizer, prioritizing explicitly passed tokenizer over infer_engine's tokenizer."""
        if self._tokenizer is not None:
            return self._tokenizer
        if self.infer_engine is not None:
            return self.infer_engine.tokenizer
        return None


class MultiTurnScheduler(RolloutScheduler, ABC):
    """
    Abstract base class for multi-turn rollout scheduling.

    Provides default implementation for multi-turn conversation management with two customization approaches:

    1. FULL CUSTOMIZATION:
       Override the `run()` method to implement completely custom multi-turn logic.
       - Gives full control over the rollout process
       - Must handle all turn management and termination logic

    2. PARTIAL CUSTOMIZATION:
       Implement the required `step()` method and optionally override `check_finished()`
       - Uses MultiTurnScheduler's run() method infrastructure
       - Only need to implement turn transition logic in step()
       - Optionally customize termination conditions

    Note: You must implement at least one of these approaches in your subclass.

    Options:
        - If each round's response_token_ids are included in the RolloutOutput,
          the Trainer can skip encoding the completion text into token_ids when calculating loss.
          This avoids potential training inconsistencies due to asymmetric encode/decode behavior.
          See: https://github.com/0russwest0/Agent-R1/issues/30#issuecomment-2826155367

        - If both response_token_ids and response_loss_mask are returned in the RolloutOutput,
          you can manually control the loss mask for each token.
          The Trainer will use the provided loss_mask values directly when computing the loss.
          Note: Returning response_loss_mask requires that response_token_ids are also returned,
          as the two must be aligned in length for correct loss computation.

        You can refer to MathTipsScheduler as an example of how to use response_token_ids and response_loss_mask.

    Loss mask configuration:
        During rollout, some parts of the completion (e.g., environment observations embedded in completion)
        may need to be masked out from loss computation.
        There are two supported strategies:

        1. Use the built-in `loss_scale` parameter in ms-swift and do not return response token ids.
        2. Return response_token_ids along with a corresponding response_loss_mask (of equal length) to indicate the loss mask for each token. # noqa
    """

    async def run(
        self,
        infer_request: "RolloutInferRequest",
        request_config: "RequestConfig",
        **kwargs,
    ) -> Union["RolloutOutput", List["RolloutOutput"]]:
        """Execute multi-turn conversation rollout with built-in turn management logic.

        This implements the default multi-turn interaction flow that can be overridden
        to customize conversation handling behavior. The default logic provides:

        1. Automatic conversation turn management and stopping conditions
        2. Seamless message accumulation across multiple turns
        3. Response token tracking and loss mask management
        4. Configurable early stopping mechanisms

        Args:
            infer_request: The initial inference request containing conversation messages
            request_config: Configuration parameters for the inference request
            **kwargs: Additional inference parameters passed to the engine

        Returns:
            RolloutOutput containing the complete conversation history and metadata,
            or a list of outputs for batched requests

        Customization Approaches:
            - Override check_finished() to implement custom stopping criteria
            - Override step() to customize turn-to-turn transition logic
            - Override this entire run() method for completely custom multi-turn behavior

        Important Notes:
            - Method overriding is only supported when using server mode (swift rollout)
              with vllm_use_async_engine=True
            - Custom implementations must maintain async/await compatibility
            - Ensure proper handling of conversation state across turns

        Example:
            class CustomScheduler(MultiTurnScheduler):
                async def run(self, infer_request, request_config, **kwargs):
                    # Implement custom multi-turn conversation logic
                    # Must return RolloutOutput or List[RolloutOutput]
                    ...
        """
        current_request = infer_request
        current_turn = 1
        rollout_infos = {}
        total_response_ids = []
        total_response_loss_mask = []
        total_rollout_logprobs = []
        while True:
            messages = current_request.messages
            if current_turn == 1:
                remove_response(messages)
            response: "ChatCompletionResponse" = await self.infer_engine.infer_async(
                current_request, request_config, **kwargs
            )
            response_choice: "ChatCompletionResponseChoice" = response.choices[0]
            if current_turn > 1 and (not messages[-1]["content"]):
                remove_response(messages)
            completion = response_choice.message.content
            is_continuation = False
            if messages[-1]["role"] == "assistant":
                messages[-1]["content"] += completion
                is_continuation = True
            else:
                messages.append({"role": "assistant", "content": completion})
            should_stop = self.check_finished(
                current_request, response_choice, current_turn
            )
            if self.max_turns:
                should_stop = should_stop or current_turn >= self.max_turns
            if should_stop:
                current_logprobs = self._extract_logprobs_from_choice(response_choice)
                final_token_ids = response_choice.token_ids
                if is_continuation and total_response_ids:
                    total_response_ids[-1].extend(final_token_ids)
                    if total_response_loss_mask:
                        total_response_loss_mask[-1].extend([1] * len(final_token_ids))
                    if total_rollout_logprobs and current_logprobs:
                        total_rollout_logprobs[-1].extend(current_logprobs)
                elif not total_response_ids:
                    if final_token_ids:
                        total_response_ids = [list(final_token_ids)]
                        total_response_loss_mask = [[1] * len(final_token_ids)]
                    if current_logprobs:
                        total_rollout_logprobs = [current_logprobs]
                final_rollout_logprobs = total_rollout_logprobs
                if total_rollout_logprobs:
                    total_logprob_count = sum(
                        (len(turn_lps) for turn_lps in total_rollout_logprobs)
                    )
                    if total_response_loss_mask:
                        total_loss_mask_1_count = sum(
                            (sum(mask) for mask in total_response_loss_mask)
                        )
                        if total_loss_mask_1_count != total_logprob_count:
                            final_rollout_logprobs = []
                    elif total_response_ids:
                        total_response_id_count = sum(
                            (len(turn_ids) for turn_ids in total_response_ids)
                        )
                        if total_response_id_count != total_logprob_count:
                            final_rollout_logprobs = []
                    else:
                        final_rollout_logprobs = []
                return RolloutOutput(
                    response=response,
                    messages=messages,
                    response_token_ids=total_response_ids,
                    response_loss_mask=total_response_loss_mask,
                    rollout_infos={**rollout_infos, "num_turns": current_turn},
                    rollout_logprobs=final_rollout_logprobs,
                )
            ret = self.step(current_request, response_choice, current_turn)
            current_request: "RolloutInferRequest" = ret["infer_request"]
            return_token_id = False
            if "response_token_ids" in ret:
                if is_continuation and total_response_ids:
                    total_response_ids[-1].extend(ret["response_token_ids"])
                else:
                    total_response_ids.append(ret["response_token_ids"])
                return_token_id = True
            if "response_loss_mask" in ret:
                assert (
                    return_token_id
                ), "You must return response_token_ids if you want to return response_loss_mask"
                assert len(ret["response_loss_mask"]) == len(
                    ret["response_token_ids"]
                ), "response_loss_mask must have the same length as response_token_ids"
                if is_continuation and total_response_loss_mask:
                    total_response_loss_mask[-1].extend(ret["response_loss_mask"])
                else:
                    total_response_loss_mask.append(ret["response_loss_mask"])
            if "rollout_infos" in ret:
                rollout_infos.update(ret["rollout_infos"])
            if "rollout_logprobs" in ret and ret["rollout_logprobs"]:
                current_logprobs = ret["rollout_logprobs"]
            else:
                current_logprobs = self._extract_logprobs_from_choice(response_choice)
            if current_logprobs:
                if is_continuation and total_rollout_logprobs:
                    total_rollout_logprobs[-1].extend(current_logprobs)
                else:
                    total_rollout_logprobs.append(current_logprobs)
            current_turn += 1

    def step(
        self,
        infer_request: "RolloutInferRequest",
        response_choice: "ChatCompletionResponseChoice",
        current_turn: int,
    ) -> Dict:
        """
        Handles transition between conversation turns.

        Args:
            infer_request: Current inference request
            response_choice: Response from current turn
            current_turn: Current turn number

        Returns:
            Dict[str, Any]: A dictionary containing inference results with the following structure:
                - infer_request (required): Main inference request object
                - response_token_ids (Optional[List[int]]): Token IDs of response for current rollout turn
                - response_loss_mask (Optional[List[int]]): Loss mask for response tokens (same length as response_token_ids) # noqa
                - rollout_logprobs (Optional[List[float]]): Log probabilities for response tokens.
                    If not provided, will be extracted from response_choice.logprobs as fallback.
                    Useful when modifying response content (e.g., adding prompts) to avoid logprob misalignment.
                - rollout_infos (Optional[Dict[str, Any]]): Additional metadata (must be serializable)

        """
        raise NotImplementedError(
            "Please implement the `step` method in your MultiTurnScheduler subclass, or override the `run` method."
        )

    def check_finished(
        self,
        infer_request: "RolloutInferRequest",
        response_choice: "ChatCompletionResponseChoice",
        current_turn: int,
    ) -> bool:
        """
        Default termination logic for checking if a multi-turn rollout should end.

        This method is invoked by:
        - The base class MultiTurnScheduler.run() method, OR
        - Custom run() methods when explicitly called

        Note: This is the default implementation that can be overridden by subclasses for custom termination logic.

        Termination Conditions:
        1. When response hits length limit (finish_reason == 'length')
        2. When conversation reaches max_turns (if max_turns is set)

        Args:
            infer_request: The inference request object
            response_choice: Contains generation results including finish_reason
            current_turn: Current conversation turn count

        Returns:
            bool: True to terminate conversation, False to continue
        """
        if response_choice.finish_reason == "length":
            return True
        if self.max_turns and current_turn >= self.max_turns:
            return True
        return False

    @staticmethod
    def _extract_logprobs_from_choice(
        response_choice: "ChatCompletionResponseChoice",
    ) -> List[float]:
        """Extract logprobs list from response choice for rollout importance sampling.

        Args:
            response_choice: The response choice containing logprobs

        Returns:
            List of logprob values, or empty list if not available
        """
        if response_choice.logprobs is None:
            return []
        if "content" in response_choice.logprobs:
            return [item["logprob"] for item in response_choice.logprobs["content"]]
        return []
