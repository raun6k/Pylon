from collections.abc import Callable

PREFILL_MAX_WAIT_SECONDS = 0.1


def prefill_chunks(
    active,
    token_budget: int,
    now: float,
    prepare: Callable[[object], None] | None = None,
):
    """Pick prompt chunks from every prefilling request, up to `token_budget` tokens.

    A request that has waited 100 ms goes first; otherwise a prompt that finishes inside
    the remaining budget goes first. `prepare` runs before a request's chunk is sized, so
    the engine can attach cached prefix blocks and shorten the work.
    """
    pending = [
        item
        for item in active
        if item.pending_token_id is None
        and item.prompt_offset < len(item.request.input_ids)
    ]
    chosen = []
    budget = token_budget
    while pending and budget > 0:
        item = _choose_prefill(pending, budget, now)
        pending.remove(item)
        if prepare is not None:
            prepare(item)
        count = min(budget, len(item.request.input_ids) - item.prompt_offset)
        if count < 1:
            continue
        chosen.append(
            (item, item.request.input_ids[item.prompt_offset : item.prompt_offset + count])
        )
        budget -= count
    return chosen


def _choose_prefill(pending, budget: int, now: float):
    aged = [
        item
        for item in pending
        if now - item.prefill_wait_started >= PREFILL_MAX_WAIT_SECONDS
    ]
    fitting = [
        item
        for item in pending
        if len(item.request.input_ids) - item.prompt_offset <= budget
    ]
    if aged:
        return min(aged, key=lambda item: item.prefill_wait_started)
    return (fitting or pending)[0]
