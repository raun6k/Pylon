PREFILL_MAX_WAIT_SECONDS = 0.1


def pack_tick(active, prefill_chunk_size: int, now: float):
    decoding = [item for item in active if item.pending_token_id is not None]
    selected = list(decoding)
    chunks = [[item.pending_token_id] for item in decoding]
    budget = max(prefill_chunk_size, len(decoding) + 1) - len(decoding)
    pending = [item for item in active if item.pending_token_id is None]
    while pending and budget > 0:
        chosen = _choose_prefill(pending, budget, now)
        pending.remove(chosen)
        count = min(budget, len(chosen.request.input_ids) - chosen.prompt_offset)
        selected.append(chosen)
        chunks.append(
            chosen.request.input_ids[
                chosen.prompt_offset : chosen.prompt_offset + count
            ]
        )
        budget -= count
    return selected, chunks


def one_prefill_chunk(active, prefill_chunk_size: int, now: float):
    pending = [
        item
        for item in active
        if item.pending_token_id is None
        and item.prompt_offset < len(item.request.input_ids)
    ]
    if not pending or prefill_chunk_size < 1:
        return None
    chosen = _choose_prefill(pending, prefill_chunk_size, now)
    count = min(
        prefill_chunk_size, len(chosen.request.input_ids) - chosen.prompt_offset
    )
    if count < 1:
        return None
    tokens = chosen.request.input_ids[
        chosen.prompt_offset : chosen.prompt_offset + count
    ]
    return chosen, tokens


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
