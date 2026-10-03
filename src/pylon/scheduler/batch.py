PREFILL_MAX_WAIT_SECONDS = 0.1


def pack_tick(active, prefill_chunk_size: int, now: float):
    decoding = [item for item in active if item.pending_token_id is not None]
    selected = list(decoding)
    chunks = [[item.pending_token_id] for item in decoding]
    budget = max(prefill_chunk_size, len(decoding) + 1) - len(decoding)
    pending = [item for item in active if item.pending_token_id is None]
    while pending and budget > 0:
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
        chosen = (
            min(aged, key=lambda item: item.prefill_wait_started)
            if aged
            else (fitting or pending)[0]
        )
        pending.remove(chosen)
        count = min(
            budget, len(chosen.request.input_ids) - chosen.prompt_offset
        )
        selected.append(chosen)
        chunks.append(
            chosen.request.input_ids[
                chosen.prompt_offset : chosen.prompt_offset + count
            ]
        )
        budget -= count
    return selected, chunks
