from collections.abc import Sequence

NGRAM_ORDER = 4
NGRAM_KEY_LENGTH = NGRAM_ORDER - 1


class PromptNgram:
    """Suffix map of one request's prompt. Generated tokens are not inserted."""

    def __init__(self, prompt: Sequence[int]) -> None:
        self._next: dict[tuple[int, int, int], int] = {}
        if len(prompt) < NGRAM_ORDER:
            return
        for index in range(len(prompt) - NGRAM_ORDER + 1):
            key = (
                int(prompt[index]),
                int(prompt[index + 1]),
                int(prompt[index + 2]),
            )
            self._next[key] = int(prompt[index + NGRAM_KEY_LENGTH])

    def propose(self, context: Sequence[int], limit: int) -> tuple[int, ...]:
        if limit < 1 or len(context) < NGRAM_KEY_LENGTH:
            return ()
        window = (
            int(context[-3]),
            int(context[-2]),
            int(context[-1]),
        )
        drafts: list[int] = []
        for _ in range(limit):
            token_id = self._next.get(window)
            if token_id is None:
                break
            drafts.append(token_id)
            window = (window[1], window[2], token_id)
        return tuple(drafts)


def verification_tokens(pending_token_id: int, drafts: Sequence[int]) -> list[int]:
    if not drafts:
        raise ValueError("Verification needs at least one draft.")
    return [int(pending_token_id), *[int(token_id) for token_id in drafts[:-1]]]


def accept_greedy(greedy: Sequence[int], drafts: Sequence[int]) -> tuple[int, ...]:
    if len(greedy) != len(drafts) or not drafts:
        raise ValueError("Each draft needs one greedy token.")
    accepted: list[int] = []
    for token_id, draft in zip(greedy, drafts, strict=True):
        if int(token_id) != int(draft):
            accepted.append(int(token_id))
            return tuple(accepted)
        accepted.append(int(draft))
    return tuple(accepted)
