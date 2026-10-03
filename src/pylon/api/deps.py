from fastapi import HTTPException, Request

from pylon.frontend import TextGenerator


def get_generator(request: Request) -> TextGenerator:
    generator: TextGenerator | None = getattr(request.app.state, "generator", None)
    if generator is None:
        raise HTTPException(status_code=503, detail="The pylon generator is not ready.")
    return generator
