from __future__ import annotations

from physionet_sleep.recipe.algo_def import AlgoDef


class RecipeBranch:
    def __init__(
        self,
        name: str,
        steps: list[AlgoDef] | None = None,
        children: list["RecipeBranch"] | None = None,
    ) -> None:
        steps = steps if steps is not None else []
        children = children if children is not None else []
        assert isinstance(name, str)
        assert isinstance(steps, list)
        assert isinstance(children, list) and all(isinstance(c, RecipeBranch) for c in children)
        self.name = name
        self.steps = steps
        self.children = children

    def is_leaf(self) -> bool:
        return len(self.children) == 0
