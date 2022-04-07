
from pathlib import Path

oppositors = (
    'abs',
    'modular',
    'quasi',
    'quasi_reflect',
    'over',
    'integers_by_order'
)


result = []

for opp in oppositors:
    result.append(
        f"""
#### `{opp}` oppositor

[Code](tests/op_{opp}.py)
![](tests/output/{opp}.png)
        """
    )

Path('res.txt').write_text(
    '\n\n'.join(result)
)

