"""Print commit-pinned upstream setup commands; never execute them or download weights."""

from vsr_evalkit.backends import source_pins

for key, directory in [
    ("dover", "DOVER"),
    ("fastvqa", "FAST-VQA-and-FasterVQA"),
    ("raft", "RAFT"),
]:
    source = source_pins()[key]
    print(f"git clone --no-checkout {source['url']}.git third_party/{directory}")
    print(f"git -C third_party/{directory} checkout --detach {source['commit']}")
print("# Code only. Supply weight paths manually; see vsr-eval weights info.")
