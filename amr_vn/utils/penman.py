import re
from typing import Iterable, Tuple, List

SNT_PREFIX = "#::snt"


def parse_amr_blocks(lines: Iterable[str]) -> Iterable[Tuple[str, str]]:
    """
    Parse a corpus where each block starts with "#::snt <sentence>" then a PENMAN graph.
    Blocks are separated by blank lines or another "#::snt".
    Yields (sentence, amr_string).
    """
    sent = None
    amr_lines: List[str] = []

    def flush():
        nonlocal sent, amr_lines
        if sent is not None and amr_lines:
            yield_sent, yield_amr = (
                sent.strip(),
                "\n".join([l.rstrip() for l in amr_lines]).strip(),
            )
            # reset
            sent, amr_lines = None, []
            return (yield_sent, yield_amr)
        return None

    for raw in lines:
        line = raw.rstrip("\n")
        if line.startswith(SNT_PREFIX):
            # commit previous
            item = flush()
            if item:
                yield item
            sent = line[len(SNT_PREFIX) :].strip()
            amr_lines = []
            continue
        if line.strip() == "" and sent is not None and amr_lines:
            item = flush()
            if item:
                yield item
            continue
        if sent is not None:
            amr_lines.append(line)

    # final
    item = flush()
    if item:
        yield item


def basic_penman_format(text: str) -> str:
    """
    Minimal formatter. Ensures parentheses are balanced and strips trailing spaces.
    Does not change token order.
    """
    t = "\n".join(l.rstrip() for l in text.splitlines()).strip()
    # naive balance check; if unbalanced, return as-is
    if t.count("(") != t.count(")"):
        return t
    # simple indentation based on paren depth
    out = []
    depth = 0
    for ch in t:
        if ch == ")":
            depth = max(0, depth - 1)
        if ch == "(":
            out.append("\n" + "    " * depth + ch)
            depth += 1
        elif ch == ")":
            out.append(ch)
        else:
            out.append(ch)
    formatted = "".join(out).strip()
    # compact multiple blank lines
    formatted = re.sub(r"\n\s*\n+", "\n", formatted)
    return formatted if formatted else t


def format_amr_penman(raw_amr: str) -> str:
    lines = []
    indent = "   "
    tokens = raw_amr.strip().replace("(", " ( ").replace(")", " ) ").split()
    stack = []
    current_line = ""

    for token in tokens:
        if token == "(":
            if current_line:
                lines.append(current_line)
            current_line = indent * len(stack)
            stack.append("(")
        elif token == ")":
            if current_line:
                lines.append(current_line)
            if stack:
                stack.pop()
            current_line = ""
        else:
            if not current_line:
                current_line = indent * len(stack)
            current_line += token + " "

    if current_line:
        lines.append(current_line.strip())

    return "\n".join(lines)
