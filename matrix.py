import colorsys  # Polygon regions.
from PIL import Image, ImageChops
from pprint import pprint
import numpy as np
import PIL
import torch

SPLROW = ";"
SPLCOL = ","
KEYROW = "ADDROW"
KEYCOL = "ADDCOL"
KEYBASE = "ADDBASE"
KEYCOMM = "ADDCOMM"
KEYBRK = "BREAK"
NLN = "\n"
DKEYINOUT = { # Out/in, horizontal/vertical or row/col first.
("out",False): KEYROW,
("in",False): KEYCOL,
("out",True): KEYCOL,
("in",True): KEYROW,
}
fidentity = lambda x: x
ffloatd = lambda c: (lambda x: floatdef(x,c))
fspace = lambda x: " {} ".format(x)
fcountbrk = lambda x: x.count(KEYBRK)
fint = lambda x: int(x)

def floatdef(x, vdef):
    """Attempt conversion to float, use default value on error.    
    Mainly for empty ratios, double commas.
    """
    try:
        return float(x)
    except ValueError:
        print("'{}' is not a number, converted to {}".format(x,vdef))
        return vdef

class Region():
    """Specific Region used to split a layer to single prompts."""
    def __init__(self, st, ed, base, breaks):
        """Range with start and end values, base weight and breaks count for context splitting."""
        self.start = st # Range for the cell (cols only).
        self.end = ed
        self.base = base # How much of the base prompt is applied (difference).
        self.breaks = breaks # How many unrelated breaks the prompt contains.

class Row():
    """Row containing cell refs and its own ratio range."""
    def __init__(self, st, ed, cols):
        """Range with start and end values, base weight and breaks count for context splitting."""
        self.start = st # Range for the row.
        self.end = ed
        self.cols = cols # List of cells.
        
def is_l2(l):
    return isinstance(l[0],list) 

def l2_count(l):
    cnt = 0
    for row in l:
        cnt + cnt + len(row)
    return cnt

def list_percentify(l):
    """
    Convert each row in L2 to relative part of 100%. 
    Also works on L1, applying once globally.
    """
    lret = []
    if is_l2(l):
        for row in l:
            # row2 = [float(v) for v in row]
            row2 = [v / sum(row) for v in row]
            lret.append(row2)
    else:
        row = l[:]
        # row2 = [float(v) for v in row]
        row2 = [v / sum(row) for v in row]
        lret = row2
    return lret

def list_cumsum(l):
    """
    Apply cumsum to L2 per row, ie newl[n] = l[0:n].sum .
    Works with L1.
    Actually edits l inplace, idc.
    """
    lret = []
    if is_l2(l):
        for row in l:
            for (i,v) in enumerate(row):
                if i > 0:
                    row[i] = v + row[i - 1]
            lret.append(row)
    else:
        row = l[:]
        for (i,v) in enumerate(row):
            if i > 0:
                row[i] = v + row[i - 1]
        lret = row
    return lret

def list_rangify(l):
    """
    Merge every 2 elems in L2 to a range, starting from 0.  
    """
    lret = []
    if is_l2(l):
        for row in l:
            row2 = [0] + row
            row3 = []
            for i in range(len(row2) - 1):
                row3.append([row2[i],row2[i + 1]]) 
            lret.append(row3)
    else:
        row2 = [0] + l
        row3 = []
        for i in range(len(row2) - 1):
            row3.append([row2[i],row2[i + 1]]) 
        lret = row3
    return lret

def ratiosdealer(split_ratio2,split_ratio2r):
    split_ratio2 = list_percentify(split_ratio2)
    split_ratio2 = list_cumsum(split_ratio2)
    split_ratio2 = list_rangify(split_ratio2)
    split_ratio2r = list_percentify(split_ratio2r)
    split_ratio2r = list_cumsum(split_ratio2r)
    split_ratio2r = list_rangify(split_ratio2r)
    return split_ratio2,split_ratio2r

def round_dim(x,y):
    """Return division of two numbers, rounding 0.5 up.    
    Seems that dimensions which are exactly 0.5 are rounded up - see 680x488, second iter.
    A simple mod check should get the job done.
    If not, can always brute force the divisor with +-1 on each of h/w.
    """
    return x // y + (x % y >= y // 2)       

def _is_seq(x):
    """True for list/tuple (but not strings)."""
    return isinstance(x, (list, tuple))


def _keyconverter_one(prompt: str, split_ratio: str, usebase: bool) -> str:
    """Convert BREAK tokens in a *single* prompt into ADDROW/ADDCOL/ADDBASE anchors."""
    if SPLROW not in split_ratio:  # Commas only - interpret as 1d.
        split_ratio2 = split_l2(split_ratio, SPLROW, SPLCOL, map_function=ffloatd(1))
        split_ratio2r = [1]
    else:
        (split_ratio2r, split_ratio2) = split_l2(
            split_ratio, SPLROW, SPLCOL, indsingles=True, map_function=ffloatd(1)
        )
    (split_ratio2, split_ratio2r) = ratiosdealer(split_ratio2, split_ratio2r)

    txtkey = fspace(DKEYINOUT[("in", False)]) + NLN
    lkeys = [txtkey.join([""] * len(cell)) for cell in split_ratio2]
    txtkey = fspace(DKEYINOUT[("out", False)]) + NLN
    template = txtkey.join(lkeys)
    if usebase:
        template = fspace(KEYBASE) + NLN + template
    changer = template.split(NLN)
    changer = [l.strip() for l in changer]
    keychanger = changer[:-1]

    out_prompt = prompt
    for change in keychanger:
        if change == KEYBASE and KEYBASE in out_prompt:
            continue
        out_prompt = out_prompt.replace(KEYBRK, change, 1)
    return out_prompt


def keyconverter(self, split_ratio, usebase):
    """Convert BREAKS to ADDCOMM/ADDBASE/ADDCOL/ADDROW.

    Backwards compatible:
      - single: self.prompt is str, split_ratio is str  -> self.prompt becomes str
      - batch : self.prompt is list[str] (or str) and split_ratio is list[str]
               -> self.prompt becomes list[str]
    """
    # Batch mode: split_ratio is a list/tuple.
    if _is_seq(split_ratio):
        if _is_seq(self.prompt):
            prompts = list(self.prompt)
        else:
            prompts = [self.prompt] * len(split_ratio)

        if len(prompts) != len(split_ratio):
            raise ValueError(
                f"Batch length mismatch: len(prompt)={len(prompts)} vs len(split_ratio)={len(split_ratio)}"
            )

        out_prompts = []
        for p, sr in zip(prompts, split_ratio):
            out_prompts.append(_keyconverter_one(str(p), str(sr), bool(usebase)))
        self.prompt = out_prompts
        return out_prompts

    # Single mode.
    self.prompt = _keyconverter_one(str(self.prompt), str(split_ratio), bool(usebase))
    return self.prompt
def split_l2(s, key_row, key_col, indsingles = False, map_function = fidentity, split_struct = None):
    lret = []
    if split_struct is None:
        lrows = s.split(key_row)
        lrows = [row.split(key_col) for row in lrows]
        # print(lrows)
        for r in lrows:
            cell = [map_function(x) for x in r]
            lret.append(cell)
        if indsingles:
            lsingles = [row[0] for row in lret]
            lcells = [row[1:] if len(row) > 1 else row for row in lret]
            lret = (lsingles,lcells)
    else:
        lrows = str(s).split(key_row)
        r = 0
        lcells = []
        lsingles = []
        vlast = 1
        for row in lrows:
            row2 = row.split(key_col)
            row2 = [map_function(x) for x in row2]
            vlast = row2[-1]
            indstop = False
            while not indstop:
                if (r >= len(split_struct) # Too many cell values, ignore.
                or (len(row2) == 0 and len(split_struct) > 0)): # Cell exhausted.
                    indstop = True
                if not indstop:
                    if indsingles: # Singles split.
                        lsingles.append(row2[0]) # Row ratio.
                        if len(row2) > 1:
                            row2 = row2[1:]
                    if len(split_struct[r]) >= len(row2): # Repeat last value.
                        indstop = True
                        broadrow = row2 + [row2[-1]] * (len(split_struct[r]) - len(row2))
                        r = r + 1
                        lcells.append(broadrow)
                    else: # Overfilled this row, cut and move to next.
                        broadrow = row2[:len(split_struct[r])]
                        row2 = row2[len(split_struct[r]):]
                        r = r + 1
                        lcells.append(broadrow)
        # If not enough new rows, repeat the last one for entire base, preserving structure.
        cur = len(lcells)
        while cur < len(split_struct):
            lcells.append([vlast] * len(split_struct[cur]))
            cur = cur + 1
        lret = lcells
        if indsingles:
            lsingles = lsingles + [lsingles[-1]] * (len(split_struct) - len(lsingles))
            lret = (lsingles,lcells)
    return lret
    
def _matrixdealer_one(prompt: str, split_ratio: str, baseratio, usebase: bool):
    """Build Region/Row structures for a single (prompt, split_ratio, baseratio)."""
    # The original code expects prompt already has ADDROW/ADDCOL anchors.
    _prompt = prompt
    if KEYBASE in _prompt:
        _prompt = _prompt.split(KEYBASE, 1)[1]

    if not (KEYCOL in _prompt.upper() or KEYROW in _prompt.upper()):
        raise ValueError(
            "matrixdealer expects prompt to contain ADDROW/ADDCOL anchors. " 
            "Did you call keyconverter() first?"
        )

    # Prompt anchors, count breaks between special keywords.
    lbreaks = split_l2(_prompt, KEYROW, KEYCOL, map_function=fcountbrk)

    if (SPLROW not in split_ratio and (KEYROW in _prompt.upper()) != (KEYCOL in _prompt.upper())):
        # 1d integrated into 2d when using only one axis of anchors and commas in ratio.
        split_ratio = "1" + SPLCOL + split_ratio
        (split_ratio2r, split_ratio2) = split_l2(
            split_ratio, SPLROW, SPLCOL,
            indsingles=True, map_function=ffloatd(1), split_struct=lbreaks
        )
    else:
        (split_ratio2r, split_ratio2) = split_l2(
            split_ratio, SPLROW, SPLCOL,
            indsingles=True, map_function=ffloatd(1), split_struct=lbreaks
        )

    # Per-cell base weights (can be scalar/float or structured string like "0.2,0.3;...").
    baseratio2 = split_l2(baseratio, SPLROW, SPLCOL, map_function=ffloatd(0), split_struct=lbreaks)

    (split_ratio_ranges, split_row_ranges) = ratiosdealer(split_ratio2, split_ratio2r)

    # Merge various L2s to cells and rows.
    drows = []
    for r, _ in enumerate(lbreaks):
        dcells = []
        for c, _ in enumerate(lbreaks[r]):
            d = Region(
                split_ratio_ranges[r][c][0],
                split_ratio_ranges[r][c][1],
                baseratio2[r][c],
                lbreaks[r][c],
            )
            dcells.append(d)
        drow = Row(split_row_ranges[r][0], split_row_ranges[r][1], dcells)
        drows.append(drow)

    return drows, baseratio2


def matrixdealer(self, split_ratio, baseratio):
    """Parse anchors in self.prompt and build Region/Row split structures.

    Backwards compatible:
      - single: self.prompt is str, split_ratio is str, baseratio is float/str
               -> self.split_ratio becomes List[Row], self.baseratio becomes L2 list
      - batch : self.prompt is list[str], split_ratio is list[str], baseratio is list[float/str]
               -> self.split_ratio becomes list[List[Row]], self.baseratio becomes list[L2]
    """
    # Batch mode.
    if _is_seq(split_ratio) or _is_seq(baseratio) or _is_seq(self.prompt):
        # Normalize to lists.
        prompts = list(self.prompt) if _is_seq(self.prompt) else [self.prompt]
        srs = list(split_ratio) if _is_seq(split_ratio) else [split_ratio] * len(prompts)
        brs = list(baseratio) if _is_seq(baseratio) else [baseratio] * len(prompts)

        n = max(len(prompts), len(srs), len(brs))
        if len(prompts) != n:
            prompts = prompts * n
        if len(srs) != n:
            if len(srs) == 1:
                srs = srs * n
        if len(brs) != n:
            if len(brs) == 1:
                brs = brs * n

        if not (len(prompts) == len(srs) == len(brs)):
            raise ValueError(
                f"Batch length mismatch: len(prompt)={len(prompts)}, len(split_ratio)={len(srs)}, len(baseratio)={len(brs)}"
            )

        all_drows = []
        all_baseratios = []
        for p, sr, br in zip(prompts, srs, brs):
            drows, br2 = _matrixdealer_one(str(p), str(sr), br, bool(getattr(self, 'usebase', False)))
            all_drows.append(drows)
            all_baseratios.append(br2)

        self.split_ratio = all_drows
        self.baseratio = all_baseratios
        return all_drows, all_baseratios

    # Single mode.
    drows, br2 = _matrixdealer_one(str(self.prompt), str(split_ratio), baseratio, bool(getattr(self, 'usebase', False)))
    self.split_ratio = drows
    self.baseratio = br2
    return drows, br2
