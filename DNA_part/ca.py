import random
import csv
import os
import sys
import math
import numpy as np
from collections import Counter, defaultdict
from contextlib import redirect_stdout, redirect_stderr
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import colorsys
from matplotlib.patches import Patch

from simulation import NUCLEOTIDES, COMP, extract_kmers


# ================================================================
# Constants
# ================================================================

CA_MIN_DS_FOR_SPLIT = 6
CA_MAX_PIECE_LEN    = 40
CA_MAX_INIT_LEN     = 12
CA_MU               =  0.0002
SPLIT_COOLDOWN      = 5      
COMPLETION_ERROR    = 0.002 
KMER_SIZES          = [4, 6, 8, 10, 12]


# ================================================================
# Resource field parameters
# ================================================================
RESOURCE_MAX                = 10
RESOURCE_INIT               = 5
RESOURCE_COST_PER_COL       = 1
RESOURCE_SPAWN_COST_PER_COL = 1
RESOURCE_INTERACT_MIN       = 1
RESOURCE_SPAWN_THRESHOLD    = 1

# ================================================================
# Helpers
# ================================================================
# Columns are (top, bottom) tuples.

def col_is_paired(t, b):
    return t is not None and b is not None and COMP[t] == b

def col_has_mismatch(t, b):
    return t is not None and b is not None and COMP[t] != b


# ================================================================
# Piece representation
# ================================================================
# A piece is a list of (top, bottom) column tuples.
def piece_is_complete_ds(cols):
    if not cols:
        return False
    for t, b in cols:
        if not col_is_paired(t, b):
            return False
    return True

def piece_n_paired(cols):
    n = 0
    for t, b in cols:
        if col_is_paired(t, b):
            n += 1
    return n

def piece_n_gaps(cols):
    n = 0
    for t, b in cols:
        if not col_is_paired(t, b):
            n += 1
    return n

def piece_has_any_pairing(cols):
    for t, b in cols:
        if col_is_paired(t, b):
            return True
    return False

def piece_top_seq(cols):
    runs = []
    current = []
    for t, _ in cols:
        if t is not None:
            current.append(t)
        else:
            if current:
                runs.append("".join(current))
                current = []
    if current:
        runs.append("".join(current))
    return runs

def piece_bottom_seq(cols):
    runs = []
    current = []
    for _, b in reversed(cols):
        if b is not None:
            current.append(b)
        else:
            if current:
                runs.append("".join(current))
                current = []
    if current:
        runs.append("".join(current))
    return runs

def piece_all_sequences(cols):
    return piece_top_seq(cols) + piece_bottom_seq(cols)

def piece_primary_sequence(cols):
    runs = piece_top_seq(cols) + piece_bottom_seq(cols)
    if not runs:
        return ""
    return max(runs, key=len)

# ================================================================
# Stability scoring
# ================================================================
# Stability determines which piece survives a competition.

def stability_score(cols):
    if not cols:
        return (0, 0, 0, 0)
    has_ds = 1 if piece_has_any_pairing(cols) else 0
    n_paired = piece_n_paired(cols)
    length = len(cols)

    if has_ds:
        n_gaps = piece_n_gaps(cols)
        return (1, -n_gaps, n_paired, length)
    else:
        # Pure ss: gap count is meaningless (every column is "unpaired").
        # Stability depends only on length.
        return (0, 0, 0, length)


# ================================================================
# Rand piece generation
# ================================================================
def random_piece(max_len=CA_MAX_INIT_LEN, ss_only=False):
    length = random.choices(
        population=list(range(1, max_len + 1)),
        weights=[12, 11, 10, 9,8, 7, 6, 5, 4, 3, 2, 1][:max_len],
    )[0]

    if ss_only:
        topo = random.choice(["ss_top", "ss_bot"])
    else:
        topo = random.choices(
            ["ss_top", "ss_bot", "partial_ds", "full_ds"],
            weights=[20, 20, 25, 35],
        )[0]

    cols = []

    if topo == "ss_top":
        for _ in range(length):
            nuc = random.choice(NUCLEOTIDES)
            cols.append((nuc, None))

    elif topo == "ss_bot":
        for _ in range(length):
            nuc = random.choice(NUCLEOTIDES)
            cols.append((None, nuc))

    elif topo == "full_ds":
        for _ in range(length):
            nuc = random.choice(NUCLEOTIDES)
            cols.append((nuc, COMP[nuc]))

    else:  # partial_ds
        if length == 1:
            nuc = random.choice(NUCLEOTIDES)
            cols.append((nuc, COMP[nuc]))
        else:
            core_len = random.randint(1, length - 1)
            overhang_len = length - core_len

            left_len = random.randint(0, overhang_len)
            right_len = overhang_len - left_len

            left_on_top = random.choice([True, False])
            right_on_top = random.choice([True, False])

            for _ in range(left_len):
                nuc = random.choice(NUCLEOTIDES)
                if left_on_top:
                    cols.append((nuc, None))
                else:
                    cols.append((None, nuc))

            for _ in range(core_len):
                nuc = random.choice(NUCLEOTIDES)
                cols.append((nuc, COMP[nuc]))

            for _ in range(right_len):
                nuc = random.choice(NUCLEOTIDES)
                if right_on_top:
                    cols.append((nuc, None))
                else:
                    cols.append((None, nuc))

    return cols

# ================================================================
# Mutation
# ================================================================
def mutate_piece(cols, mu=CA_MU):
    new_cols = []
    for t, b in cols:
        if t is not None and random.random() < mu:
            t = random.choice([x for x in NUCLEOTIDES if x != t])
        if b is not None and random.random() < mu:
            b = random.choice([x for x in NUCLEOTIDES if x != b])
        new_cols.append((t, b))
    return new_cols


# ================================================================
# Mismatch resolution
# ================================================================
def resolve_mismatches(cols):
    new_cols = []
    for t, b in cols:
        if col_has_mismatch(t, b):
            # Randomly keep one strand, eject the other
            if random.random() < 0.5:
                new_cols.append((t, None))
            else:
                new_cols.append((None, b))
        else:
            new_cols.append((t, b))
    return new_cols


# ================================================================
# Complementary association
# ================================================================
def _ca_score_cols(cols):
    s = 0
    for t, b in cols:
        if t is not None:
            s += 1
        if b is not None:
            s += 1
        if col_is_paired(t, b):
            s += 1
    return s


def _ca_has_internal_holes(cols):
    paired_idx = [i for i, (t, b) in enumerate(cols) if col_is_paired(t, b)]
    if len(paired_idx) < 2:
        return False
    lo, hi = paired_idx[0], paired_idx[-1]
    for i in range(lo, hi + 1):
        if not col_is_paired(cols[i][0], cols[i][1]):
            return True
    return False


def _ca_combine_col(c1, c2):
    t1, b1 = c1
    t2, b2 = c2

    if col_is_paired(t1, b1):
        return None
    if col_is_paired(t2, b2):
        return None

    # Reject same-strand overlap
    if t1 is not None and t2 is not None:
        return None
    if b1 is not None and b2 is not None:
        return None
        
    t = t1 if t1 is not None else t2
    b = b1 if b1 is not None else b2

    if t is not None and b is not None and COMP[t] != b:
        return None

    return (t, b)




def try_merge(a_cols, b_cols):
    if not a_cols or not b_cols:
        return None

    before_score = _ca_score_cols(a_cols) + _ca_score_cols(b_cols)

    len_a = len(a_cols)
    len_b = len(b_cols)

    # offset = start index of b relative to a
    # offset = 0 means b[0] aligns with a[0]
    # offset = 2 means b[0] aligns with a[2]
    # offset = -1 means b starts one column before a
    for offset in range(-len_b + 1, len_a):
        start = min(0, offset)
        end = max(len_a, offset + len_b)
        result_len = end - start
        if result_len > CA_MAX_PIECE_LEN:
            continue

        merged = []
        ok = True

        for pos in range(start, end):
            a_idx = pos
            b_idx = pos - offset

            a_here = 0 <= a_idx < len_a
            b_here = 0 <= b_idx < len_b

            if a_here and b_here:
                combined = _ca_combine_col(a_cols[a_idx], b_cols[b_idx])
                if combined is None:
                    ok = False
                    break
                merged.append(combined)
            elif a_here:
                merged.append(a_cols[a_idx])
            elif b_here:
                merged.append(b_cols[b_idx])
            else:
                raise ValueError("Impossible alignment state")

        if not ok:
            continue

        # if _ca_has_internal_holes(merged):
        #     continue

        after_score = _ca_score_cols(merged)
        if after_score <= before_score:
            continue

        return merged

    return None


# ================================================================
# Pairing
# ================================================================
def try_pair_absorb(target_cols, donor_cols):
    for d_t, d_b in donor_cols:
        if d_t is not None and d_b is not None:
            continue

        # This allows also that single strands will cede single nucleotides to the target
        if d_t is not None:
            nuc = d_t
        elif d_b is not None:
            nuc = d_b
        else:
            continue

        for i, (t, b) in enumerate(target_cols):
            if t is not None and b is None and COMP[t] == nuc:
                new_cols = list(target_cols)
                new_cols[i] = (t, nuc)
                return new_cols, True

            if b is not None and t is None and COMP[nuc] == b:
                new_cols = list(target_cols)
                new_cols[i] = (nuc, b)
                return new_cols, True

    return target_cols, False

# ================================================================
# Elongation 
# ================================================================
def try_elongate(target_cols, donor_cols):
    if not target_cols or not donor_cols:
        return target_cols, False

    if len(target_cols) + len(donor_cols) > CA_MAX_PIECE_LEN:
        return target_cols, False

    donor_has_top = any(t is not None for t, _ in donor_cols)
    donor_has_bot = any(b is not None for _, b in donor_cols)

    if donor_has_top and donor_has_bot:
        return target_cols, False

    target_has_top = any(t is not None for t, _ in target_cols)
    target_has_bot = any(b is not None for _, b in target_cols)

    # Case A  
    if not (target_has_top and target_has_bot):
        if stability_score(target_cols) >= stability_score(donor_cols):
            reference_is_top = target_has_top
            first, second = target_cols, donor_cols
        else:
            reference_is_top = donor_has_top
            first, second = donor_cols, target_cols

        def rewrite(cols, use_top):
            out = []
            for t, b in cols:
                nuc = t if t is not None else b
                if nuc is None:
                    raise ValueError("Empty column in ss elongation")
                out.append((nuc, None) if use_top else (None, nuc))
            return out

        return rewrite(first, reference_is_top) + rewrite(second, reference_is_top), True

    # Case B
    if donor_has_top:
        extended = list(target_cols)
        for t, _ in donor_cols:
            if t is not None:
                extended.append((t, None))
        return extended, True
    else:
        extended = []
        for _, b in donor_cols:
            if b is not None:
                extended.append((None, b))
        extended.extend(target_cols)
        return extended, True


# ================================================================
# Interactions
# ================================================================
def try_interact(piece_a, piece_b, allow_pairing, allow_merging):
    if not piece_a or not piece_b:
        return None

    # 1. Merging
    if allow_merging:
        merged = try_merge(piece_a, piece_b)
        if merged is not None:
            return merged
        merged = try_merge(piece_b, piece_a)
        if merged is not None:
            return merged

    # 2. Pairing 
    if allow_pairing:
        if len(piece_a) >= len(piece_b):
            result, ok = try_pair_absorb(piece_a, piece_b)
        else:
            result, ok = try_pair_absorb(piece_b, piece_a)
        if ok:
            return result

    # 3. Elongation
    if len(piece_a) >= len(piece_b):
        result, ok = try_elongate(piece_a, piece_b)
    else:
        result, ok = try_elongate(piece_b, piece_a)
    if ok:
        return result

    return None


# ================================================================
# Splitting (replication of complete ds)
# ================================================================
def complete_ss_to_ds(cols):
    completed = []
    for t, b in cols:
        if t is not None and b is None:
            if random.random() < COMPLETION_ERROR:
                # Wrong nucleotide: pick a non-complement
                wrong = random.choice([x for x in NUCLEOTIDES if x != COMP[t]])
                completed.append((t, wrong))
            else:
                completed.append((t, COMP[t]))
        elif b is not None and t is None:
            if random.random() < COMPLETION_ERROR:
                wrong = random.choice([x for x in NUCLEOTIDES if x != COMP[b]])
                completed.append((wrong, b))
            else:
                completed.append((COMP[b], b))
        elif t is not None and b is not None:
            completed.append((t, b))
        else:
            raise ValueError("Column with both strands None in complete_ss_to_ds")
    return completed


def try_split(cols):
    if not piece_is_complete_ds(cols):
        return None
    if len(cols) < CA_MIN_DS_FOR_SPLIT:
        return None

    # Step 1: denature into two ss
    top_ss = [(t, None) for t, b in cols]
    bot_ss = [(None, b) for t, b in cols]

    # Step 2: auto-complete each ss back to ds
    child_left = complete_ss_to_ds(top_ss)
    child_right = complete_ss_to_ds(bot_ss)

    return child_left, child_right


# ================================================================
# Cellular automation core
# ================================================================
def run_ca(
    n_cells,
    n_cycles,
    allow_pairing,
    allow_merging,
    seed,
    mu=CA_MU,
    verbose=False,
    ss_only=False,
):
    random.seed(seed)
    np_rng = np.random.default_rng(seed)

    # Initialise grid: every cell gets a random piece
    grid = [random_piece(ss_only=ss_only) for _ in range(n_cells)]
    
    cooldown = [0] * n_cells  # cycles remaining before cell can split

    # Initialise resource field
    resources = np.full(n_cells, RESOURCE_INIT, dtype=np.int32)

    history = []  # will store a compact snapshot each cycle
    split_log = []
    resource_history = []

    # Store initial state
    history.append(_snapshot(grid))
    resource_history.append(resources.astype(np.float32).copy())

    for cycle in range(1, n_cycles + 1):
        if verbose and cycle % 50 == 0:
            n_occupied = sum(1 for c in grid if c is not None)
            print(f"  cycle {cycle}/{n_cycles}  occupied={n_occupied}/{n_cells}")

        # Tick cooldowns
        for i in range(n_cells):
            if cooldown[i] > 0:
                cooldown[i] -= 1

        # Phase 0: Mutation
        for i in range(n_cells):
            if grid[i] is not None:
                grid[i] = mutate_piece(grid[i], mu=mu)

        # Phase 0b: Mismatch resolution
        for i in range(n_cells):
            if grid[i] is None:
                continue
            grid[i] = resolve_mismatches(grid[i])

        # Phase 0c: Resource diffusion and replenishment
        # Each cell shares DIFFUSION_RATE of its resource with each neighbor
        # (periodic boundary), then receives a slow uniform replenishment.
        # left_idx  = np.roll(resources, 1)    # resources[(i-1) % n_cells]
        # right_idx = np.roll(resources, -1)   # resources[(i+1) % n_cells]
        # outflow   = 2.0 * DIFFUSION_RATE * resources
        # inflow    = DIFFUSION_RATE * (left_idx + right_idx)
        # resources = resources - outflow + inflow + REPLENISH_RATE
        # resources = np.clip(resources, 0.0, RESOURCE_MAX)

        # Phase 1: Neighbor interactions
        # Process cells in random order to avoid directional bias.
        order = list(range(n_cells))
        np_rng.shuffle(order)
        consumed = set()  

        for i in order:
            if i in consumed or grid[i] is None:
                continue

            # Random direction: look left-first or right-first
            left_nb = (i - 1) % n_cells
            right_nb = (i + 1) % n_cells
            if random.random() < 0.5:
                neighbors = [left_nb, right_nb]
            else:
                neighbors = [right_nb, left_nb]

            for j in neighbors:
                if j in consumed or grid[j] is None:
                    continue
                if j == i:
                    continue  

                # Resource gate: cell i must have minimum free nucleotides to donate to any growth operation.
                if resources[i] < RESOURCE_INTERACT_MIN:
                    break  

                result = try_interact(
                    grid[i], grid[j],
                    allow_pairing=allow_pairing,
                    allow_merging=allow_merging,
                )
                if result is not None:
                    cols_gained = max(0, len(result) - len(grid[i]))
                    cost = cols_gained * RESOURCE_COST_PER_COL
                    resources[i] = max(0.0, resources[i] - cost)

                    grid[i] = result
                    grid[j] = None
                    cooldown[j] = 0  
                    consumed.add(j)

        # Phase 2: Splitting (replication)
        split_cells = []
        for i in range(n_cells):
            if grid[i] is None:
                continue
            if cooldown[i] > 0:
                continue  # still in refractory period
            split_result = try_split(grid[i])
            if split_result is None:
                continue

            child_left, child_right = split_result

            split_log.append({
                "cycle": cycle,
                "cell": i,
                "parent_seq": piece_primary_sequence(grid[i]),
                "parent_len": len(grid[i]),
                "child_top_seq": piece_primary_sequence(child_left),
                "child_bot_seq": piece_primary_sequence(child_right),
            })

            split_cells.append((i, child_left, child_right))

        for i, child_left, child_right in split_cells:
            j_left = (i - 1) % n_cells
            j_right = (i + 1) % n_cells

            grid[i] = None
            cooldown[i] = 0

            # Place left child
            if grid[j_left] is None:
                grid[j_left] = child_left
                cooldown[j_left] = SPLIT_COOLDOWN
            elif stability_score(child_left) > stability_score(grid[j_left]):
                grid[j_left] = child_left
                cooldown[j_left] = SPLIT_COOLDOWN

            # Place right child
            if grid[j_right] is None:
                grid[j_right] = child_right
                cooldown[j_right] = SPLIT_COOLDOWN
            elif stability_score(child_right) > stability_score(grid[j_right]):
                grid[j_right] = child_right
                cooldown[j_right] = SPLIT_COOLDOWN

        # Phase 3: Resource-gated environmental injection
        for i in range(n_cells):
            if grid[i] is None:
                if resources[i] < RESOURCE_SPAWN_THRESHOLD:
                    continue
                spawn_prob = resources[i] / RESOURCE_MAX
                if random.random() > spawn_prob:
                    continue
                
                new_piece = random_piece(ss_only=ss_only)
                
                cost = len(new_piece) * RESOURCE_SPAWN_COST_PER_COL
                if resources[i] < cost:
                    continue  
                grid[i] = new_piece
                resources[i] = max(0.0, resources[i] - cost)
                cooldown[i] = 0
            else:
                if resources[i] < RESOURCE_INTERACT_MIN:
                    continue
                absorb_prob = resources[i] / RESOURCE_MAX
                if random.random() > absorb_prob:
                    continue

                new_piece = random_piece(ss_only=ss_only)
                
                len_before = len(grid[i])
                result = try_interact(
                    grid[i], new_piece,
                    allow_pairing=allow_pairing,
                    allow_merging=allow_merging,
                )
                if result is not None:
                    cols_gained = max(0, len(result) - len_before)
                    cost = cols_gained * RESOURCE_COST_PER_COL
                    resources[i] = max(0.0, resources[i] - cost)
                    grid[i] = result

        # Record state
        history.append(_snapshot(grid))
        resource_history.append(resources.astype(np.float32).copy())

    return history, split_log, resource_history

def _snapshot(grid):
    snap = []
    for cols in grid:
        if cols is None:
            snap.append("")
        else:
            snap.append(piece_primary_sequence(cols))
    return snap


# ================================================================
# K-mer analysis
# ================================================================
def compute_kmer_grid(history, k):
    N_TOP = 64  # number of distinct k-mers to colour

    # Pass 1: global k-mer census
    global_counts = Counter()
    for snap in history:
        for seq in snap:
            if len(seq) >= k:
                for km in extract_kmers(seq, k):
                    global_counts[km] += 1

    if not global_counts:
        n_cycles_plus1 = len(history)
        n_cells = len(history[0]) if history else 0
        return (
            np.full((n_cycles_plus1, n_cells), -1, dtype=np.int16),
            [],
            global_counts,
        )

    top_kmers = [km for km, _ in global_counts.most_common(N_TOP)]
    top_set = {km: idx for idx, km in enumerate(top_kmers)}

    # Pass 2: assign k-mer identity per cell
    n_times = len(history)
    n_cells = len(history[0])
    kmer_grid = np.full((n_times, n_cells), -1, dtype=np.int16)

    for t, snap in enumerate(history):
        for i, seq in enumerate(snap):
            if len(seq) < k:
                continue
            kmers_in_cell = extract_kmers(seq, k)
            for km in kmers_in_cell:
                if km in top_set:
                    kmer_grid[t, i] = top_set[km]
                    break  

    return kmer_grid, top_kmers, global_counts


def compute_per_cycle_kmer_metrics(history, k):
    rows = []
    for t, snap in enumerate(history):
        counter = Counter()
        for seq in snap:
            for km in extract_kmers(seq, k):
                counter[km] += 1

        total = sum(counter.values())
        unique = len(counter)
        top1 = counter.most_common(1)[0][1] if counter else 0

        singletons = sum(1 for c in counter.values() if c == 1)
        rmf = (total - singletons) / total if total > 0 else 0.0
        red = 1.0 - (unique / total) if total > 0 else 0.0

        rows.append({
            "cycle": t,
            f"total_kmers_k{k}": total,
            f"unique_kmers_k{k}": unique,
            f"top1_count_k{k}": top1,
            f"repeat_mass_fraction_k{k}": round(rmf, 6),
            f"redundancy_k{k}": round(red, 6),
        })
    return rows


# ================================================================
# Split-based heredity tracking
# ================================================================
def compute_heredity_metrics(history, split_log, k):
    MAX_TRACK_AGE = 100
    rows = []

    for ev in split_log:
        sc = ev["cycle"]
        child_seqs = [ev["child_top_seq"], ev["child_bot_seq"]]
        child_kmers = set()
        for seq in child_seqs:
            child_kmers.update(extract_kmers(seq, k))

        if not child_kmers:
            continue

        for age in range(0, MAX_TRACK_AGE + 1):
            t = sc + age
            if t >= len(history):
                break

            soup_kmers = set()
            for seq in history[t]:
                soup_kmers.update(extract_kmers(seq, k))

            surviving = child_kmers & soup_kmers
            frac = len(surviving) / len(child_kmers)
            rows.append({
                "split_cycle": sc,
                "age": age,
                "frac_surviving": round(frac, 4),
                "n_surviving": len(surviving),
                "n_child_kmers": len(child_kmers),
            })

    return rows

# ================================================================
# Visualization
# ================================================================
def plot_resource_field(resource_history, label, outpath):
    mat = np.stack(resource_history, axis=0)   # (n_times, n_cells)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        mat,
        origin="upper",
        aspect="auto",
        interpolation="nearest",
        cmap="YlOrRd",
        vmin=0.0,
        vmax=RESOURCE_MAX,
    )
    fig.colorbar(im, ax=ax, label="Resource level (free nucleotides)")
    ax.set_xlabel("Cell index")
    ax.set_ylabel("Cycle")
    ax.set_title(f"CA resource field — {label}")
    fig.savefig(outpath, dpi=700, bbox_inches="tight")
    plt.close(fig)

def plot_ca_spacetime(kmer_grid, top_kmers, k, label, outpath):
    def _hex_from_rgb01(rgb):
        r, g, b = rgb
        return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))

    def _generate_distinct_colors(n):
        hues = np.linspace(0.0, 1.0, 360, endpoint=False)
        sats = [0.65, 0.80, 0.95]
        vals = [0.80, 0.92, 1.00]

        candidates = []
        for h in hues:
            for s in sats:
                for v in vals:
                    candidates.append(colorsys.hsv_to_rgb(float(h), float(s), float(v)))
        candidates = np.array(candidates, dtype=float)

        chosen = [candidates[0]]
        remaining = np.delete(candidates, 0, axis=0)

        while len(chosen) < n:
            chosen_arr = np.array(chosen, dtype=float)
            d = np.sqrt(((remaining[:, None, :] - chosen_arr[None, :, :]) ** 2).sum(axis=2))
            min_d = d.min(axis=1)
            idx = int(min_d.argmax())
            chosen.append(remaining[idx])
            remaining = np.delete(remaining, idx, axis=0)

        return [_hex_from_rgb01(c) for c in chosen]

    n_top = len(top_kmers)
    if n_top == 0:
        # Nothing to plot — save a blank placeholder
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.text(0.5, 0.5, f"No {k}-mers detected",
                ha="center", va="center", fontsize=14)
        ax.set_title(f"{label} | {k}-mer CA  (no data)")
        fig.savefig(outpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    base_colors = [
        "#e5e7eb",  
    ]

    base_colors.extend(_generate_distinct_colors(n_top))
    
    cmap = ListedColormap(base_colors)
    boundaries = list(range(-1, n_top + 1))
    norm = BoundaryNorm(boundaries, cmap.N)

    plot_data = kmer_grid.astype(np.int16)

    def _max_run_length_1d(arr, value):
        best = 0
        cur = 0
        for x in arr:
            if x == value:
                cur += 1
                if cur > best:
                    best = cur
            else:
                cur = 0
        return best

    max_domain = np.zeros(n_top, dtype=int)
    for t in range(kmer_grid.shape[0]):
        row = kmer_grid[t]
        for idx in range(n_top):
            r = _max_run_length_1d(row, idx)
            if r > max_domain[idx]:
                max_domain[idx] = r

    TOP_HIGHLIGHT = 8  
    replicating_ids = list(np.argsort(max_domain)[-TOP_HIGHLIGHT:])
    replicating_ids.sort()

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(
        plot_data,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
        origin="upper",
    )
    ax.set_xlabel("Cell index (spatial)")
    ax.set_ylabel("Cycle (time)")
    ax.set_title(f"{label} | {k}-mer dominance CA spacetime diagram")

    # Legend
    
    legend_elements = [Patch(facecolor=base_colors[0], label="no top k-mer")]

    for idx in replicating_ids:
        km = top_kmers[idx]
        legend_elements.append(Patch(facecolor=base_colors[idx + 1], label=km))

    ax.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=6,      
        frameon=True,
        ncol=1,
    )

    fig.subplots_adjust(right=0.78)

    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_emergence_curves(metrics_by_cond, k, outpath):
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {
        "pairing_merging": "#2563eb",
        "elongation_only": "#16a34a",
    }
    labels_map = {
        "pairing_merging": "pairing + merging",
        "elongation_only": "elongation only (null)",
    }

    for cond, rows_list in metrics_by_cond.items():
        cycle_vals = defaultdict(list)
        for seed_rows in rows_list:
            for row in seed_rows:
                cyc = row["cycle"]
                val = row.get(f"repeat_mass_fraction_k{k}", 0.0)
                cycle_vals[cyc].append(val)

        cycles = sorted(cycle_vals.keys())
        means = [np.mean(cycle_vals[c]) for c in cycles]
        stds = [np.std(cycle_vals[c]) for c in cycles]
        means = np.array(means)
        stds = np.array(stds)

        color = colors.get(cond, "#666666")
        lbl = labels_map.get(cond, cond)
        ax.plot(cycles, means, color=color, label=lbl, linewidth=1.5)
        ax.fill_between(cycles, means - stds, means + stds,
                        color=color, alpha=0.15)

    ax.set_xlabel("Cycle")
    ax.set_ylabel(f"Repeat mass fraction ({k}-mer)")
    ax.set_title(f"CA: {k}-mer repeat emergence")
    ax.legend()
    ax.set_xlim(0, max(cycles) if cycles else 1)
    ax.set_ylim(0, 1)
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_piece_length_over_time(history_by_cond, outpath):
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {
        "pairing_merging": "#2563eb",
        "elongation_only": "#16a34a",
    }
    labels_map = {
        "pairing_merging": "pairing + merging",
        "elongation_only": "elongation only (null)",
    }

    for cond, hist_list in history_by_cond.items():
        cycle_vals = defaultdict(list)
        for hist in hist_list:
            for t, snap in enumerate(hist):
                lengths = [len(seq) for seq in snap if seq]
                mean_len = np.mean(lengths) if lengths else 0.0
                cycle_vals[t].append(mean_len)

        cycles = sorted(cycle_vals.keys())
        means = [np.mean(cycle_vals[c]) for c in cycles]
        stds = [np.std(cycle_vals[c]) for c in cycles]

        color = colors.get(cond, "#666666")
        lbl = labels_map.get(cond, cond)
        ax.plot(cycles, means, color=color, label=lbl, linewidth=1.5)
        ax.fill_between(cycles, np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds),
                        color=color, alpha=0.15)

    ax.set_xlabel("Cycle")
    ax.set_ylabel("Mean sequence length")
    ax.set_title("CA: Mean piece length over time")
    ax.legend()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_splits_over_time(split_logs_by_cond, outpath):
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {
        "pairing_merging": "#2563eb",
        "elongation_only": "#16a34a",
    }
    labels_map = {
        "pairing_merging": "pairing + merging",
        "elongation_only": "elongation only (null)",
    }

    for cond, logs_list in split_logs_by_cond.items():
        cycle_counts = defaultdict(list)
        for log in logs_list:
            cum = defaultdict(int)
            running = 0
            for ev in sorted(log, key=lambda e: e["cycle"]):
                running += 1
                cum[ev["cycle"]] = running
            # Fill forward
            if cum:
                max_cyc = max(cum.keys())
                prev = 0
                for c in range(0, max_cyc + 1):
                    if c in cum:
                        prev = cum[c]
                    cycle_counts[c].append(prev)

        if not cycle_counts:
            continue

        cycles = sorted(cycle_counts.keys())
        means = [np.mean(cycle_counts[c]) for c in cycles]
        stds = [np.std(cycle_counts[c]) for c in cycles]

        color = colors.get(cond, "#666666")
        lbl = labels_map.get(cond, cond)
        ax.plot(cycles, means, color=color, label=lbl, linewidth=1.5)
        ax.fill_between(cycles, np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds),
                        color=color, alpha=0.15)

    ax.set_xlabel("Cycle")
    ax.set_ylabel("Cumulative split events")
    ax.set_title("CA: Replication (split) events")
    ax.legend()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ================================================================
# Experiment
# ================================================================
def run_experiment(
    n_cells=512,
    n_cycles=512,
    n_seeds=5,
    mu=CA_MU,
    out_dir="ca_results",
):
    os.makedirs(out_dir, exist_ok=True)

    CONDITIONS = [
        ("pairing_merging", True, True, False),    
        ("elongation_only", False, False, True),
    ]

    # Storage for cross-seed aggregation
    all_histories = defaultdict(list)    
    all_split_logs = defaultdict(list)  
    all_kmer_metrics = defaultdict(list) 
    all_resource_histories = defaultdict(list)

    total_runs = n_seeds * len(CONDITIONS)
    done = 0

    for cond_label, allow_pair, allow_merge, ss_only in CONDITIONS:
        for s in range(1, n_seeds + 1):
            done += 1
            print(f"[{done}/{total_runs}] {cond_label} seed={s} "
                  f"({n_cells} cells × {n_cycles} cycles)")

            history, split_log, resource_history = run_ca(
                n_cells=n_cells,
                n_cycles=n_cycles,
                allow_pairing=allow_pair,
                allow_merging=allow_merge,
                seed=s,
                mu=mu,
                verbose=True,
                ss_only=ss_only,
            )

            all_histories[cond_label].append(history)
            all_split_logs[cond_label].append(split_log)
            all_resource_histories[cond_label].append(resource_history)

            for k in KMER_SIZES:
                kmer_rows = compute_per_cycle_kmer_metrics(history, k)
                all_kmer_metrics[cond_label].append(kmer_rows)

            if s == 1:
                for k in KMER_SIZES:
                    kmer_grid, top_kmers, _ = compute_kmer_grid(history, k)
                    png_path = os.path.join(
                        out_dir,
                        f"ca_spacetime_{cond_label}_k{k}.png"
                    )
                    plot_ca_spacetime(
                        kmer_grid, top_kmers, k, cond_label, png_path
                    )
                    print(f"    saved {png_path}")

                # Resource field spacetime diagram (seed 1 only)
                res_png = os.path.join(out_dir, f"ca_resource_{cond_label}.png")
                plot_resource_field(resource_history, cond_label, res_png)
                print(f"    saved {res_png}")

            n_splits = len(split_log)
            print(f"    splits={n_splits}")

    print("\nGenerating cross-seed summary plots...")

    # Emergence curves
    for k in KMER_SIZES:
        metrics_by_cond = {}
        for cond_label, _, _, _ in CONDITIONS:
            seed_lists = []
            entries = all_kmer_metrics[cond_label]
            for idx in range(0, len(entries), len(KMER_SIZES)):
                k_offset = KMER_SIZES.index(k)
                if idx + k_offset < len(entries):
                    seed_lists.append(entries[idx + k_offset])
            metrics_by_cond[cond_label] = seed_lists

        outpath = os.path.join(out_dir, f"ca_emergence_k{k}.png")
        plot_emergence_curves(metrics_by_cond, k, outpath)
        print(f"  saved {outpath}")

    # Piece length over time
    outpath = os.path.join(out_dir, "ca_piece_length.png")
    plot_piece_length_over_time(all_histories, outpath)
    print(f"  saved {outpath}")

    # Split events over time
    outpath = os.path.join(out_dir, "ca_splits.png")
    plot_splits_over_time(all_split_logs, outpath)
    print(f"  saved {outpath}")

    # Write CSVs
    all_split_rows = []
    for cond_label, _, _, _ in CONDITIONS:
        for s_idx, log in enumerate(all_split_logs[cond_label]):
            for ev in log:
                row = dict(ev)
                row["seed"] = s_idx + 1
                row["cond"] = cond_label
                all_split_rows.append(row)

    if all_split_rows:
        csv_path = os.path.join(out_dir, "ca_splits.csv")
        fieldnames = sorted(all_split_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(all_split_rows)
        print(f"  saved {csv_path} ({len(all_split_rows)} rows)")

    # Per-cycle metrics CSV (aggregate across conditions and seeds)
    all_metric_rows = []
    for cond_label, _, _, _ in CONDITIONS:
        entries = all_kmer_metrics[cond_label]
        n_k = len(KMER_SIZES)
        seed_idx = 0
        for block_start in range(0, len(entries), n_k):
            seed_idx += 1
            per_cycle = defaultdict(dict)
            for k_offset in range(n_k):
                if block_start + k_offset >= len(entries):
                    break
                for row in entries[block_start + k_offset]:
                    cyc = row["cycle"]
                    per_cycle[cyc].update(row)

            for cyc in sorted(per_cycle.keys()):
                merged_row = per_cycle[cyc]
                merged_row["seed"] = seed_idx
                merged_row["cond"] = cond_label
                all_metric_rows.append(merged_row)

    if all_metric_rows:
        csv_path = os.path.join(out_dir, "ca_metrics.csv")
        key_union = set()
        for r in all_metric_rows:
            key_union.update(r.keys())
        fieldnames = ["seed", "cycle", "cond"] + sorted(
            key_union - {"seed", "cycle", "cond"})
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(all_metric_rows)
        print(f"  saved {csv_path} ({len(all_metric_rows)} rows)")

    # Heredity metrics CSV
    all_heredity_rows = []
    for cond_label, _, _, _ in CONDITIONS:
        for s_idx, (hist, log) in enumerate(
            zip(all_histories[cond_label], all_split_logs[cond_label])
        ):
            for k in KMER_SIZES:
                hrows = compute_heredity_metrics(hist, log, k)
                for hr in hrows:
                    hr["seed"] = s_idx + 1
                    hr["cond"] = cond_label
                    hr["k"] = k
                    all_heredity_rows.append(hr)

    if all_heredity_rows:
        csv_path = os.path.join(out_dir, "ca_heredity.csv")
        fieldnames = ["seed", "cond", "k", "split_cycle", "age",
                      "frac_surviving", "n_surviving", "n_child_kmers"]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(all_heredity_rows)
        print(f"  saved {csv_path} ({len(all_heredity_rows)} rows)")

    print("\nDone.")


# ================================================================
# Entry 
# ================================================================
if __name__ == "__main__":
    log_path = "ca_simulation.log"

    with open(log_path, "w", encoding="utf-8") as log_f:
        with redirect_stdout(log_f), redirect_stderr(log_f):
            print("=" * 60)
            print("Barricelli DNA-norm Cellular Automaton")
            print("=" * 60)
            print()

            run_experiment(
                n_cells=512,
                n_cycles=512,
                n_seeds=5,
                mu=CA_MU,
                out_dir="ca_results",
            )

    print(f"Wrote log to {log_path}")


