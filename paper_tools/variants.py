"""Run notes -> SRT variant configuration.

Every experiment of the paper is the same method (methods/SRT_TTA_paper.py) run with a
different ``--note``. This table is the single source of truth for what each note means;
the method reads it at set-up time and paper_tools/collect_runs.py reads it to map runs
to table rows / figure points.

The default ("full") configuration reproduces the original SRT_TTA code exactly:
alpha = 1.75, beta = 3, rho_pi = rho_s = 0.9, sink gate + coupled weight, sign vote.
"""
import re

DEFAULT = dict(
    alpha=1.75,          # power of the affinity  A_alpha(p, pi)
    beta=3.0,            # temperature of the coupled weight  exp(beta * a_i * c_i)
    rho_pi=0.9,          # EMA momentum of the running marginal pi
    rho_s=0.9,           # EMA momentum of the orientation state s
    gate=True,           # sink-repulsive gate  a_i >= mean(a)
    weight='coupled',    # 'coupled' exp(beta*a*c) | 'conf' exp(beta*c) | 'none' 1
    orient='vote',       # 'vote' (ours) | 'none' | 'vote_raw' | 'per_sample' | 'oracle'
    intervention=None,   # None | 'all' | 'sa' | 'sr'   (Fig. 2b)
)

# note -> overrides of DEFAULT
VARIANTS = {
    'full': {},
    'ntu120': dict(alpha=1.25, beta=1.0, rho_pi=0.99),
    # ---------------------------------------------------------------- Tab. 3 (component ablation)
    'ablation_vanilla': dict(gate=False, weight='none', orient='none'),        # (a) entropy minimization
    'abl_conf': dict(gate=False, weight='conf', orient='none'),                # (b) + confidence weight
    'abl_gate_a2': dict(alpha=2.0, weight='none', orient='none'),              # (c) + sink gate (cosine)
    'abl_gate_wt_a2': dict(alpha=2.0, orient='none'),                          # (d) + coupled weight (cosine)
    'ablation_no_z2': dict(orient='none'),                                     # (e) + power affinity = SREM
    'ablation_no_alpha': dict(gate=False, weight='none'),                      # (f) entropy minimization + ATC
    # ---------------------------------------------------------------- Tab. 4 top (orientation rules)
    'ori_per_sample': dict(orient='per_sample'),     # lower-entropy view per sample
    'ori_no_sign': dict(orient='vote_raw'),          # EMA of the raw entropy gap (no sign)
    'ori_oracle': dict(orient='oracle'),             # true orientation of the domain
    # ---------------------------------------------------------------- Fig. 2(b)
    # (Fig. 2(c) uses the Source baseline wrapped by paper_tools.paper_logging, PAPER_PROBE_GAP=1)
    'interv_all': dict(intervention='all', orient='none'),
    'interv_sa': dict(intervention='sa', orient='none'),
    'interv_sr': dict(intervention='sr', orient='none'),
    # ---------------------------------------------------------------- timing (Tab. S4)
    'cost': {},
}

# rows of Tab. 3, in order: (row id, note)
ABLATION_ROWS = [
    ('a', 'ablation_vanilla'), ('b', 'abl_conf'), ('c', 'abl_gate_a2'), ('d', 'abl_gate_wt_a2'),
    ('e', 'ablation_no_z2'), ('f', 'ablation_no_alpha'), ('g', 'full'),
]
# rows of Tab. 4 (top), in order: (row id, note)
ORIENT_ROWS = [
    ('none', 'ablation_no_z2'), ('per_sample', 'ori_per_sample'), ('per_batch', 'sens_rho_0.00'),
    ('no_sign', 'ori_no_sign'), ('ours', 'full'), ('oracle', 'ori_oracle'),
]

_SENS = re.compile(r'^sens_(alpha|rho|beta)_([0-9]*\.?[0-9]+)$')


def parse_note(note):
    """Return the full configuration dict for a run note (unknown notes -> ValueError)."""
    note = (note or 'full').strip()
    cfg = dict(DEFAULT)
    if note in VARIANTS:
        cfg.update(VARIANTS[note])
    else:
        m = _SENS.match(note)
        if not m:
            raise ValueError(
                f'[SRT_TTA_paper] unknown run note {note!r}; '
                f'known: {sorted(VARIANTS)} or sens_alpha_X / sens_rho_X / sens_beta_X')
        key = {'alpha': 'alpha', 'rho': 'rho_s', 'beta': 'beta'}[m.group(1)]
        cfg[key] = float(m.group(2))
    cfg['note'] = note
    return cfg


def sens_value(note, kind):
    """sens_value('sens_alpha_1.50', 'alpha') -> 1.5 ; None if the note is not such a sweep point."""
    m = _SENS.match(note or '')
    if m and m.group(1) == kind:
        return float(m.group(2))
    return None
