from genrec.models.TIGER.TIGER import TIGER


class TIGERTWMTL(TIGER):
    """TIGER model variant used with Token-Weighted Multi-Target Learning.

    The model architecture is intentionally identical to the baseline TIGER T5
    model. TWMTL changes only the downstream training objective, implemented in
    the paired trainer, so tokenizer training and inference behavior stay intact.
    """

