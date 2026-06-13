from genrec.models.LETTER.LETTER import LETTERT5ForConditionalGeneration


class LETTERTWMTL(LETTERT5ForConditionalGeneration):
    """LETTER model variant used with Token-Weighted Multi-Target Learning.

    Architecture and inference stay identical to LETTER; TWMTL is applied only
    by the paired downstream trainer.
    """

