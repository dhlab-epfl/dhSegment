from .diva.process import diva_post_processing_fn
# from .Ornaments.ornaments_post_processing import ornaments_post_processing_fn
from .page.process import page_post_processing_fn
from .cbad.process import cbad_post_processing_fn
# from .DIBCO.dibco_post_processing import dibco_binarization_fn
# from .Cini.cini_post_processing import cini_post_processing_fn
from .diva.evaluation import eval_fn
# from .Cini.cini_evaluation import cini_evaluate_folder
from .cbad.evaluation import eval_fn
# from .DIBCO.dibco_evaluation import dibco_evaluate_folder
# from .Ornaments.ornaments_evaluation import ornament_evaluate_folder
from .page.evaluation import eval_fn
# from .evaluation.base import evaluate_epoch