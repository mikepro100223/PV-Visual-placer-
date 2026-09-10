from rooftop_pv.cli import parser


def test_swiss_prediction_and_evaluation_use_same_resolution():
    predict = parser().parse_args(["predict", "example.jpg"])
    evaluate = parser().parse_args(["evaluate"])
    assert predict.imgsz == evaluate.imgsz == 512
    assert predict.confidence == evaluate.confidence == .25


def test_obstacle_preparation_does_not_silently_sample_or_overwrite():
    arguments = parser().parse_args(["prepare-obstacles"])
    assert arguments.max_images is None
    assert not hasattr(arguments, "overwrite")
