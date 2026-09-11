# Model and asset notices

The root [`LICENSE`](../LICENSE) applies to project source code. It does **not**
grant a blanket licence for model weights, training datasets, geodata or aerial
imagery. Preserve the original terms and attribution for every asset below.

## Tracked model weights

| Asset | Provenance | Distribution note |
| --- | --- | --- |
| `swiss_best.pt` | PV segmentation checkpoint trained for this project using the Swiss PV dataset documented in the repository README. That dataset is listed by its publisher as CC0. | Keep the dataset and model attribution in the README. Verify any new training input before redistributing a replacement checkpoint. |
| `rid_best.pt` | Rooftop-obstacle checkpoint associated with the RID/RID2 evaluation and training workflow. | RID is published under CC BY-NC 4.0. Treat this checkpoint and any checkpoint trained with RID/RID2 material as non-commercial unless the rights holder grants broader permission. Preserve the authors' attribution, licence link and modification notice. |
| `*_metrics.json`, `*_status.json` | Project-generated evaluation and status metadata. | These files describe the corresponding checkpoint; do not remove their provenance fields. |

The repository does not claim that a trained model necessarily has the same
licence as its source code or dataset. If a model's provenance or downstream
rights are unclear, do not publish it as an open-source release asset; publish
only the source code and instructions to obtain authorised weights instead.

## Required RID attribution

RID dataset: Sebastian Krapf, Lukas Bogenrieder, Fabian Netzler, Georg Balke
and Markus Lienkamp. Licence: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
The licence requires appropriate credit, a link to the licence and an indication
of changes; it does not permit commercial use of the licensed material.

The original sources and additional implementation notes are linked in the
repository [README](../README.md#sources). Consult the original dataset terms
before adding further training data or publishing a new model.

## External map and geodata

Swiss imagery, Sonnendach results, swisstopo height data and SITG survey data
are requested from their providers at runtime or used under their respective
terms. They are not redistributed under the project AGPL licence.
