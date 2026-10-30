# Phase 4 — model release audit

Audit date: 2026-10-30. No Phase 3F accuracy experiments, new training, or B3-V3 search was performed.

## Release decision

No model checkpoint is approved for public upload in this audit. The Cassava team fine-tuned checkpoint and the Coffee ResNet18 checkpoint both pass local hash and inference checks, but the available provenance does not establish redistribution rights. Accordingly, no new model release was published and no checkpoint was uploaded. The existing [`phase4-integration` source-only release](https://github.com/HiyoriVn/Leaf-Disease-Detection/releases/tag/phase4-integration) remains unchanged.

`checksums.sha256` records the two requested candidate files' local hashes and intended asset filenames. It is a verification aid only; it does not describe assets hosted by GitHub.

## Demo candidates

| Model | Actual local file | Bytes | SHA-256 | Architecture / preprocessing / class order | Local verification | Public status |
|---|---|---:|---|---|---|---|
| Team fine-tuned Cassava MobileNetV3-Small | `checkpoints/experiments/full_backbone_smooth_20261028/best.pth` | 6,201,897 | `7cb54b7de90ec775eb346a0a5ade31dfcdac7fff2e03db5264c1aa8c6fc98cd0` | `timm mobilenetv3_small_100`; RGB; OpenCV `INTER_AREA` resize 256, center crop 224, ImageNet normalization; five labels CBB, CBSD, CGM, CMD, Healthy | Config hash matched; strict load; `[1, 5]` inference output | Blocked: trained on Kaggle Cassava competition data; the inspected data page marks the dataset subject to competition rules. No evidence found that those rules grant redistribution of derived model weights. ImageNet initialization adds upstream dataset terms to review. |
| Coffee ResNet18 | `C:\Users\Admin\Downloads\coffee_model.pth` (outside the repository) | 44,792,523 | `12f958c9c08b0cc0b13311e86a23f1089c3e3476a8984a87db79cfd78f884900` | torchvision `resnet18`; RGB resize 224×224; ImageNet normalization; checkpoint order `Cerscospora`, `Healthy`, `Leaf rust`, `Phoma` | Registry hash and class mapping matched; model loaded; `[1, 4]` output | Blocked: supplied checkpoint has no recorded source URL or training-data identity/license. The training script uses ImageNet ResNet18 weights, whose upstream documentation calls out ImageNet terms. |

## Complete local model artifact inventory

The following inventory covers every `.pth`, `.safetensors`, and SavedModel payload file found under `checkpoints/`, plus the Coffee file supplied outside the repository. Logs and run metadata are not model weights.

| Artifact | Bytes | SHA-256 | Notes |
|---|---:|---|---|
| `checkpoints/cassava_mobilenetv3_best.pth` | 6,224,097 | `4043a480d7a48d636f4feefbe31df4c52eeadd8edb8404d31319eddd807ac4bb` | Published Cassava MobileNetV3-Small; strict-load/inference verified; publisher model card reports MIT, but upstream data/derived-weight terms are not fully established here |
| `checkpoints/cassava_efficientnet_b3_best_full.pth` | 43,308,023 | `7924296913c5f3a6c9d822212cdb0a8326b6d5c4cd2778078d6c47bebd0d406b` | B3 baseline, not V3; strict-load/inference verified; same-source Cassava data and ImageNet terms remain unresolved |
| `checkpoints/experiments/efficientnet_b3_pilot_local_20261029/best.pth` | 43,308,023 | `be66172f657e5dc545bfce15ef3dfb466f4936b8eed059ac8364bdc6f0dd1186` | Pilot experiment checkpoint; not a demo release candidate |
| `checkpoints/experiments/efficientnet_b3_pilot_local_20261029/last.pth` | 48,128,581 | `2b0876eb9315ea32c0433ad5bd4ee8e979e0d8c5faec107490da2bbbebf98ace` | Pilot training state; not a demo release candidate |
| `checkpoints/experiments/full_backbone_smooth_20261028/best.pth` | 6,201,897 | `7cb54b7de90ec775eb346a0a5ade31dfcdac7fff2e03db5264c1aa8c6fc98cd0` | Team fine-tuned Cassava candidate above |
| `checkpoints/experiments/pilot_cpu_20261028/best.pth` | 6,201,897 | `b9bfa087ab474ca664f92877cf2207e6309ee389cbef8c5710df100d38ce88a2` | Cassava CPU pilot; not a demo release candidate |
| `checkpoints/experiments/pilot_cuda_20261028/best.pth` | 6,201,897 | `793806d710394e0ea109b12bb4210e9da20ee83fef149ac4e0ceaaeccd4e2154` | Cassava CUDA pilot; not a demo release candidate |
| `checkpoints/experiments/standard_full_20261028/best.pth` | 6,201,897 | `2a221ebc6174964df6296d291ee7d8ead1e489e03b2e09eccbebc2e4a0cafdd3` | Cassava experiment; not a demo release candidate |
| `checkpoints/experiments/weighted_full_20261028/best.pth` | 6,201,897 | `af790048013c19eb3fff9cba77acece2247d5d4ec077de40463a749be9ab4996` | Cassava experiment; not a demo release candidate |
| `checkpoints/experiments/weighted_full_fast_20261028/best.pth` | 6,201,897 | `417116ee8d1643a3e159720ef7242540a205b159c3decf785469e196bdfe3c40` | Cassava experiment; not a demo release candidate |
| `checkpoints/imagenet_efficientnet_b3_ra2_in1k.safetensors` | 49,335,454 | `279d2a53898aa89dab43fd6bd7df9f706aea4cb9cf916223988bbcb8e5850469` | ImageNet initialization only; not a fine-tuned demo model |
| `checkpoints/imagenet_mobilenetv3_small_100.safetensors` | 10,241,912 | `46d2c063b18125884c48937afa4c49e18128869e52e8db96df48bf0a4d7ff697` | ImageNet initialization only; not a demo model |
| `checkpoints/cropnet/saved_model.pb` | 1,515,889 | `aac5d946061721786ff2cd3e8e49515d1cf660f9982a10266b519cd679410992` | Optional six-class CropNet artifact; not selected for this Cassava/Coffee release |
| `checkpoints/cropnet/variables/variables.data-00000-of-00001` | 16,971,880 | `dd6919af91cf984853e8c57a30a8bd08adbad99d424fda57474e871904a7a1e0` | CropNet SavedModel payload |
| `checkpoints/cropnet/variables/variables.index` | 5,607 | `54749e7c840c85fb08409ec1911128675e743498735e08f12f50ee8d27bd9199` | CropNet SavedModel payload |
| External `C:\Users\Admin\Downloads\coffee_model.pth` | 44,792,523 | `12f958c9c08b0cc0b13311e86a23f1089c3e3476a8984a87db79cfd78f884900` | Team Coffee model candidate above |

EfficientNet-B3 V3 remains absent from the prior verified checkout inventory. At the user's direction it is disabled in the demo registry and is not being searched for or retrained. Its model-specific config and archived validation record have not been deleted.

## Install-path verification and limits

The Cassava candidate was strict-loaded using `configs/cassava_finetuned_app.json`; Coffee was loaded from its actual external path using the registry SHA and expected class order. Both produced finite logits on a synthetic RGB image. `checksums.sha256` was compared with the bytes at these local paths and matched.

The clean-clone download flow cannot be validated end to end because the current GitHub release has no model assets and no model download URLs exist. README now names the exact target paths, provides PowerShell SHA checks, explains the current absence of assets, and states that model files must come from a future approved release or other authorized source. No URL has been invented.

## Permission evidence and blockers

- The [Kaggle Cassava competition data page](https://www.kaggle.com/competitions/cassava-leaf-disease-classification/data) lists the dataset license as “Subject to Competition Rules.” The repo's training report and config confirm the team MobileNet was trained using that dataset. I did not find a grant that explicitly permits public redistribution of derived weights.
- The [timm MobileNetV3 pretrained model card](https://huggingface.co/timm/mobilenetv3_small_100.lamb_in1k) states Apache-2.0. The [upstream timm documentation](https://github.com/huggingface/pytorch-image-models#licenses) notes ImageNet is for non-commercial research and says implications for weights are unclear. Apache-2.0 for code/model card alone does not resolve ImageNet data terms.
- The [RoCoLe paper page](https://www.sciencedirect.com/science/article/pii/S2352340919307693) is an open-access dataset description, but that does not identify the dataset used by this particular Coffee checkpoint. The Coffee checkpoint and the repo's old Coffee training script do not establish provenance for the supplied binary or rights in its training data. Its torchvision ImageNet initialization is another unresolved source.

To unblock public model assets, record permission for the Cassava source dataset and derived weights, and obtain Coffee checkpoint provenance plus explicit checkpoint/data redistribution terms from its owner. Once that evidence exists, the checksums in `checksums.sha256` can be used against the exact assets before creating a model release.

## Verification results

- Cassava team checkpoint strict load and synthetic RGB inference succeeded; returned five finite logits in the configured class order.
- Coffee ResNet18 strict load, registry hash/class validation, and synthetic RGB inference succeeded; returned four finite logits and three ranked predictions.
- Candidate checksums.sha256 values matched both actual local files byte-for-byte.
- Registry discovery offered the team Cassava model and Coffee model when their files were available, and did not offer B3-V3.
- Full Phase 4 suite: **30 passed**. These are local file checks; the GitHub release asset download/install path remains untestable until model assets are approved and published.

