Executive Summary
Mình đã khảo sát codebase bằng đọc file, không sửa code. Flow hiện tại là R2GenCMN/A3Net-style, batch dạng tuple, loss chính duy nhất là Masked NLL trong modules/loss.py (line 5). Model train output đã là log_probs vì BaseCMN._forward (line 489) gọi F.log_softmax(...) ở dòng 492, nên công thức -input.gather(...) hiện tại là đúng.

Vị trí ít phá code nhất để gắn Weighted Masked NLL: thêm reports_weights vào dataset/collate/trainer dưới dạng tuple field thứ 5 sau reports_masks, nhưng giữ backward compatibility bằng cách trainer unpack linh hoạt hoặc collate luôn trả weight khi args.use_weighted_loss=True. Không cần sửa model forward.

Current Training/Data/Loss Flow
Entrypoint train chính là main_train.py (line 174). Args/config đọc bằng argparse ở parse_agrs (line 49), script shell truyền config từ train_iu_xray.sh (line 27) và train_mimic_cxr.sh (line 1). Config được truyền tuần tự vào Tokenizer(args), R2DataLoader(args, tokenizer, ...), BaseCMNModel(args, tokenizer), Trainer(...) tại main_train.py (line 197).

Seed/device/AMP:

Seed: torch.manual_seed, np.random.seed, deterministic cudnn tại main_train.py (line 191). Chưa thấy torch.cuda.manual_seed_all hoặc worker seed.
Device/DataParallel: BaseTrainer._prepare_device (line 281), DataParallel tại trainer.py (line 49). Không thấy DDP.
AMP: arg --use_amp ở main_train.py (line 70), GradScaler/autocast ở trainer.py (line 57).
flowchart TD
  A[main_train.py main] --> B[parse_agrs argparse]
  B --> C[dump run_config/env to save_dir]
  C --> D[set seed/cudnn]
  D --> E[Tokenizer(args)]
  E --> F[R2DataLoader train/val/test]
  F --> G[Dataset tokenizes reports in BaseDataset]
  E --> H[BaseCMNModel(args, tokenizer)]
  H --> I[VisualExtractor + BaseCMN encoder_decoder]
  I --> J[build_optimizer/build_lr_scheduler]
  J --> K[Trainer.train]
  K --> L[Trainer._train_epoch]
  L --> M[model(images, reports_ids, mode='train')]
  M --> N[BaseCMN._forward returns log_probs]
  N --> O[compute_loss Masked NLL]
  O --> P[backward AMP/normal + optimizer.step]
  L --> Q[compute_nll_sum_and_tokens train log]
  L --> R[val teacher-forcing NLL aggregation]
  R --> S[val/test sample decoding]
  S --> T[compute_scores BLEU/METEOR/ROUGE]
  T --> U[monitor val_metric]
  U --> V[current_checkpoint.pth/model_best.pth]
Detailed File-By-File Findings

modules/datasets.py (line 14): BaseDataset đọc annotation.json, lấy self.ann[self.split], tokenizes ngay trong __init__ ở dòng 24-26. Raw report vẫn nằm trong example['report'], nhưng không return ra batch.
IuxrayMultiImageDataset.getitem (line 118): trả (image_id, image, report_ids, report_masks, seq_length). image là stack 2 ảnh.
MimiccxrSingleImageDataset.getitem (line 136): trả tuple tương tự, nhưng image_id bị overwrite thành full image path ở dòng 141.
modules/dataloaders.py (line 9): R2DataLoader chọn dataset theo args.dataset_name ở dòng 34-37. collate_fn pad ids/masks theo max length trong batch ở dòng 52-66.
modules/tokenizers.py (line 7): tokenizer word-level, không phải subword. __call__ clean report, split bằng whitespace, thêm BOS/EOS ở dòng 147-153.
models/models.py (line 9): wrapper BaseCMNModel; train mode gọi encoder-decoder mode='forward' ở dòng 27-29 hoặc 43-45.
modules/base_cmn.py (line 489): _forward cắt input decoder bằng seq[:, :-1] ở base_cmn.py (line 479), output là F.log_softmax(self.logit(out), dim=-1) ở dòng 492.
modules/trainer.py (line 347): train loop unpack batch 4 field ở dòng 357, gọi loss ở dòng 366, backward ở dòng 369-372, validation NLL ở dòng 410-426, sample metrics ở dòng 431-474.
modules/metrics.py (line 8): compute_scores dùng BLEU_1..4, METEOR, ROUGE_L. CIDEr tồn tại trong pycocoevalcap/eval.py (line 38) nhưng không được dùng trong trainer.
modules/tester.py (line 100): test chỉ sample decode, tính metric, ghi res.csv/gts.csv ở dòng 122-124. Không tính test loss.
Checkpoint: trainer.py (line 296) lưu current_checkpoint.pth, model_best.pth trong args.save_dir.
Tensor Shape Table

Tensor	Nơi tạo	Shape thực tế
IU images	IuxrayMultiImageDataset + collate	[B, 2, 3, 224, 224]
MIMIC images	MimiccxrSingleImageDataset + collate	[B, 3, 224, 224]
reports_ids	dataloaders.py (line 57)	LongTensor [B, L_batch]
reports_masks	same	FloatTensor [B, L_batch], 1 for real token incl. BOS/EOS
decoder input seq	base_cmn.py (line 479)	reports_ids[:, :-1], [B, L-1]
loss target	loss.py (line 20)	reports_ids[:, 1:], [B, L-1]
loss mask	same	reports_masks[:, 1:], [B, L-1]
output log_probs	base_cmn.py (line 492)	[B, L-1, vocab_size+1]
visual patches IU	models.py (line 23)	likely [B, 98, 2048] for ResNet 7x7 x 2
visual patches MIMIC	visual_extractor.py (line 16)	likely [B, 49, 2048]
Loss Flow Analysis
Current loss:

LanguageModelCriterion.forward(input, target, mask) at loss.py (line 9).
It truncates target/mask to input.size(1) at lines 11-12.
NLL is -input.gather(2, target.long().unsqueeze(2)).squeeze(2) * mask at line 13.
Divisor is torch.sum(mask) at line 14. No clamp_min.
compute_loss creates a new criterion every call at loss.py (line 18). .mean() is redundant because criterion already returns scalar.
compute_nll_sum_and_tokens is used for train logging and validation aggregation at trainer.py (line 385) and trainer.py (line 419). This is token-level global aggregation, better than mean-of-batch-means.
No auxiliary loss found by rg except compute_mlc clinical efficacy script, not training loss.
Important: train optimization uses self.criterion(...) at trainer.py (line 366), but reported train_loss cuối epoch is recomputed unweighted NLL via compute_nll_sum_and_tokens at trainer.py (line 396). Khi thêm weighted loss, cần log rõ train_loss_weighted và train_nll_unweighted.

Tokenizer Alignment Analysis
Sequence convention chắc chắn từ code:

Tokenizer.__call__: [BOS] + clean_report(report).split() + [EOS] at tokenizers.py (line 147).
Dataset truncates toàn bộ list ids bằng [:max_seq_length] ở datasets.py (line 25). Nếu report quá dài, EOS có thể bị cắt mất.
Decoder input là reports_ids[:, :-1]; target loss là reports_ids[:, 1:]. Vì vậy timestep 0 nhận BOS và dự đoán token đầu tiên, timestep cuối nhận token trước EOS và dự đoán EOS.
Với weights nên lưu cùng chiều với reports_ids: [w_BOS] + token_weights + [w_EOS], PAD sẽ được collate bằng 0 nếu theo mask. Loss sẽ dùng reports_weights[:, 1:], nghĩa là weight của token được dự đoán, không phải token input.
Tokenizer là word-level, dấu chấm . là một token riêng do cleaner tạo ' . '. Ví dụ raw "No pleural effusion." sau clean nhiều khả năng thành "no pleural effusion ." theo clean_report_iu_xray (line 110) hoặc clean_report_mimic_cxr (line 121).
Vì vậy ReportGraphWeighter phải dùng chính tokenizer.clean_report(report).split() để tránh lệch với ids. Không dùng cleaner riêng kiểu clean_report().split() nếu logic khác file này.
Test alignment đề xuất:

raw = "No pleural effusion."
tokens = tokenizer.clean_report(raw).split()
ids = tokenizer(raw)
weights = [1.0] + weighter.weight_tokens(tokens) + [1.0]
print(tokens)
print(ids)
print([tokenizer.get_token_by_id(i) for i in ids])
print(weights)
print("loss target tokens:", [tokenizer.get_token_by_id(i) for i in ids[1:]])
print("loss weights:", weights[1:])
Expected: weight của no, pleural, effusion, . nằm ở weights[1:], align với reports_ids[:, 1:].

Integration Plan For Weighted Masked NLL
Patch tối thiểu cho modules/loss.py (line 5):

LanguageModelCriterion.forward(input, target, mask, weights=None)
Nếu weights is None: giữ công thức cũ.
Nếu có weights: truncate cùng target/mask, cast .to(input.device).float(), effective mask là mask.float() * weights.
Denominator nên effective.sum().clamp_min(1e-8) để tránh zero.
Công thức:

loss = sum(mask * weight * nll) / sum(mask * weight)
API nên là:

compute_loss(output, reports_ids, reports_masks, reports_weights=None)
compute_nll_sum_and_tokens(output, reports_ids, reports_masks, reports_weights=None)
Backward compatibility:

reports_weights=None cho loss y hệt cũ.
args.use_weighted_loss=False thì trainer gọi như cũ.
args.use_weighted_loss=True thì dataset/collate/trainer bắt buộc có reports_weights.
Config đề xuất thêm ở main_train.py (line 162):

--use_weighted_loss
--weight_mode none|clinical_rule|report_graph
--max_weight
--normalize_weights
--debug_weight_samples
--weight_cache_path
Recommended Implementation Option
Chọn Option A trước: sinh reports_weights trong BaseDataset.__init__ hoặc __getitem__, dựa trên example['report'] và tokenizer cleaner.

Lý do:

Raw report có sẵn trong self.examples ở datasets.py (line 23).
Tokenization đã diễn ra ở dataset, nên sinh weight tại đây dễ align nhất.
Ít sửa model nhất: chỉ dataset/collate/trainer/loss.
Debug dễ vì có thể in raw report + ids + weights từ cùng example.
Cụ thể với tuple batch:

Dataset hiện trả (image_id, image, report_ids, report_masks, seq_length).
Nếu thêm weight, nên trả (image_id, image, report_ids, report_masks, report_weights, seq_length).
Collate trả (image_id_batch, image_batch, reports_ids, reports_masks, reports_weights).
Trainer đổi unpack tại trainer.py (line 357), 410 (line 410), 431 (line 431), 456 (line 456). Không cần sửa model/evaluator logic nếu evaluator bỏ qua weights.
Nơi đặt weighter: modules/report_weighting.py hoặc modules/loss_weighting.py. Mình nghiêng modules/report_weighting.py vì weight sinh từ report/tokenizer, không phải bản thân criterion.

Risks And Mitigations

Risk	Evidence	Mitigation
Token alignment lệch	tokenizer cleaner riêng tại tokenizers.py (line 110)	Weighter dùng tokenizer.clean_report(...).split()
EOS bị cắt khi report dài	dataset truncates ids at datasets.py (line 25)	Truncate weights cùng ids; cân nhắc đảm bảo EOS nếu cần
Output vocab là vocab_size+1	base_cmn.py (line 389)	Không đổi trong patch loss; note token id extra có thể decode <unk> nếu sample ra
Denominator zero	loss.py (line 14)	clamp_min(1e-8)
Weight quá lớn unstable	weighted formula mới	max_weight, optional normalize mean weight near 1
Validation weighted loss khó so baseline	val currently unweighted global NLL	Log cả weighted và unweighted
Batch tuple dễ vỡ	Trainer hard-unpack 4 fields	Helper unpack batch hoặc gate theo config
CPU/GPU mismatch	trainer chỉ move ids/masks ở trainer.py (line 359)	Move reports_weights.to(self.device)
DataParallel	trainer.py (line 49)	Weight chỉ vào loss ngoài model, không ảnh hưởng scatter
Clinical modules chưa có	rg không thấy CheXbert/RadGraph parser nội bộ	Bắt đầu bằng clinical_rule, graph structured nhẹ không external heavy model
Concrete Patch Plan
Commit 1: weighted criterion only
Files: modules/loss.py, tests mới.
Goal: weights=None giữ loss cũ; all-one weights bằng loss cũ.
Expected: train hiện tại không đổi.

Commit 2: all-one reports_weights through batch
Files: modules/datasets.py, modules/dataloaders.py, modules/trainer.py, modules/tester.py nếu test hard-unpack.
Goal: batch có weight nhưng mặc định toàn 1.
Expected: loss/metrics không đổi ngoài sai số rất nhỏ.

Commit 3: clinical rule weighter
Files: modules/report_weighting.py, main_train.py, dataset integration.
Goal: weight theo role clinical đơn giản, không external model.
Expected: debug sample in đúng token/weight.

Commit 4: report graph weighter
Files: modules/report_weighting.py, optional cache path.
Goal: graph-based token weighting, log stats.
Expected: reproducible nếu cache bật.

Commit 5: ablation
Runs: baseline NLL, weighted clinical_rule, weighted report_graph.
Log: train_loss_weighted, val_loss_weighted, val_nll_unweighted, BLEU/METEOR/ROUGE, weight stats.

Unit Tests And Debug Checklist
Unit tests nên có:

Weighted all ones bằng old loss.
Padding weight khác nhau không ảnh hưởng khi mask = 0.
Token high weight làm loss tăng khi token đó sai.
Shape mismatch truncate đúng theo output.size(1).
Denominator không zero khi weights toàn 0 hoặc mask toàn 0.
Debug batch đầu tiên nên log:

raw_report
clean_tokens
ids
decoded tokens
mask
weights
target tokens = ids[1:]
target weights = weights[1:]
top weighted tokens
nll per token
weighted nll per token
Questions/Unknowns That Require Manual Confirmation

Dataset thật data/iu_xray/annotation.json và data/mimic_cxr/annotation.json không có trong workspace hiện tại; chỉ thấy data/r2gencmn.md. Cần xác nhận schema thực tế có luôn key report, image_path, id như code giả định.
Chưa xác định yêu cầu cụ thể của ReportGraphWeighter: rule clinical-role/graph schema, danh sách entity/relation, max weight mặc định.
Nếu muốn dùng CIDEr để monitor, hiện trainer chưa dùng CIDEr dù package có sẵn.
Nếu weighted validation dùng để chọn checkpoint, cần quyết định monitor metric vẫn là BLEU_4 hay chuyển sang val_loss_weighted/val_nll_unweighted. Hiện mặc định monitor là val_BLEU_4 tại main_train.py (line 129), nên weighted loss không trực tiếp quyết định checkpoint.