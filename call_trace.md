# Call Trace: `uv run python scripts/train_all.py`

## Entry point — [scripts/train_all.py](scripts/train_all.py)
```
main()
├── parse_args()
├── setup_logger("rlhf.main")
└── cfg.paths.make_dirs()
```

---

## Stage 0 — Sanity Check — [utils/reward_utils.py](utils/reward_utils.py)
```
run_sanity_check()
└── compute_hedge_score(text)          ← called once per test case (6×)
```

---

## Stage 1 — SFT — [training/sft_trainer.py](training/sft_trainer.py)
```
run_sft(cfg)
├── setup_logger("rlhf.sft")
├── cfg.paths.make_dirs()
├── load_tokenizer(model_name)
│   └── GPT2Tokenizer.from_pretrained(model_name)
├── GPT2LMHeadModel.from_pretrained(model_name)
├── build_sft_dataset(SFT_DATA, tokenizer, max_length)
├── TrainingArguments(...)
├── Trainer(model, args, train_dataset)
├── trainer.train()                    ← HuggingFace training loop
├── model.save_pretrained(sft_checkpoint)
└── tokenizer.save_pretrained(sft_checkpoint)
```
Back in `main()`: `sft_model.to(device)`

---

## Stage 2 — Reward Model — [training/rm_trainer.py](training/rm_trainer.py)
```
run_rm_training(sft_model, tokenizer, cfg, device)
├── setup_logger("rlhf.rm")
├── cfg.paths.make_dirs()
├── build_reward_model(model_name, sft_model, device)   ← models/
├── compute_hedge_score(w), compute_hedge_score(l)      ← label verification (per pair)
├── train_reward_model(model, pairs, tokenizer, ...)    ← models/ — Bradley-Terry loss loop
└── [eval loop per text]
    ├── get_reward(reward_model, tokenizer, text, device)
    └── compute_hedge_score(text)
```

---

## Stage 3 — PPO — [training/ppo_trainer.py](training/ppo_trainer.py)
```
run_ppo_training(sft_model, reward_model, tokenizer, cfg, device)
├── setup_logger("rlhf.ppo")
├── cfg.paths.make_dirs()
├── build_policy_model(model_name, sft_model)           ← models/
├── build_ref_model(model_name, sft_model)              ← models/ (frozen)
├── build_ppo_dataset(FINANCIAL_PROMPTS, tokenizer)     ← data/
├── build_ppo_config(cfg)
├── PPOTrainer(config, policy, ref, tokenizer, dataset)
└── [for each step × num_steps (default 30)]
    ├── ppo_trainer.generate(query_tensors, ...)        ← policy rollout
    ├── tokenizer.batch_decode(response_tensors)
    ├── get_reward(reward_model, tokenizer, text, device)  ← per completion
    ├── ppo_trainer.step(queries, responses, rewards)   ← PPO gradient update
    └── compute_hedge_score(text)                       ← logging metric
```
Then: `policy_model.save_pretrained(...)`, `tokenizer.save_pretrained(...)`

---

## Evaluation — [evaluation/evaluator.py](evaluation/evaluator.py)
```
compare_sft_vs_rlhf(sft_model, policy_model, tokenizer, device)
└── [for each prompt in EVAL_PROMPTS]
    ├── generate_from_lm(sft_model, tokenizer, prompt, device)
    │   └── model.generate(...)  →  tokenizer.decode(...)
    ├── generate_completion(policy_model, tokenizer, prompt, device)   ← models/
    ├── compute_hedge_score(sft_text)
    └── compute_hedge_score(rlhf_text)

reward_hacking_demo(init_kl_coef)
└── compute_hedge_score(text)          ← per REWARD_HACKING_EXAMPLES entry

plot_training_curves(training_log, save_path)      ← unless --no_plots
└── matplotlib: plt.subplots() → plt.savefig() → plt.show()
```

---

## Summary flow
```
main()
 ├─ Stage 0: run_sanity_check()
 ├─ Stage 1: run_sft()           → sft_model, tokenizer
 ├─ Stage 2: run_rm_training()   → reward_model
 ├─ Stage 3: run_ppo_training()  → policy_model, training_log
 └─ Eval:    compare_sft_vs_rlhf()
             reward_hacking_demo()
             plot_training_curves()
```
