import torch
from datasets import load_from_disk

class RewardModelTrainer:
    def __init__(self, configs):
        self.configs = configs
        self.tokenizer = None
        self.model = None
        self.train_dataset = None
        self.eval_dataset = None

    def load_dataset(self):
        self.train_dataset = load_from_disk(dataset_path=self.configs.train_path)
        self.eval_dataset = load_from_disk(dataset_path=self.configs.eval_path)

    def init_model_and_tokenizer(self):
        pass 

    @staticmethod
    def pairwise_loss(chosen_rewards, rejected_rewards, epsilion:float=1e+12):
        differance = chosen_rewards - rejected_rewards
        loss = -torch.log(torch.sigmoid(differance) + epsilion).mean()
        return loss

   def train(self):
        # 1) load dataset
        self.load_dataset()

        # 2) init tokenizer & model
        self.init_tokenizer_and_model()

        # 3) wrap datasets with our PreferencePairDataset
        max_length = getattr(self.configs, "max_length", 512)
        train_ds = PreferencePairDataset(self.train_dataset, self.tokenizer, max_length=max_length)
        train_loader = DataLoader(
            train_ds,
            batch_size=getattr(self.configs, "batch_size", 2),
            shuffle=True,
            collate_fn=collate_fn,
        )

        eval_loader = None
        if self.eval_dataset is not None:
            eval_ds = PreferencePairDataset(self.eval_dataset, self.tokenizer, max_length=max_length)
            eval_loader = DataLoader(eval_ds, batch_size=getattr(self.configs, "batch_size", 2), collate_fn=collate_fn)

        # 4) optimizer and scheduler
        lr = getattr(self.configs, "reward_lr", 1e-5)
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=getattr(self.configs, "weight_decay", 0.0))

        total_steps = len(train_loader) * getattr(self.configs, "epochs", 1)
        scheduler = get_scheduler(
            name=getattr(self.configs, "lr_scheduler_type", "linear"),
            optimizer=optimizer,
            num_warmup_steps=getattr(self.configs, "warmup_steps", 0),
            num_training_steps=total_steps,
        )

        scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() and getattr(self.configs, "use_amp", True) else None

        # 5) training loop
        self.model.train()
        print("Starting reward model training...")
        global_step = 0
        for epoch in range(getattr(self.configs, "epochs", 1)):
            for step, batch in enumerate(train_loader):
                optimizer.zero_grad()

                # Move inputs
                chosen_ids = batch["chosen_input_ids"].to(device)
                chosen_mask = batch["chosen_attention_mask"].to(device)
                rejected_ids = batch["rejected_input_ids"].to(device)
                rejected_mask = batch["rejected_attention_mask"].to(device)

                if scaler:
                    with torch.cuda.amp.autocast():
                        chosen_logits = self.model(input_ids=chosen_ids, attention_mask=chosen_mask).logits.squeeze(-1)
                        rejected_logits = self.model(input_ids=rejected_ids, attention_mask=rejected_mask).logits.squeeze(-1)
                        loss = self.pairwise_loss(chosen_logits, rejected_logits)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    chosen_logits = self.model(input_ids=chosen_ids, attention_mask=chosen_mask).logits.squeeze(-1)
                    rejected_logits = self.model(input_ids=rejected_ids, attention_mask=rejected_mask).logits.squeeze(-1)
                    loss = self.pairwise_loss(chosen_logits, rejected_logits)
                    loss.backward()
                    optimizer.step()

                scheduler.step()
                global_step += 1

                if global_step % getattr(self.configs, "log_steps", 50) == 0:
                    # compute simple pairwise accuracy on this minibatch
                    with torch.no_grad():
                        acc = (chosen_logits > rejected_logits).float().mean().item()
                    print(f"Epoch {epoch} Step {global_step} loss={loss.item():.4f} batch_acc={acc:.3f}")

            # optional per-epoch eval
            if eval_loader is not None:
                eval_acc = self.evaluate(eval_loader)
                print(f"Epoch {epoch} eval_pairwise_acc={eval_acc:.4f}")

        # 6) save model & tokenizer
        out_dir = getattr(self.configs, "reward_output_dir", getattr(self.configs, "output_dir", "reward_model"))
        os.makedirs(out_dir, exist_ok=True)
        print(f"Saving reward model to {out_dir} ...")
        # If using PEFT, save_peft_model recommended
        try:
            self.model.save_pretrained(out_dir)
        except Exception:
            # If model is PEFT wrapper
            from peft import PeftModel
            if isinstance(self.model, PeftModel):
                self.model.base_model.save_pretrained(out_dir)
                self.model.save_pretrained(out_dir)
            else:
                raise
        self.tokenizer.save_pretrained(out_dir)
        print("Saved.")
