from genrec.quantization.optimizers.base_optimizer import AbstractTokenizerOptimizer
from genrec.quantization.tokenizers.base_tokenizer import AbstractTokenizer
import torch
import logging
import os
from tqdm import tqdm
import numpy as np
from collections import Counter

class RQVAETrainer:
    def __init__(
        self, 
        config: dict,  
        tokenizer: AbstractTokenizer,
        optimizer: AbstractTokenizerOptimizer,
        accelerator = None
    ):
        self.config   = config
        self.tokenizer  = tokenizer  
        self.optimizer = optimizer
        self.accelerator = accelerator
        self.epochs = self.config.get('epochs')
        self.device = self.accelerator.device if self.accelerator else torch.device(self.config.get('device'))
        self.log_interval = self.config.get('log_interval')
        self.checkpoint_path = self.config.get('checkpoint_path')
        self.save_interval = self.config.get('save_interval')
        self.item_popularity = self.config.get('item_popularity', {})
        self.popularity_balance_weight = float(self.config.get('popularity_balance_weight', 0.0))
        self.target_popularity_balance_weight = self.popularity_balance_weight
        self.popularity_balance_schedule = self.config.get('popularity_balance_schedule', 'constant').lower()
        self.popularity_balance_start_epoch = int(self.config.get('popularity_balance_start_epoch', 0))
        self.popularity_balance_warmup_epochs = int(self.config.get('popularity_balance_warmup_epochs', 0))
        self.tensorboard_enabled = self._as_bool(self.config.get('tensorboard_enabled', False))
        self.tensorboard_dir = self.config.get('tensorboard_dir')
        if not self.tensorboard_dir:
            self.tensorboard_dir = os.path.join(os.path.dirname(self.checkpoint_path), "tensorboard")
        self.summary_writer = None
        if self.popularity_balance_schedule not in {'constant', 'linear', 'delayed'}:
            raise ValueError(
                f"Invalid popularity_balance_schedule: {self.popularity_balance_schedule}. "
                "Must be 'constant', 'linear', or 'delayed'."
            )

        self.save_best_on = self.config.get('save_best_on', 'collision_rate').lower()
        if self.save_best_on not in ['utilization', 'collision_rate']:
            raise ValueError(f"Invalid 'save_best_on' value: {self.save_best_on}. "
                             f"Must be 'utilization' or 'collision_rate'.")
        
        self.best_metric_value = 0.0 
        self.best_epoch = 0
        logging.info(f"The best model will be saved based on the best value of '{self.save_best_on}'.")
        if self.accelerator is None or self.accelerator.is_main_process:
            os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
            logging.info(f"The best model will be saved based on the best value of '{self.save_best_on}'.")
        # self.tokenizer.to(self.device)
        # self.optimizer.move_optimizer_state_to_device(self.device)

    @staticmethod
    def _as_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    def _quant_loss_weight(self) -> float:
        return float(getattr(self.optimizer, 'quant_loss_weight', self.config.get('quant_loss_weight', 1.0)))

    def _rq_loss_without_pop(self, recon_loss: float, commit_loss: float) -> float:
        return recon_loss + self._quant_loss_weight() * commit_loss

    def _init_training_trace(self):
        if self.accelerator is not None and not self.accelerator.is_main_process:
            return
        if self.tensorboard_enabled:
            self._init_tensorboard_writer()

    def _init_tensorboard_writer(self):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ModuleNotFoundError:
            logging.info("TensorBoard is not installed; RQ-VAE scalar events will not be written.")
            return
        os.makedirs(self.tensorboard_dir, exist_ok=True)
        self.summary_writer = SummaryWriter(log_dir=self.tensorboard_dir)
        logging.info(f"RQ-VAE TensorBoard scalars will be written to {self.tensorboard_dir}")

    def _append_training_trace(
        self,
        epoch: int,
        train_loss: float,
        train_recon: float,
        train_commit: float,
        train_pop_balance: float,
        effective_popularity_weight: float,
    ):
        if self.accelerator is not None and not self.accelerator.is_main_process:
            return
        rq_loss_without_pop = self._rq_loss_without_pop(train_recon, train_commit)
        weighted_pop_balance = effective_popularity_weight * train_pop_balance
        self._append_tensorboard_scalars(
            epoch + 1,
            train_loss,
            rq_loss_without_pop,
            train_recon,
            train_commit,
            train_pop_balance,
            weighted_pop_balance,
            effective_popularity_weight,
        )

    def _append_tensorboard_scalars(
        self,
        epoch: int,
        train_loss: float,
        rq_loss_without_pop: float,
        train_recon: float,
        train_commit: float,
        train_pop_balance: float,
        weighted_pop_balance: float,
        effective_popularity_weight: float,
    ):
        if self.summary_writer is None:
            return
        self.summary_writer.add_scalar("rqvae/total_loss", train_loss, epoch)
        self.summary_writer.add_scalar("rqvae/rq_loss_without_pop", rq_loss_without_pop, epoch)
        self.summary_writer.add_scalar("rqvae/recon_loss", train_recon, epoch)
        self.summary_writer.add_scalar("rqvae/commit_loss", train_commit, epoch)
        self.summary_writer.add_scalar("rqvae/quant_loss_weight", self._quant_loss_weight(), epoch)
        self.summary_writer.add_scalar("rqvae/popularity_balance_loss", train_pop_balance, epoch)
        self.summary_writer.add_scalar(
            "rqvae/weighted_popularity_balance_loss",
            weighted_pop_balance,
            epoch,
        )
        self.summary_writer.add_scalar(
            "rqvae/effective_popularity_balance_weight",
            effective_popularity_weight,
            epoch,
        )
        self.summary_writer.flush()

    def _batch_popularity_weights(self, item_ids):
        if self.popularity_balance_weight <= 0.0:
            return None
        if torch.is_tensor(item_ids):
            item_ids = item_ids.detach().cpu().tolist()
        weights = [float(self.item_popularity.get(int(item_id), 0.0)) for item_id in item_ids]
        return torch.tensor(weights, dtype=torch.float32, device=self.device)

    def _scheduled_popularity_balance_weight(self, epoch: int) -> float:
        if self.target_popularity_balance_weight <= 0.0:
            return 0.0
        if self.popularity_balance_schedule == 'constant':
            return self.target_popularity_balance_weight

        start_epoch = max(self.popularity_balance_start_epoch, 0)
        if epoch < start_epoch:
            return 0.0
        if self.popularity_balance_schedule == 'delayed':
            return self.target_popularity_balance_weight

        warmup_epochs = max(self.popularity_balance_warmup_epochs, 0)
        if warmup_epochs == 0:
            return self.target_popularity_balance_weight

        progress = min(1.0, (epoch - start_epoch + 1) / warmup_epochs)
        return self.target_popularity_balance_weight * progress

    def _set_popularity_balance_weight(self, weight: float):
        self.popularity_balance_weight = weight
        if hasattr(self.optimizer, 'popularity_balance_weight'):
            self.optimizer.popularity_balance_weight = weight

    def _calculate_codebook_utilization(self, train_dataloader, log_output=True):
        self.tokenizer.eval()
        
        unwrapped_tokenizer = self.accelerator.unwrap_model(self.tokenizer) if self.accelerator else self.tokenizer
        
        codebook_usage = [Counter() for _ in range(unwrapped_tokenizer.n_codebooks)]
        total_samples = 0
        
        with torch.no_grad():
            for _, embeddings in train_dataloader:
                embeddings = embeddings.to(self.device)
                
                indices = unwrapped_tokenizer.encode(embeddings)
                
                if self.accelerator is not None:
                    indices = self.accelerator.gather_for_metrics(indices)
                total_samples += indices.size(0)
                
                for layer_idx in range(unwrapped_tokenizer.n_codebooks):
                    layer_indices = indices[:, layer_idx].cpu().numpy()
                    for idx in layer_indices:
                        codebook_usage[layer_idx][idx] += 1
        
        utilization_rates = [len(usage) / unwrapped_tokenizer.codebook_size for usage in codebook_usage]
        
        if log_output:
            for i, rate in enumerate(utilization_rates):
                logging.info(f"Layer {i + 1}: {len(codebook_usage[i])}/{unwrapped_tokenizer.codebook_size} codes used, "
                             f"utilization rate: {rate:.4f}")
        avg_utilization = np.mean(utilization_rates)
        if log_output:
            logging.info(f"Average codebook utilization rate: {avg_utilization:.4f}")
            
        return utilization_rates, avg_utilization

    def _calculate_collision_rate(self, train_dataloader, log_output=True):
        self.tokenizer.eval()
        
        unwrapped_tokenizer = self.accelerator.unwrap_model(self.tokenizer) if self.accelerator else self.tokenizer
        
        indices_set = set()
        total_samples = 0
        with torch.no_grad():
            for _, embeddings in train_dataloader:
                embeddings = embeddings.to(self.device)
                
                indices = unwrapped_tokenizer.encode(embeddings)
                
                if self.accelerator is not None:
                    indices = self.accelerator.gather_for_metrics(indices)
                total_samples += indices.size(0)
                cpu_indices = indices.cpu().numpy()
                for index_tuple in cpu_indices:
                    code = "-".join(map(str, index_tuple))
                    indices_set.add(code)
        
        if total_samples == 0:
            collision_rate = 0.0
        else:
            collision_rate = (total_samples - len(indices_set)) / total_samples
        
        if log_output:
            logging.info(f"Collision Analysis: {total_samples - len(indices_set)} collisions found for {total_samples} samples.")
            logging.info(f"Total Unique Codes: {len(indices_set)}")
            logging.info(f"Collision rate: {collision_rate:.4f}")
            
        return collision_rate

    def _train_one_epoch(self, train_dataloader, epoch: int):
        self.tokenizer.train()
        total_loss, total_recon_loss, total_commit_loss, total_popularity_balance_loss = 0.0, 0.0, 0.0, 0.0
        is_main = self.accelerator is None or self.accelerator.is_main_process
        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch+1}/{self.epochs} [Training]",
            leave=False,
            disable=not is_main,
        )
        for step, (item_ids, embeddings) in enumerate(progress_bar, start=1):
            embeddings = embeddings.to(self.device)
            self.optimizer.zero_grad()
            tokenizer_output = self.tokenizer(embeddings)
            popularity_weights = self._batch_popularity_weights(item_ids)
            loss, reconstruction_loss, commit_loss, popularity_balance_loss = self.optimizer.compute_loss(
                embeddings,
                tokenizer_output,
                popularity_weights=popularity_weights,
            )
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            total_recon_loss += reconstruction_loss.item()
            total_commit_loss += commit_loss.item()
            total_popularity_balance_loss += popularity_balance_loss.item()
            if is_main and self.log_interval and step % self.log_interval == 0:
                weighted_popularity_balance_loss = self.popularity_balance_weight * popularity_balance_loss.item()
                rq_loss_without_pop = self._rq_loss_without_pop(
                    reconstruction_loss.item(),
                    commit_loss.item(),
                )
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.6e}',
                    'rq_no_pop': f'{rq_loss_without_pop:.6e}',
                    'recon_loss': f'{reconstruction_loss.item():.6e}',
                    'commit_loss': f'{commit_loss.item():.6e}',
                    'pop_balance_loss': f'{popularity_balance_loss.item():.6e}',
                    'weighted_pop': f'{weighted_popularity_balance_loss:.6e}',
                })
        return (
            total_loss / len(train_dataloader),
            total_recon_loss / len(train_dataloader),
            total_commit_loss / len(train_dataloader),
            total_popularity_balance_loss / len(train_dataloader),
        )

    def _save_checkpoint(self, epoch, metric_value=None, is_best=False, utilization_rate=None, collision_rate=None):
        if self.accelerator is not None and not self.accelerator.is_main_process:
            return None
        model_to_save = self.accelerator.unwrap_model(self.tokenizer) if self.accelerator else self.tokenizer
        if is_best:
            torch.save(model_to_save.state_dict(), self.checkpoint_path)
            logging.info(f"Best model saved to {self.checkpoint_path} ({self.save_best_on}: {metric_value:.4f})")
            return self.checkpoint_path
        else:
            base_path, ext = os.path.splitext(self.checkpoint_path)
            
            if utilization_rate is not None and collision_rate is not None:
                checkpoint_filename = f"{base_path}_epoch{epoch+1}_util{utilization_rate:.4f}_coll{collision_rate:.4f}{ext}"
            else:
                checkpoint_filename = f"{base_path}_epoch{epoch+1}{ext}"
                
            torch.save(model_to_save.state_dict(), checkpoint_filename)
            logging.info(f"Checkpoint saved to {checkpoint_filename}")
            return checkpoint_filename

    def fit(self, train_dataloader,valid_dataloader):
        logging.info("Start Training Tokenizer...")
        is_main = self.accelerator is None or self.accelerator.is_main_process
        if is_main:
            logging.info("Start Training Tokenizer...")
        if self.accelerator is not None:
            actual_optimizer = self.optimizer.optimizer if hasattr(self.optimizer, 'optimizer') else self.optimizer
            
            self.tokenizer, actual_optimizer, prepared_train_dl, prepared_valid_dl = self.accelerator.prepare(
                self.tokenizer, actual_optimizer, train_dataloader.dl, valid_dataloader.dl
            )
            train_dataloader.dl = prepared_train_dl
            valid_dataloader.dl = prepared_valid_dl
            
            if hasattr(self.optimizer, 'optimizer'):
                self.optimizer.optimizer = actual_optimizer
            else:
                self.optimizer = actual_optimizer
        if is_main:
            self._init_training_trace()
        for epoch in range(self.epochs):
            effective_popularity_weight = self._scheduled_popularity_balance_weight(epoch)
            self._set_popularity_balance_weight(effective_popularity_weight)
            train_loss, train_recon, train_commit, train_pop_balance = self._train_one_epoch(train_dataloader, epoch)
            if is_main:
                rq_loss_without_pop = self._rq_loss_without_pop(train_recon, train_commit)
                weighted_pop_balance = effective_popularity_weight * train_pop_balance
                logging.info(f"Epoch {epoch+1}/{self.epochs} | Train Loss: {train_loss:.8e} | "
                            f"RQ Loss Without Pop: {rq_loss_without_pop:.8e} | "
                            f"Train Recon Loss: {train_recon:.8e} | Train Commit Loss: {train_commit:.8e} | "
                            f"Train Popularity Balance Loss: {train_pop_balance:.8e} | "
                            f"Weighted Popularity Balance Loss: {weighted_pop_balance:.8e} | "
                            f"Popularity Balance Weight: {effective_popularity_weight:.8e}")
                self._append_training_trace(
                    epoch,
                    train_loss,
                    train_recon,
                    train_commit,
                    train_pop_balance,
                    effective_popularity_weight,
                )

            if (epoch + 1) % 1000 == 0:
                if is_main:
                    logging.info(f"\n=== Metrics Analysis at Epoch {epoch+1} ===")
                self._calculate_codebook_utilization(valid_dataloader, log_output=True)
                self._calculate_collision_rate(valid_dataloader, log_output=True)
                if is_main:
                    logging.info("=" * 60)

            if (epoch + 1) % self.save_interval == 0:
                _, avg_utilization = self._calculate_codebook_utilization(valid_dataloader, log_output=False)
                collision_rate = self._calculate_collision_rate(valid_dataloader, log_output=False)
                
                if self.save_best_on == 'utilization':
                    comparable_metric = avg_utilization
                    original_metric_for_log = avg_utilization
                else: # collision_rate
                    comparable_metric = 1.0 - collision_rate
                    original_metric_for_log = collision_rate

                if comparable_metric >= self.best_metric_value:
                    self.best_metric_value = comparable_metric
                    self.best_epoch = epoch
                    self._save_checkpoint(epoch, metric_value=original_metric_for_log, is_best=True)
                    if is_main:
                        logging.info(f"New best model found at epoch {epoch+1} with {self.save_best_on}: {original_metric_for_log:.4f}")
                
                self._save_checkpoint(epoch, utilization_rate=avg_utilization, collision_rate=collision_rate) 
        if is_main:                
            logging.info("\n=== Final Metrics Analysis ===")
        _, final_avg_utilization = self._calculate_codebook_utilization(valid_dataloader, log_output=True)
        final_collision_rate = self._calculate_collision_rate(valid_dataloader, log_output=True)
        if is_main:
            logging.info("=" * 60)
        
        self._save_checkpoint(self.epochs - 1, utilization_rate=final_avg_utilization, collision_rate=final_collision_rate)
        
        if self.save_best_on == 'utilization':
            final_comparable_metric = final_avg_utilization
            final_original_metric = final_avg_utilization
        else: # collision_rate
            final_comparable_metric = 1.0 - final_collision_rate
            final_original_metric = final_collision_rate

        if final_comparable_metric > self.best_metric_value:
            self.best_metric_value = final_comparable_metric
            self.best_epoch = self.epochs - 1
            self._save_checkpoint(self.epochs - 1, metric_value=final_original_metric, is_best=True)
            if is_main:
                logging.info(f"Final model is the best with {self.save_best_on}: {final_original_metric:.4f}")
        else:
            best_original_value = self.best_metric_value if self.save_best_on == 'utilization' else 1.0 - self.best_metric_value
            if is_main: 
                logging.info(f"Best model was at epoch {self.best_epoch+1} with {self.save_best_on}: {best_original_value:.4f}")
        
        best_original_value_final = self.best_metric_value if self.save_best_on == 'utilization' else 1.0 - self.best_metric_value
        if is_main:
            logging.info(f"Training complete. Best {self.save_best_on}: {best_original_value_final:.4f} at epoch {self.best_epoch+1}")
        if self.summary_writer is not None:
            self.summary_writer.flush()
            self.summary_writer.close()
            self.summary_writer = None
