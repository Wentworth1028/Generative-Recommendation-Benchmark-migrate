"""Training-time adapter for tokenizer debias objectives."""

import torch


class PopularityDebiasController:
    """Own popularity scheduling, item weights, and distributed EMA updates."""

    def __init__(self, config, optimizer, item_popularity, device, accelerator=None):
        self.config = config
        self.optimizer = optimizer
        self.item_popularity = item_popularity
        self.device = device
        self.accelerator = accelerator
        self.enabled = bool(getattr(optimizer, "is_debias_optimizer", False))
        self.weight = 0.0
        self.target_weight = 0.0
        self.schedule = "constant"
        self.start_epoch = 0
        self.warmup_epochs = 0
        self.log_distribution = False
        self.distribution_interval = 100
        if self.enabled:
            self.target_weight = float(self._config_get("balance_weight", "popularity_balance_weight", 0.0))
            self.weight = self.target_weight
            self.schedule = str(
                self._config_get("balance_schedule", "popularity_balance_schedule", "constant")
            ).lower()
            self.start_epoch = int(self._config_get("balance_start_epoch", "popularity_balance_start_epoch", 0))
            self.warmup_epochs = int(self._config_get("balance_warmup_epochs", "popularity_balance_warmup_epochs", 0))
            self.log_distribution = self._as_bool(
                self._config_get("log_distribution", "popularity_balance_log_distribution", False)
            )
            self.distribution_interval = max(
                1,
                int(self._config_get("distribution_interval", "popularity_balance_distribution_interval", 100) or 100),
            )
            if self.schedule not in {"constant", "linear", "delayed"}:
                raise ValueError(
                    f"Invalid popularity_balance_schedule: {self.schedule}. "
                    "Must be 'constant', 'linear', or 'delayed'."
                )
        self.item_popularity_tensor = self._build_item_popularity_tensor()

    def _build_item_popularity_tensor(self):
        if not self.item_popularity:
            return torch.empty(0, dtype=torch.float32)
        try:
            max_item_id = max(int(item_id) for item_id in self.item_popularity.keys())
        except (TypeError, ValueError):
            return torch.empty(0, dtype=torch.float32)
        lookup = torch.zeros(max_item_id + 1, dtype=torch.float32)
        for item_id, popularity in self.item_popularity.items():
            try:
                lookup[int(item_id)] = float(popularity)
            except (TypeError, ValueError):
                continue
        return lookup

    @staticmethod
    def _as_bool(value):
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    def _config_get(self, key, legacy_key=None, default=None):
        if legacy_key is not None and legacy_key in self.config:
            return self.config.get(legacy_key)
        if key in self.config:
            return self.config.get(key)
        return default

    def batch_weights(self, item_ids):
        if (
            not self.enabled
            or self.weight <= 0.0
            or self.item_popularity_tensor.numel() == 0
        ):
            return None
        if torch.is_tensor(item_ids):
            item_ids = item_ids.detach().long().cpu()
        else:
            item_ids = torch.as_tensor(item_ids, dtype=torch.long)
        if item_ids.numel() == 0:
            return torch.empty_like(item_ids, dtype=torch.float32, device=self.device)
        item_ids = item_ids.clamp_min(0)
        item_ids = item_ids.clamp_max(self.item_popularity_tensor.size(0) - 1)
        weights = self.item_popularity_tensor.index_select(0, item_ids.reshape(-1)).view(item_ids.shape)
        return weights.to(device=self.device)

    def scheduled_weight(self, epoch):
        if not self.enabled or self.target_weight <= 0.0:
            return 0.0
        if self.schedule == "constant":
            return self.target_weight
        start_epoch = max(self.start_epoch, 0)
        if epoch < start_epoch:
            return 0.0
        if self.schedule == "delayed":
            return self.target_weight
        warmup_epochs = max(self.warmup_epochs, 0)
        if warmup_epochs == 0:
            return self.target_weight
        progress = min(1.0, (epoch - start_epoch + 1) / warmup_epochs)
        return self.target_weight * progress

    def set_weight(self, weight):
        self.weight = weight
        if self.enabled:
            self.optimizer.popularity_balance_weight = weight

    def entropy_gap(self, popularity_balance_loss):
        if not self.enabled or not hasattr(self.optimizer, "popularity_balance_entropy_floor"):
            return 0.0
        if getattr(self.optimizer, "popularity_balance_mode", "batch_entropy") != "batch_entropy":
            return 0.0
        return popularity_balance_loss - float(self.optimizer.popularity_balance_entropy_floor())

    def distribution_metrics(self, tokenizer_output, popularity_weights):
        if not self.enabled or not self.log_distribution or self.weight <= 0.0:
            return {}
        return self.optimizer.compute_popularity_distribution_metrics(
            tokenizer_output,
            popularity_weights=popularity_weights,
        )

    def apply_ema_update(self):
        if not self.enabled or not hasattr(self.optimizer, "pop_pending_ema_observation"):
            return
        observation, item_count = self.optimizer.pop_pending_ema_observation()
        if observation is None or item_count <= 0:
            return

        def is_sparse(value):
            return isinstance(value, dict) and "keys" in value and "values" in value

        def scale(value, factor):
            if is_sparse(value):
                return {"keys": value["keys"], "values": value["values"] * factor}
            if isinstance(value, dict):
                return {key: scale(child, factor) for key, child in value.items()}
            if value is None:
                return None
            return value * factor

        def reduce(value):
            if is_sparse(value):
                if self.accelerator is None:
                    return value
                return {
                    "keys": self.accelerator.gather_for_metrics(value["keys"]),
                    "values": self.accelerator.gather_for_metrics(value["values"]),
                }
            if isinstance(value, dict):
                return {key: reduce(child) for key, child in value.items()}
            if value is None:
                return None
            if self.accelerator is not None:
                return self.accelerator.reduce(value, reduction="sum")
            return value

        def divide(value, denominator):
            if is_sparse(value):
                return {"keys": value["keys"], "values": value["values"] / denominator}
            if isinstance(value, dict):
                return {key: divide(child, denominator) for key, child in value.items()}
            if value is None:
                return None
            return value / denominator

        def first_tensor(value):
            if is_sparse(value):
                return value["values"]
            if isinstance(value, dict):
                for child in value.values():
                    found = first_tensor(child)
                    if found is not None:
                        return found
                return None
            return value

        with torch.no_grad():
            reference = first_tensor(observation)
            if reference is None:
                return
            count = reference.new_tensor(float(item_count))
            weighted = scale(observation, count)
            if self.accelerator is not None:
                weighted = reduce(weighted)
                count = self.accelerator.reduce(count, reduction="sum")
            global_count = int(count.item())
            mean_observation = divide(weighted, count.clamp_min(1.0))
            self.optimizer.apply_popularity_ema_update(mean_observation, global_count)
