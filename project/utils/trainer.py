# utils/trainer.py
class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, device, config, logger, log_dir):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.logger = logger
        self.log_dir = log_dir
        
        self.best_val_loss = float('inf')
        self.unfreeze_epoch = config['training'].get('unfreeze_epoch', 0)
    
    def train(self):
        for epoch in range(self.config['training']['epochs']):
            # 특정 에폭 이후 백본 학습 시작
            if epoch == self.unfreeze_epoch:
                self.model.unfreeze_backbones()
                self.logger.info("Unfreezing backbone networks")
            
            # 학습
            train_loss = self._train_epoch(epoch)
            
            # 검증
            val_loss, metrics = self._validate_epoch(epoch)
            
            # 체크포인트 저장
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(self.log_dir / 'best_model.pth')
            
            # 로깅
            self.logger.info(
                f"Epoch {epoch+1}/{self.config['training']['epochs']} - "
                f"Train Loss: {train_loss:.4f} - "
                f"Val Loss: {val_loss:.4f} - "
                f"mAP: {metrics['mAP']:.4f}"
            )
            
            # 에폭마다 스케줄러 업데이트
            if self.scheduler is not None:
                self.scheduler.step()
    
    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        
        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.device)
            targets = {k: v.to(self.device) for k, v in targets.items()}
            
            self.optimizer.zero_grad()
            loss = self.model(images, targets)
            loss.backward()
            self.optimizer.step()
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)
    
    def _validate_epoch(self, epoch):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for images, targets in self.val_loader:
                images = images.to(self.device)
                targets = {k: v.to(self.device) for k, v in targets.items()}
                
                loss = self.model(images, targets)
                total_loss += loss.item()
        
        val_loss = total_loss / len(self.val_loader)
        metrics = self.evaluate()
        
        return val_loss, metrics
    
    def evaluate(self):
        """모델 성능 평가"""
        return evaluate_model(
            model=self.model,
            data_loader=self.val_loader,
            device=self.device,
            config=self.config
        )
    
    def save_checkpoint(self, path):
        """체크포인트 저장"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        torch.save(checkpoint, path)