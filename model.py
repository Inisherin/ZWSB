"""
双流时序深度网络 (Dual-Stream Temporal Network) 用于谵妄识别。

架构:
    外观流 (ResNet18) ─┐
                        ├→ 特征融合 → 时序Transformer → 片段嵌入 → 多片段注意力融合 → 分类
    动态流 (帧差+CNN) ──┘
"""
import torch
import torch.nn as nn
import torchvision.models as models
import config


class AppearanceStream(nn.Module):
    """外观流：使用预训练ResNet18提取每帧的面部外观特征"""

    def __init__(self, out_dim=512, pretrained=True):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        # 去掉最后的全连接层和平均池化层
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # 输出 [B, 512, 1, 1]
        self.out_dim = out_dim

    def forward(self, x):
        """
        Args:
            x: [B, T, C, H, W] 帧序列
        Returns:
            [B, T, out_dim] 每帧的外观特征
        """
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        feat = self.features(x)           # [B*T, 512, 1, 1]
        feat = feat.reshape(B, T, self.out_dim)
        return feat


class MotionStream(nn.Module):
    """动态流：计算帧差并用轻量CNN提取运动特征"""

    def __init__(self, out_dim=128):
        super().__init__()
        self.out_dim = out_dim
        # 轻量CNN处理帧差图像
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(128, out_dim)

    def forward(self, x):
        """
        Args:
            x: [B, T, C, H, W] 帧序列
        Returns:
            [B, T, out_dim] 每帧的动态特征（第一帧为零向量）
        """
        B, T, C, H, W = x.shape
        # 计算帧差: frame[t] - frame[t-1]
        diff = x[:, 1:] - x[:, :-1]                  # [B, T-1, C, H, W]
        # 第一帧没有前一帧，用零填充
        zero_first = torch.zeros(B, 1, C, H, W, device=x.device)
        diff = torch.cat([zero_first, diff], dim=1)   # [B, T, C, H, W]

        diff = diff.reshape(B * T, C, H, W)
        feat = self.cnn(diff)                          # [B*T, 128, 1, 1]
        feat = feat.reshape(B * T, 128)
        feat = self.fc(feat)                           # [B*T, out_dim]
        feat = feat.reshape(B, T, self.out_dim)
        return feat


class TemporalTransformer(nn.Module):
    """时序Transformer编码器：建模帧序列的时间依赖关系"""

    def __init__(self, d_model=640, nhead=8, num_layers=2, dropout=0.3, max_frames=None):
        super().__init__()
        self.d_model = d_model
        max_frames = max_frames or config.MAX_FRAMES_PER_CLIP
        # 可学习的CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        # 位置编码（支持动态长度：推理时对超长序列插值）
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_frames + 1, d_model)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, lengths):
        """
        Args:
            x: [B, T, d_model] 融合后的特征序列
            lengths: [B] 每个序列的有效长度
        Returns:
            [B, d_model] CLS token的输出作为片段嵌入
        """
        B, T, D = x.shape
        # 添加CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)                 # [B, T+1, D]
        # 添加位置编码（推理时如果序列长度超过预设，做线性插值）
        pos_emb = self.pos_embedding
        if T + 1 > pos_emb.shape[1]:
            import torch.nn.functional as F
            pos_emb = F.interpolate(
                pos_emb.transpose(1, 2), size=T + 1, mode='linear', align_corners=False
            ).transpose(1, 2)
        x = x + pos_emb[:, :T + 1, :]

        # 构建padding mask: True表示需要忽略的位置
        # CLS token始终有效(False)，之后根据lengths决定
        mask = torch.ones(B, T + 1, dtype=torch.bool, device=x.device)
        mask[:, 0] = False  # CLS token
        for i in range(B):
            mask[i, 1:lengths[i] + 1] = False

        x = self.transformer(x, src_key_padding_mask=mask)
        x = self.norm(x[:, 0])                          # 取CLS token输出 [B, D]
        return x


class MultiClipFusion(nn.Module):
    """多片段注意力融合：对不同测试环节的片段进行加权聚合"""

    def __init__(self, d_model=640):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.Tanh(),
            nn.Linear(d_model // 4, 1)
        )

    def forward(self, clip_embeddings, clip_mask):
        """
        Args:
            clip_embeddings: [B, num_clips, d_model] 各片段的嵌入
            clip_mask: [B, num_clips] 有效片段标记 (1=有效, 0=填充)
        Returns:
            [B, d_model] 融合后的患者级表示
        """
        # 计算注意力权重
        attn_scores = self.attention(clip_embeddings).squeeze(-1)  # [B, num_clips]
        # 将无效片段的权重设为极小值
        attn_scores = attn_scores.masked_fill(clip_mask == 0, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=1)           # [B, num_clips]
        # 加权求和
        # 处理全部无效的情况
        attn_weights = attn_weights.masked_fill(torch.isnan(attn_weights), 0.0)
        fused = torch.bmm(attn_weights.unsqueeze(1), clip_embeddings).squeeze(1)
        return fused


class AudioStream(nn.Module):
    """音频流：1D CNN 处理 MFCC 特征序列，输出紧凑音频嵌入"""

    def __init__(self, n_mfcc_total=None, out_dim=None):
        super().__init__()
        n_mfcc_total = n_mfcc_total or (config.N_MFCC * 3)
        out_dim = out_dim or config.AUDIO_DIM
        self.out_dim = out_dim
        # 1D CNN: 输入 [B, n_mfcc_total, T_audio], 输出 [B, out_dim, 1]
        self.cnn = nn.Sequential(
            nn.Conv1d(n_mfcc_total, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, out_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        """
        Args:
            x: [B, T_audio, n_mfcc_total]
        Returns:
            [B, out_dim]
        """
        x = x.transpose(1, 2)              # [B, n_mfcc_total, T_audio]
        x = self.cnn(x).squeeze(-1)        # [B, out_dim]
        return x


class DeliriumNet(nn.Module):
    """
    谵妄识别多模态时序网络（视觉双流 + 音频流 + 语义分数）。

    输入：患者的多个视频片段（3-4段）
    输出：谵妄概率 logits
    """

    def __init__(self, pretrained=True, use_audio=None):
        super().__init__()
        self.use_audio = use_audio if use_audio is not None else config.USE_AUDIO

        # 视觉双流
        self.appearance_stream = AppearanceStream(
            out_dim=config.APPEARANCE_DIM, pretrained=pretrained
        )
        self.motion_stream = MotionStream(out_dim=config.MOTION_DIM)
        self.fusion_proj = nn.Sequential(
            nn.Linear(config.FUSED_DIM, config.FUSED_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(config.DROPOUT)
        )
        self.temporal_transformer = TemporalTransformer(
            d_model=config.FUSED_DIM,
            nhead=config.NUM_HEADS,
            num_layers=config.NUM_TRANSFORMER_LAYERS,
            dropout=config.DROPOUT
        )

        # 音频流
        if self.use_audio:
            self.audio_stream = AudioStream(
                n_mfcc_total=config.N_MFCC * 3,
                out_dim=config.AUDIO_DIM
            )
            clip_emb_dim = config.FUSED_DIM_MULTIMODAL  # 769
        else:
            clip_emb_dim = config.FUSED_DIM              # 640

        self.clip_emb_dim = clip_emb_dim

        # 多片段融合和分类头
        self.clip_fusion = MultiClipFusion(d_model=clip_emb_dim)
        self.classifier = nn.Sequential(
            nn.Linear(clip_emb_dim, config.HIDDEN_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM, config.NUM_CLASSES)
        )

    def forward_single_clip(self, clip_frames, clip_length, audio_frames=None, semantic_score=None):
        """
        处理单个视频片段（含可选音频）。

        Args:
            clip_frames: [B, T, C, H, W]
            clip_length: [B]
            audio_frames: [B, T_audio, N_MFCC*3] 或 None
            semantic_score: [B, 1] 语义得分 或 None
        Returns:
            [B, clip_emb_dim]
        """
        app_feat = self.appearance_stream(clip_frames)    # [B, T, 512]
        mot_feat = self.motion_stream(clip_frames)        # [B, T, 128]
        fused = torch.cat([app_feat, mot_feat], dim=-1)   # [B, T, 640]
        fused = self.fusion_proj(fused)
        visual_emb = self.temporal_transformer(fused, clip_length)  # [B, 640]

        if self.use_audio and audio_frames is not None:
            audio_emb = self.audio_stream(audio_frames)   # [B, 128]
            if semantic_score is None:
                semantic_score = torch.full(
                    (audio_emb.shape[0], 1), 0.5, device=audio_emb.device
                )
            clip_emb = torch.cat([visual_emb, audio_emb, semantic_score], dim=-1)  # [B, 769]
        else:
            clip_emb = visual_emb

        return clip_emb

    def forward(self, clips, clip_lengths, clip_mask,
                audio_features=None, semantic_scores=None, audio_mask=None):
        """
        Args:
            clips: [B, num_clips, T, C, H, W]
            clip_lengths: [B, num_clips]
            clip_mask: [B, num_clips]
            audio_features: [B, num_clips, T_audio, N_MFCC*3] 或 None
            semantic_scores: [B, num_clips, 1] 或 None
            audio_mask: [B, num_clips] 或 None
        Returns:
            [B, 1] 谵妄概率 logits
        """
        B, num_clips, T, C, H, W = clips.shape
        clip_embeddings = []

        for i in range(num_clips):
            audio_i = audio_features[:, i] if audio_features is not None else None
            sem_i = semantic_scores[:, i] if semantic_scores is not None else None
            clip_emb = self.forward_single_clip(
                clips[:, i], clip_lengths[:, i], audio_i, sem_i
            )
            clip_embeddings.append(clip_emb)

        clip_embeddings = torch.stack(clip_embeddings, dim=1)  # [B, num_clips, clip_emb_dim]
        patient_emb = self.clip_fusion(clip_embeddings, clip_mask)
        logits = self.classifier(patient_emb)
        return logits


def count_parameters(model):
    """统计模型参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    model = DeliriumNet(pretrained=False)
    total, trainable = count_parameters(model)
    print(f"总参数量: {total:,}")
    print(f"可训练参数量: {trainable:,}")

    B, NC, T = 2, config.MAX_CLIPS, 10
    clips = torch.randn(B, NC, T, 3, config.FACE_SIZE, config.FACE_SIZE)
    clip_lengths = torch.tensor([[10, 8, 5, 0], [10, 10, 7, 3]])
    clip_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.float)
    audio = torch.randn(B, NC, config.MAX_AUDIO_FRAMES, config.N_MFCC * 3)
    sem = torch.rand(B, NC, 1)
    audio_mask = clip_mask.clone()

    logits = model(clips, clip_lengths, clip_mask, audio, sem, audio_mask)
    print(f"输出形状: {logits.shape}")  # [2, 1]
    print(f"clip_emb_dim: {model.clip_emb_dim}")
