import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# https://nbviewer.jupyter.org/github/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb
class VectorQuantizerEMA(nn.Module):
    def __init__(
        self, num_embeddings, embedding_dim, commitment_cost, decay=0.0, epsilon=1e-5
    ):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer("_ema_cluster_size", torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        # Euclidean Distance = Sqrt(Sum(Square(Differences)))
        # We ignore sqrt because we're taking the nearest neighbor
        # Which doesn't change when we take sqrt
        # Since we're working with multi-dimensional matrices
        # We can get rid of the sum as this is a vectorized operation
        # So we compute Square(Differences)
        # I.E. (A - B)^2 = A^2 + B^2 - 2AB
        distances = (
            torch.sum(flat_input ** 2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )

        # Encoding
        # Get Nearest Neighbors
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        # Create a one hot encoded matrix where 1 indicates the nearest neighbor
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=inputs.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            # N_t = N_(t-1) * gamma + n_t * (1 - gamma)
            self._ema_cluster_size = self._ema_cluster_size * self._decay + (
                1 - self._decay
            ) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon)
                * n
            )

            # m_t = m_(t-1) * gamma + z_t * (1 - gamma)
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(
                self._ema_w * self._decay + (1 - self._decay) * dw
            )

            # e = N_t / m_t
            self._embedding.weight = nn.Parameter(
                self._ema_w / self._ema_cluster_size.unsqueeze(1)
            )

        # Loss
        e_latent_loss = nn.functional.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return (
            loss,
            quantized.permute(0, 3, 1, 2).contiguous(),
            perplexity,
            encoding_indices,
        )

    def quantize_encoding_indices(self, encoding_indices, target_shape, device):
        # For use in inference/fusion generation
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=device
        )
        encodings.scatter_(1, encoding_indices, 1)
        quantized = torch.matmul(encodings, self._embedding.weight).view(target_shape)
        return quantized.permute(0, 3, 1, 2).contiguous()


class VQVAE(nn.Module):
    def __init__(
        self,
        image_channels=3,
        num_layers=4,
        kernel_size=2,
        stride=2,
        padding=0,
        input_image_dimensions=96,
        small_conv=False,
        embedding_dim=64,
        max_filters=512,
        use_max_filters=False,
        num_embeddings=512,
        commitment_cost=0.25,
        decay=0.99,
    ):
        super(VQVAE, self).__init__()
        if small_conv:
            num_layers += 1
        if not use_max_filters:
            max_filters = embedding_dim
        channel_sizes = self.calculate_channel_sizes(
            image_channels, max_filters, num_layers
        )

        # Encoder
        encoder_layers = nn.ModuleList()
        # Encoder Convolutions
        for i, (in_channels, out_channels) in enumerate(channel_sizes):
            if small_conv and i == 0:
                # 1x1 Convolution
                encoder_layers.append(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    )
                )
            else:
                # Convolutional Layer
                encoder_layers.append(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        bias=False,
                    )
                )
            # Batch Norm
            encoder_layers.append(nn.BatchNorm2d(out_channels))
            # ReLU
            encoder_layers.append(nn.ReLU())

        # Final Conv Layer to ensure we have embedding_dim channels
        if use_max_filters:
            encoder_layers.append(
                nn.Conv2d(
                    in_channels=out_channels,
                    out_channels=embedding_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
        # Make Encoder
        self.encoder = nn.Sequential(*encoder_layers)

        # Vector Quantizer
        self.vq_vae = VectorQuantizerEMA(
            num_embeddings, embedding_dim, commitment_cost, decay
        )

        # Decoder
        decoder_layers = nn.ModuleList()
        # Initial Conv Layer to change the channels back to the required number
        if use_max_filters:
            decoder_layers.append(
                nn.Conv2d(
                    in_channels=embedding_dim,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
        # Decoder Convolutions
        for i, (out_channels, in_channels) in enumerate(channel_sizes[::-1]):
            if small_conv and i == num_layers - 1:
                # 1x1 Transposed Convolution
                decoder_layers.append(
                    nn.ConvTranspose2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    )
                )
            else:
                # Add Transposed Convolutional Layer
                decoder_layers.append(
                    nn.ConvTranspose2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        bias=False,
                    )
                )
            # Batch Norm
            decoder_layers.append(nn.BatchNorm2d(out_channels))
            # ReLU if not final layer
            if i != num_layers - 1:
                decoder_layers.append(nn.ReLU())
            # Sigmoid if final layer
            else:
                decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

    @staticmethod
    def calculate_channel_sizes(image_channels, max_filters, num_layers):
        channel_sizes = [(image_channels, max_filters // np.power(2, num_layers - 1))]
        for i in range(1, num_layers):
            prev = channel_sizes[-1][-1]
            new = prev * 2
            channel_sizes.append((prev, new))
        return channel_sizes

    def forward(self, x):
        # Encoder
        encoded = self.encoder(x)
        # VQ-VAE
        loss, quantized, perplexity, encodings = self.vq_vae(encoded)
        # Decoder
        reconstructed = self.decoder(quantized)
        return loss, reconstructed, perplexity, encodings

    def quantize_and_decode(self, x, target_shape, device):
        quantized = self.vq_vae.quantize_encoding_indices(x, target_shape, device)
        return self.decoder(quantized)


class MultiScaleVQVAE(nn.Module):
    """
    Multi-Scale 版本的 VQ-VAE：透過雙 codebook 將「結構」與「細節顏色」拆開建模，
    讓 coarse latent 負責輪廓（大 patch），fine latent 保留像素級別的顏色資訊。
    """

    def __init__(
        self,
        image_channels=3,
        num_layers=4,
        kernel_size=2,
        stride=2,
        padding=0,
        input_image_dimensions=96,
        small_conv=False,
        embedding_dim=64,
        max_filters=512,
        use_max_filters=False,
        num_embeddings=512,
        commitment_cost=0.25,
        decay=0.99,
        structure_embedding_dim=None,
        structure_num_embeddings=None,
        structure_downsample_factor=4,
        upsample_mode="nearest",
    ):
        super(MultiScaleVQVAE, self).__init__()
        if small_conv:
            num_layers += 1
        if not use_max_filters:
            max_filters = embedding_dim
        if structure_embedding_dim is None:
            structure_embedding_dim = embedding_dim
        if structure_num_embeddings is None:
            structure_num_embeddings = num_embeddings
        assert structure_downsample_factor >= 1, "downsample factor 需為整數並且 >= 1"

        self.structure_downsample_factor = structure_downsample_factor
        channel_sizes = self.calculate_channel_sizes(
            image_channels, max_filters, num_layers
        )

        # Encoder：沿用原本 PixelSight + CNN 堆疊
        encoder_layers = nn.ModuleList()
        for i, (in_channels, out_channels) in enumerate(channel_sizes):
            if small_conv and i == 0:
                encoder_layers.append(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    )
                )
            else:
                encoder_layers.append(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        bias=False,
                    )
                )
            encoder_layers.append(nn.BatchNorm2d(out_channels))
            encoder_layers.append(nn.ReLU())

        if use_max_filters:
            encoder_layers.append(
                nn.Conv2d(
                    in_channels=out_channels,
                    out_channels=embedding_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

        self.encoder = nn.Sequential(*encoder_layers)
        self.encoder_out_channels = embedding_dim if use_max_filters else out_channels

        # 1x1 conv 壓縮，讓結構/細節有各自的通道維度
        self.detail_projection = (
            nn.Identity()
            if self.encoder_out_channels == embedding_dim
            else nn.Conv2d(
                in_channels=self.encoder_out_channels,
                out_channels=embedding_dim,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        )
        self.structure_projection = nn.Conv2d(
            in_channels=self.encoder_out_channels,
            out_channels=structure_embedding_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        # 結構/細節各自的 VQ codebook
        self.structure_vq = VectorQuantizerEMA(
            structure_num_embeddings, structure_embedding_dim, commitment_cost, decay
        )
        self.detail_vq = VectorQuantizerEMA(
            num_embeddings, embedding_dim, commitment_cost, decay
        )

        if structure_downsample_factor > 1:
            self.structure_downsampler = nn.AvgPool2d(
                kernel_size=structure_downsample_factor,
                stride=structure_downsample_factor,
                padding=0,
            )
            self.structure_upsampler = nn.Upsample(
                scale_factor=structure_downsample_factor, mode=upsample_mode
            )
        else:
            self.structure_downsampler = nn.Identity()
            self.structure_upsampler = nn.Identity()

        # Concat 後用 1x1 Conv 融合，讓 decoder 可以沿用原本架構
        self.fusion = nn.Sequential(
            nn.Conv2d(
                in_channels=embedding_dim + structure_embedding_dim,
                out_channels=embedding_dim,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(),
        )

        # Decoder：依舊是逐層反卷積，只是輸入換成融合後的 feature
        decoder_layers = nn.ModuleList()
        if use_max_filters:
            decoder_layers.append(
                nn.Conv2d(
                    in_channels=embedding_dim,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
        for i, (out_channels, in_channels) in enumerate(channel_sizes[::-1]):
            if small_conv and i == num_layers - 1:
                decoder_layers.append(
                    nn.ConvTranspose2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    )
                )
            else:
                decoder_layers.append(
                    nn.ConvTranspose2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        bias=False,
                    )
                )
            decoder_layers.append(nn.BatchNorm2d(out_channels))
            if i != num_layers - 1:
                decoder_layers.append(nn.ReLU())
            else:
                decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

    @staticmethod
    def calculate_channel_sizes(image_channels, max_filters, num_layers):
        channel_sizes = [(image_channels, max_filters // np.power(2, num_layers - 1))]
        for i in range(1, num_layers):
            prev = channel_sizes[-1][-1]
            new = prev * 2
            channel_sizes.append((prev, new))
        return channel_sizes

    def forward(self, x):
        encoded = self.encoder(x)

        # 結構分支：先做空間降採樣，再量化
        structure_features = self.structure_projection(
            self.structure_downsampler(encoded)
        )
        structure_loss, structure_quantized, structure_perplexity, structure_indices = (
            self.structure_vq(structure_features)
        )
        structure_quantized = self.structure_upsampler(structure_quantized)

        # 細節顏色分支：保留原空間解析度，直接量化顏色資訊
        detail_features = self.detail_projection(encoded)
        detail_loss, detail_quantized, detail_perplexity, detail_indices = self.detail_vq(
            detail_features
        )

        fused = self.fusion(torch.cat([structure_quantized, detail_quantized], dim=1))
        reconstructed = self.decoder(fused)

        total_loss = structure_loss + detail_loss
        perplexity = {
            "structure": structure_perplexity,
            "detail": detail_perplexity,
        }
        encodings = {"structure": structure_indices, "detail": detail_indices}
        return total_loss, reconstructed, perplexity, encodings

    def quantize_and_decode(self, encoding_indices, target_shapes, device):
        """
        encoding_indices: dict，需要同時提供 structure/detail 的 encoding。
        target_shapes: dict，需包含兩個 codebook 的 (B, H, W, C) 形狀，用來重建 quantized tensor。
        """
        if not isinstance(encoding_indices, dict):
            raise ValueError("MultiScaleVQVAE 需要 dict 形式的 encoding_indices。")
        if not isinstance(target_shapes, dict):
            raise ValueError("MultiScaleVQVAE 需要 dict 形式的 target_shapes。")

        structure_shape = target_shapes["structure"]
        detail_shape = target_shapes["detail"]
        structure_quantized = self.structure_vq.quantize_encoding_indices(
            encoding_indices["structure"], structure_shape, device
        )
        detail_quantized = self.detail_vq.quantize_encoding_indices(
            encoding_indices["detail"], detail_shape, device
        )
        structure_quantized = F.interpolate(
            structure_quantized,
            size=detail_quantized.shape[-2:],
            mode="nearest",
        )
        fused = self.fusion(torch.cat([structure_quantized, detail_quantized], dim=1))
        return self.decoder(fused)
