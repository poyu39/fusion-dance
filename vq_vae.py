"""
Primary code to train the Pixel VQ-VAE.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pytorch_msssim
import torch
import torch.nn as nn
import utils.data as data
import utils.graphics as graphics
import utils.loss as loss
from logger import setup_logger
from models import vqvae
from tqdm import tqdm

seed = 39
np.random.seed(seed)
_ = torch.manual_seed(seed)

################################################################################
#################################### Config ####################################
################################################################################

# experiment_name = f"vq_vae_v5.10_emb8"
# experiment_name = f"vq_vae_pixel_aware_loss"
experiment_name = f"vq_vae_multi_scale_codebook84_emb8"

output_prefix = f"outputs/{experiment_name}"
os.makedirs(output_prefix, exist_ok=True)

logger = setup_logger(name="VQ_VAE_Logger", project_root=os.getcwd(), log_file=os.path.join(output_prefix, "vq_vae.log"))

# Hyperparameters
learning_rate = 1e-4
epochs = 25
batch_size = 64
num_dataloader_workers = 12

# VQ-VAE Config
num_layers = 0
num_embeddings = 256
# embedding_dim = 32
embedding_dim = 8
commitment_cost = 0.25
use_max_filters = True # PixelSight Layer
max_filters = 512
small_conv = True  # Adapter Layer
use_multi_scale_codebook = True  # True 啟用多尺度 codebook（結構 vs 細節）
structure_downsample_factor = 8  # z_struct 使用 M=4 的降採樣感受野
structure_embedding_dim = embedding_dim  # 可視需要放大/縮小結構 token 維度
structure_num_embeddings = max(1, num_embeddings // 4)  # 結構 codebook 規模，預設減半以鼓勵語意分工

# Loss Config
use_sum = False  # Use a sum instead of a mean for our loss function
use_ssim_loss = True  # Pixel-aware Loss 預設會搭配 SSIM（可切換）
use_pixel_aware_loss = False  # True 時會使用 Pixel-aware 重建 loss
mse_weight = 1
ssim_weight = 0.15
palette_loss_weight = 0.5
palette_loss_kwargs = {
    "hue_weight": 1.0,  # 色相權重：強化顏色 index 差異
    "saturation_weight": 0.5,  # 飽和度權重：控制填色純度
    "value_weight": 0.25,  # 明度權重：保留亮暗資訊
}

# Data Config
image_size = 64
use_noise_images = True
load_data_to_memory = True

data_prefix = './data/Pokemon/original_data_preprocessed'
# data_prefix = "data\\Pokemon\\final\\standard"
train_data_folder = os.path.join(data_prefix, "train")
val_data_folder = os.path.join(data_prefix, "val")
test_data_folder = os.path.join(data_prefix, "test")

output_dir = os.path.join(output_prefix, "generated")
loss_output_path = output_prefix
model_output_path = os.path.join(output_prefix, "model.pt")

animation_output_path = os.path.join(output_prefix, "animation.mp4")
animation_sample_image_name = os.path.join(output_prefix, "animation_base.jpg")

test_sample_input_name = os.path.join(output_prefix, "test_sample_input.jpg")
test_sample_output_name = os.path.join(output_prefix, "test_sample_output.jpg")
################################################################################
##################################### Setup ####################################
################################################################################

# Setup Device
gpu = torch.cuda.is_available()
device = torch.device("cuda" if gpu else "cpu")
logger.info(f"GPU Available: {gpu}, Device: {device}")

# Create Output Paths
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

################################################################################
################################## Data Setup ##################################
################################################################################

# Preprocess & Create Data Loaders
transform = data.image2tensor_resize(image_size)

if load_data_to_memory:
    # Load Data
    train = data.load_images_from_folder(train_data_folder, use_noise_images)
    val = data.load_images_from_folder(val_data_folder, use_noise_images)
    test = data.load_images_from_folder(test_data_folder, use_noise_images)

    train_data = data.CustomDataset(train, transform)
    val_data = data.CustomDataset(val, transform)
    test_data = data.CustomDataset(test, transform)
else:
    train_data = data.CustomDatasetNoMemory(train_data_folder, transform, use_noise_images)
    val_data = data.CustomDatasetNoMemory(val_data_folder, transform, use_noise_images)
    test_data = data.CustomDatasetNoMemory(test_data_folder, transform, use_noise_images)
    

train_dataloader = torch.utils.data.DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_dataloader_workers,
    pin_memory=gpu,
)
val_dataloader = torch.utils.data.DataLoader(
    val_data,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_dataloader_workers,
    pin_memory=gpu,
)
test_dataloader = torch.utils.data.DataLoader(
    test_data,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_dataloader_workers,
    pin_memory=gpu,
)

# Creating a sample set to visualize the model's training
sample = data.get_samples_from_data(val_data, 16)

################################################################################
##################################### Model ####################################
################################################################################

# Create Model
if use_multi_scale_codebook:
    # 2-A Multi-Scale 版本：結構 codebook 先把形狀建起來，再補細節顏色
    model = vqvae.MultiScaleVQVAE(
        num_layers=num_layers,
        input_image_dimensions=image_size,
        small_conv=small_conv,
        embedding_dim=embedding_dim,
        num_embeddings=num_embeddings,
        commitment_cost=commitment_cost,
        use_max_filters=use_max_filters,
        max_filters=max_filters,
        structure_embedding_dim=structure_embedding_dim,
        structure_num_embeddings=structure_num_embeddings,
        structure_downsample_factor=structure_downsample_factor,
    )
else:
    # 舊實驗：單一 codebook，方便做消融對比
    model = vqvae.VQVAE(
        num_layers=num_layers,
        input_image_dimensions=image_size,
        small_conv=small_conv,
        embedding_dim=embedding_dim,
        num_embeddings=num_embeddings,
        commitment_cost=commitment_cost,
        use_max_filters=use_max_filters,
        max_filters=max_filters,
    )
logger.info(f"Model Architecture:\n{model}")
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

ssim_module = None
if use_ssim_loss:
    ssim_module = pytorch_msssim.SSIM(
        data_range=1.0, win_size=11, win_sigma=1.5, K=(0.01, 0.03)
    )


def reconstruction_loss_fn(reconstructed, target):
    """依設定切換 Pixel-aware loss 或原始 MSE+SSIM，方便進行消融。"""
    if use_pixel_aware_loss:
        return loss.pixel_aware_reconstruction_loss(
            reconstructed,
            target,
            use_sum=use_sum,
            ssim_module=ssim_module if use_ssim_loss else None,
            mse_weight=mse_weight,
            ssim_weight=ssim_weight,
            palette_weight=palette_loss_weight,
            palette_kwargs=palette_loss_kwargs,
        )
    return loss.mse_ssim_loss(
        reconstructed,
        target,
        use_sum=use_sum,
        ssim_module=ssim_module if use_ssim_loss else None,
        mse_weight=mse_weight,
        ssim_weight=ssim_weight,
    )


def parse_perplexity(perplexity_output):
    """把 dict/float 形式的 perplexity 統一成 dict，方便記錄與平均。"""
    if isinstance(perplexity_output, dict):
        return {
            k: (v.item() if torch.is_tensor(v) else float(v))
            for k, v in perplexity_output.items()
        }
    value = (
        perplexity_output.item()
        if torch.is_tensor(perplexity_output)
        else float(perplexity_output)
    )
    return {"latent": value}

################################################################################
################################### Training ###################################
################################################################################

# Log training configuration
logger.info("="*80)
logger.info("Training Configuration:")
logger.info("="*80)
logger.info(f"Experiment Name: {experiment_name}")
logger.info(f"Learning Rate: {learning_rate}")
logger.info(f"Epochs: {epochs}")
logger.info(f"Batch Size: {batch_size}")
logger.info(f"Num Dataloader Workers: {num_dataloader_workers}")
logger.info(f"Image Size: {image_size}")
logger.info(f"Use Noise Images: {use_noise_images}")
logger.info(f"Load Data to Memory: {load_data_to_memory}")
logger.info("-"*80)
logger.info(f"VQ-VAE Configuration:")
logger.info(f"  Num Layers: {num_layers}")
logger.info(f"  Num Embeddings: {num_embeddings}")
logger.info(f"  Embedding Dim: {embedding_dim}")
logger.info(f"  Commitment Cost: {commitment_cost}")
logger.info(f"  Use Max Filters: {use_max_filters}")
logger.info(f"  Max Filters: {max_filters}")
logger.info(f"  Small Conv: {small_conv}")
logger.info(f"  Use Multi-Scale Codebook: {use_multi_scale_codebook}")
if use_multi_scale_codebook:
    logger.info(f"    Structure Downsample Factor: {structure_downsample_factor}")
    logger.info(f"    Structure Embedding Dim: {structure_embedding_dim}")
    logger.info(f"    Structure Num Embeddings: {structure_num_embeddings}")
logger.info("-"*80)
logger.info(f"Loss Configuration:")
logger.info(f"  Use Sum: {use_sum}")
logger.info(f"  Use SSIM Loss: {use_ssim_loss}")
logger.info(f"  Use Pixel-aware Loss: {use_pixel_aware_loss}")
logger.info(f"  MSE Weight: {mse_weight}")
logger.info(f"  SSIM Weight: {ssim_weight}")
logger.info(f"  Palette Weight: {palette_loss_weight}")
logger.info(
    "  Palette HSV Weights: "
    f"H={palette_loss_kwargs['hue_weight']}, "
    f"S={palette_loss_kwargs['saturation_weight']}, "
    f"V={palette_loss_kwargs['value_weight']}"
)
logger.info("-"*80)
logger.info(f"Data Configuration:")
logger.info(f"  Train Data: {train_data_folder}")
logger.info(f"  Val Data: {val_data_folder}")
logger.info(f"  Test Data: {test_data_folder}")
logger.info(f"  Train Dataset Size: {len(train_data)}")
logger.info(f"  Val Dataset Size: {len(val_data)}")
logger.info(f"  Test Dataset Size: {len(test_data)}")
logger.info("-"*80)
logger.info(f"Output Configuration:")
logger.info(f"  Output Directory: {output_dir}")
logger.info(f"  Model Output Path: {model_output_path}")
logger.info("="*80)
logger.info("Starting Training...")
logger.info("="*80)

# Train
all_samples = []
all_train_loss = []
train_perplexity = []
all_val_loss = []
val_perplexity = []

# Get an initial "epoch 0" sample
model.eval()
with torch.no_grad():
    _, epoch_sample, _, _ = model(sample.to(device))

# Add sample reconstruction to our list
all_samples.append(epoch_sample.detach().cpu())

for epoch in tqdm(range(epochs), desc="Epochs"):
    train_loss = 0
    train_recon_loss = 0
    train_vq_loss = 0
    train_epoch_perplexity = []
    train_perplexity_breakdown = {}
    val_loss = 0
    val_recon_loss = 0
    val_vq_loss = 0
    val_epoch_perplexity = []
    val_perplexity_tracking = {}

    # Training Loop
    model.train()
    for iteration, batch in enumerate(tqdm(train_dataloader, desc=f"Training")):
        # Reset gradients back to zero for this iteration
        optimizer.zero_grad()

        # Move batch to device
        _, batch = batch  # Returns key, value for each Pokemon
        batch = batch.to(device)

        # Run our model & get outputs
        vq_loss, reconstructed, perplexity, _ = model(batch)

        # Calculate reconstruction loss（Pixel-aware 可視設定切換）
        batch_loss, loss_dict = reconstruction_loss_fn(reconstructed, batch)

        # Add VQ-Loss to Overall Loss
        batch_loss += vq_loss
        loss_dict["Commitment Loss"] = vq_loss.item()

        # Backprop
        batch_loss.backward()

        # Update our optimizer parameters
        optimizer.step()

        # Add the batch's loss to the total loss for the epoch
        train_loss += batch_loss.item()
        recon_component = loss_dict["MSE"] + loss_dict["SSIM"] + loss_dict.get("Palette", 0)
        train_recon_loss += recon_component
        train_vq_loss += loss_dict["Commitment Loss"]
        perplexity_dict = parse_perplexity(perplexity)
        train_epoch_perplexity.append(np.mean(list(perplexity_dict.values())))
        for key, value in perplexity_dict.items():
            train_perplexity_breakdown.setdefault(key, []).append(value)

    # Validation Loop
    model.eval()
    with torch.no_grad():
        for iteration, batch in enumerate(tqdm(val_dataloader, desc=f"Validation")):
            # Move batch to device
            _, batch = batch  # Returns key, value for each Pokemon
            batch = batch.to(device)

            # Run our model & get outputs
            vq_loss, reconstructed, perplexity, _ = model(batch)

            # Calculate reconstruction loss（驗證同樣沿用 Pixel-aware 切換邏輯）
            batch_loss, loss_dict = reconstruction_loss_fn(reconstructed, batch)

            # Add VQ-Loss to Overall Loss
            batch_loss += vq_loss
            loss_dict["Commitment Loss"] = vq_loss.item()

            # Add the batch's loss to the total loss for the epoch
            val_loss += batch_loss.item()
            recon_component = loss_dict["MSE"] + loss_dict["SSIM"] + loss_dict.get("Palette", 0)
            val_recon_loss += recon_component
            val_vq_loss += loss_dict["Commitment Loss"]
            perplexity_dict = parse_perplexity(perplexity)
            val_epoch_perplexity.append(np.mean(list(perplexity_dict.values())))
            for key, value in perplexity_dict.items():
                val_perplexity_tracking.setdefault(key, []).append(value)

        # Get reconstruction of our sample
        _, epoch_sample, _, _ = model(sample.to(device))

    # Add sample reconstruction to our list
    all_samples.append(epoch_sample.detach().cpu())

    # Compute the average losses for this epoch
    train_loss = train_loss / len(train_dataloader)
    train_recon_loss = train_recon_loss / len(train_dataloader)
    train_vq_loss = train_vq_loss / len(train_dataloader)
    all_train_loss.append((train_loss, train_recon_loss, train_vq_loss))
    train_epoch_perplexity = np.mean(train_epoch_perplexity)
    train_perplexity.append(train_epoch_perplexity)
    train_perplexity_stats = {
        key: float(np.mean(values)) for key, values in train_perplexity_breakdown.items()
    }

    val_loss = val_loss / len(val_dataloader)
    val_recon_loss = val_recon_loss / len(val_dataloader)
    val_vq_loss = val_vq_loss / len(val_dataloader)
    all_val_loss.append((val_loss, val_recon_loss, val_vq_loss))
    val_epoch_perplexity = np.mean(val_epoch_perplexity)
    val_perplexity.append(val_epoch_perplexity)
    val_perplexity_stats = {
        key: float(np.mean(values)) for key, values in val_perplexity_tracking.items()
    }

    # Print Metrics
    logger.info(
        f"\nEpoch: {epoch+1}/{epochs}:\
        \nTrain Loss = {train_loss}\
        \nTrain Reconstruction Loss = {train_recon_loss}\
        \nTrain Commitment Loss = {train_vq_loss}\
        \nTrain Perplexity = {train_epoch_perplexity}\
        \nTrain Perplexity Breakdown = {train_perplexity_stats}\
        \nVal Loss = {val_loss}\
        \nVal Reconstruction Loss = {val_recon_loss}\
        \nVal Commitment Loss = {val_vq_loss}\
        \nVal Perplexity = {val_epoch_perplexity}\
        \nVal Perplexity Breakdown = {val_perplexity_stats}"
    )

################################################################################
################################## Save & Test #################################
################################################################################
# Generate Loss Graph
graphics.draw_loss(all_train_loss, all_val_loss, loss_output_path, mode="vqvae")
graphics.plot_and_save_loss(
    train_perplexity,
    "Train Perplexity",
    val_perplexity,
    "Validation Perplexity",
    os.path.join(loss_output_path, "perplexity.jpg"),
)

# Save Model
torch.save(model.state_dict(), model_output_path)

# Plot Animation Sample
fig, axis = graphics.make_grid(("Sample", sample), 4, 4)
plt.savefig(animation_sample_image_name)

# Create & Save Animation
anim = graphics.make_animation(graphics.make_grid, all_samples)
anim.save(animation_output_path)

model.eval()

# Evaluate on Test Images
# Save Generated Images & Calculate Metrics
# Testing Loop - Standard
all_mse = []
all_ssim = []
with torch.no_grad():
    for iteration, batch in enumerate(tqdm(test_dataloader)):
        # Move batch to device
        filenames, image = batch
        image = image.to(device)

        # Run our model & get outputs
        _, reconstructed, _, _ = model(image)

        # Calculate Metrics
        mse = nn.functional.mse_loss(reconstructed, image)
        ssim_score = pytorch_msssim.ssim(
            reconstructed,
            image,
            data_range=1.0,
            win_size=11,
            win_sigma=1.5,
            K=(0.01, 0.03),
        )

        # Add metrics to tracking list
        all_mse.append(mse.detach().cpu().numpy())
        all_ssim.append(ssim_score.detach().cpu().numpy())

        # Save
        reconstructed = reconstructed.permute(0, 2, 3, 1).detach().cpu().numpy()
        for image, filename in zip(reconstructed, filenames):
            plt.imsave(os.path.join(output_dir, filename), image)

# Print Metrics
mse = np.asarray(all_mse).mean()
ssim_score = np.asarray(all_ssim).mean()
logger.info(f"\nTest Metrics - MSE = {mse}, SSIM = {ssim_score}")

# Pick a couple of sample images for an Input v Output comparison
test_sample = data.get_samples_from_data(test_data, 16)

# Plot A Set of Test Images
fig, axis = graphics.make_grid(("Test Sample", test_sample), 4, 4)
plt.savefig(test_sample_input_name)

with torch.no_grad():
    reconstructed = model(test_sample.to(device))[1].detach().cpu()

# Plot A Set of Reconstructed Test Images
fig, axis = graphics.make_grid(("Test Sample", reconstructed), 4, 4)
plt.savefig(test_sample_output_name)
