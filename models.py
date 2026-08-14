import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers

from setup import *



def sinusoidal_embedding(x):
    embedding_min_frequency = 1.0
    frequencies = tf.exp(
        tf.linspace(
            tf.math.log(embedding_min_frequency),
            tf.math.log(embedding_max_frequency),
            embedding_dims // 2,
        )
    )

    angular_speeds = 2.0 * math.pi * frequencies
    embeddings = tf.concat(
        [tf.sin(angular_speeds * x), tf.cos(angular_speeds * x)], axis=3
    )
    return embeddings


def ResidualBlock(width):
    def apply(x):
        input_width = x.shape[3]
        if input_width == width:
            residual = x
        else:
            residual = layers.Conv2D(width, kernel_size=1)(x)
        #x = layers.BatchNormalization(center=False, scale=False)(x)
        x = layers.LayerNormalization(axis=-1,center=True, scale=True)(x)
        x = layers.Conv2D(
            width, kernel_size=3, padding="same", activation=keras.activations.swish
        )(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same")(x)
        x = layers.Add()([x, residual])
        return x

    return apply


def DownBlock(width, block_depth):
    def apply(x):
        x, skips = x
        for _ in range(block_depth):
            x = ResidualBlock(width)(x)
            skips.append(x)
        x = layers.AveragePooling2D(pool_size=2)(x)
        return x

    return apply


def UpBlock(width, block_depth):
    def apply(x):
        x, skips = x
        x = layers.UpSampling2D(size=2, interpolation="bilinear")(x)
        for _ in range(block_depth):
            x = layers.Concatenate()([x, skips.pop()])
            x = ResidualBlock(width)(x)
        return x

    return apply


def get_network(image_size, input_frames, output_frames, widths, block_depth):
    noisy_images = keras.Input(shape=(image_size, image_size, input_frames+output_frames))
    noise_variances = keras.Input(shape=(1, 1, 1))

    e = layers.Lambda(sinusoidal_embedding)(noise_variances)
    e = layers.UpSampling2D(size=image_size, interpolation="nearest")(e)

    x = layers.Conv2D(widths[0], kernel_size=1)(noisy_images)
    x = layers.Concatenate()([x, e])

    skips = []
    for width in widths[:-1]:
        x = DownBlock(width, block_depth)([x, skips])

    for _ in range(block_depth):
        x = ResidualBlock(widths[-1])(x)

    for width in reversed(widths[:-1]):
        x = UpBlock(width, block_depth)([x, skips])

    x = layers.Conv2D(output_frames, kernel_size=1, kernel_initializer="zeros")(x)

    return keras.Model([noisy_images, noise_variances], x, name="residual_unet")


def get_post_network(image_size, input_frames, output_frames, widths, block_depth):
    noisy_images = keras.Input(shape=(image_size, image_size, input_frames))
    #noise_variances = keras.Input(shape=(1, 1, 1))

    #e = layers.Lambda(sinusoidal_embedding)(noise_variances)
    #e = layers.UpSampling2D(size=image_size, interpolation="nearest")(e)

    x = layers.Conv2D(widths[0], kernel_size=1)(noisy_images)
    #x = layers.Concatenate()([x, e])

    skips = []
    for width in widths[:-1]:
        x = DownBlock(width, block_depth)([x, skips])

    for _ in range(block_depth):
        x = ResidualBlock(widths[-1])(x)

    for width in reversed(widths[:-1]):
        x = UpBlock(width, block_depth)([x, skips])

    x = layers.Conv2D(output_frames, kernel_size=1, kernel_initializer="zeros")(x)

    return keras.Model([noisy_images], x, name="residual_unet")

class DiffusionModel(keras.Model):
    def __init__(self, image_size, input_frames, output_frames, widths, block_depth):
        super().__init__()

        self.normalizer = layers.Normalization()
        self.network = get_network(image_size, input_frames, output_frames, widths, block_depth)
        self.ema_network = keras.models.clone_model(self.network)
        self.input_frames = input_frames
        self.output_frames = output_frames

    def compile(self, **kwargs):
        super().compile(**kwargs)

        self.noise_loss_tracker = keras.metrics.Mean(name="n_loss")
        self.image_loss_tracker = keras.metrics.Mean(name="i_loss")

    @property
    def metrics(self):
        return [self.image_loss_tracker, self.noise_loss_tracker]

    def denormalize(self, images):
        # convert the pixel values back to 0-1 range
        images = self.normalizer.mean + images * self.normalizer.variance**0.5
        
        return tf.clip_by_value(images, 0.0, 1.0)

    def diffusion_schedule(self, diffusion_times):
        # diffusion times -> angles
        start_angle = tf.acos(max_signal_rate)
        end_angle = tf.acos(min_signal_rate)

        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

        # angles -> signal and noise rates
        signal_rates = tf.cos(diffusion_angles)
        noise_rates = tf.sin(diffusion_angles)
        # note that their squared sum is always: sin^2(x) + cos^2(x) = 1

        return noise_rates, signal_rates

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        # the exponential moving average weights are used at evaluation
        if training:
            network = self.network
        else:
            network = self.ema_network

        # predict noise component and calculate the image component using it
        pred_noises = network([noisy_images, noise_rates**2], training=training)
        pred_images = (noisy_images[:,:,:,-self.output_frames:] - noise_rates * pred_noises) / signal_rates

        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, diffusion_steps):
        # reverse diffusion = sampling
        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps
        past = initial_noise[:,:,:,:-self.output_frames]
        #future = initial_noise[-1]
        # important line:
        # at the first sampling step, the "noisy image" is pure noise
        # but its signal rate is assumed to be nonzero (min_signal_rate)
        next_noisy_images = initial_noise
        for step in range(diffusion_steps):
            noisy_images = next_noisy_images

            # separate the current noisy image to its components
            diffusion_times = tf.ones((num_images, 1, 1, 1)) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            #print("noisy im ",noisy_images.shape)
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=False
            )
            # network used in eval mode
            
            # remix the predicted components using the next signal and noise rates
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                next_diffusion_times
            )
            next_noisy_frames = (
                next_signal_rates * pred_images + next_noise_rates * pred_noises
            )
            
            #concatenate predicted single frame with past known frames 
            next_noisy_images = tf.concat([past, next_noisy_frames], axis = -1)

        return pred_images

    def generate2(self, images, diffusion_steps):
        # noise -> images -> denormalized images
        initial_noise = tf.random.normal(shape=(images.shape[0], images.shape[1], images.shape[2], self.output_frames))
        images[:,:,:,-self.output_frames:] = initial_noise
        generated_images = self.reverse_diffusion(images, diffusion_steps)
        generated_images = self.denormalize(tf.concat([images[:,:,:,:-self.output_frames],generated_images],axis=-1))
        return generated_images
        
    def train_step(self, images):
        # normalize images to have standard deviation of 1, like the noises
        
        #normalize only real images
        images = tf.cast(images, tf.float32)
        images = self.normalizer(images, training=True)
        noises = tf.random.normal(shape=(batch_size, image_size, image_size, self.output_frames))

        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(
            shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        target = images[:,:,:,-self.output_frames:]
        noisy_images = signal_rates * target + noise_rates * noises
        
        #concat the images with added noises with the originals 
        noise_two =  tf.concat([images[:,:,:,:-self.output_frames],noisy_images],axis=-1)
        #print(noise_two.shape)

        with tf.GradientTape() as tape:
            # train the network to separate noisy images to their components
            pred_noises, pred_images = self.denoise(
                noise_two, noise_rates, signal_rates, training=True
            )

            noise_loss = self.loss(noises, pred_noises)  # used for training
            image_loss = self.loss(target, pred_images)  # only used as metric

        # Training on noise_loss (default)
        gradients = tape.gradient(noise_loss, self.network.trainable_weights)
        
        # Training on image_loss
        #gradients = tape.gradient(image_loss, self.network.trainable_weights)
        
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        self.noise_loss_tracker.update_state(noise_loss)
        self.image_loss_tracker.update_state(image_loss)

        # track the exponential moving averages of weights
        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(ema * ema_weight + (1 - ema) * weight)
            
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, images):
        # normalize images to have standard deviation of 1, like the noises
        images = self.normalizer(images, training=False)
        noises = tf.random.normal(shape=(batch_size, image_size, image_size, self.output_frames))

        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(
            shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        target = images[:,:,:,-self.output_frames:]
        noisy_images = signal_rates * target + noise_rates * noises
        
        noise_two =  tf.concat([images[:,:,:,:-self.output_frames],noisy_images],axis=-1)

        # use the network to separate noisy images to their components
        pred_noises, pred_images = self.denoise(
            noise_two, noise_rates, signal_rates, training=False
        )

        noise_loss = self.loss(noises, pred_noises)
        image_loss = self.loss(target, pred_images)

        self.image_loss_tracker.update_state(image_loss)
        self.noise_loss_tracker.update_state(noise_loss)

        return {m.name: m.result() for m in self.metrics}
    
    def plotter(self, epoch, logs):
    # 1. 抓取樣本
        sample = self.test_generator50.__getitem__(1)
        hist = np.copy(sample)
    
    # 2. 生成預測 (使用模型內部的 normalizer)
        sample_norm = self.normalizer(sample)
    # 這裡 15 是擴散步數，你可以根據需要調整
        tmp_norm = self.generate2(np.copy(sample_norm), 15)
    
    # 3. 反標準化 (將 0~1 回推到實際物理數值)
    # 這裡假設降雨通道需要乘以 maxRtrain
        hist_phys = hist * self.maxRtrain
        tmp_phys = tmp_norm * self.maxRtrain

    # 4. 計算 MSE (保留你原本的邏輯，但增加可讀性)
    # 這會計算 16 個通道各自的平均誤差
        mse = np.mean(np.sum((hist_phys - tmp_phys)**2, axis=(1, 2)), axis=0)
        mse = np.round(mse, 8)
        print(f"\n📊 Epoch {epoch+1} - 各通道 MSE 數值 :")
        print(f"{mse}\n") 

    # 5. 繪製 T-2 到 T+3 的序列圖
        # 這裡微調寬度至 22，為右側獨立色標留點空間
        fig = plt.figure(figsize=(22, 5))
    # 標題對應：前三個是觀測 (Input)，後三個是預測 (Output)
        titles = ["Obs T-2", "Obs T-1", "Obs T", "Pred T+1", "Pred T+2", "Pred T+3"]
    
        for i in range(6):
            plt.subplot(1, 6, i + 1)
            if i < 3:
            # 畫觀測 (假設前三個降雨觀測通道在索引 0, 1, 2)
            # vmin/vmax 幫助固定顏色範圍，方便觀察降雨強度
                im = plt.imshow(hist_phys[0, :, :, 10+i], cmap='viridis', vmin=0, vmax=10)
            else:
            # 畫預測 (最後三個通道 T+1, T+2, T+3)
                im = plt.imshow(tmp_phys[0, :, :, 10+i], cmap='viridis', vmin=0, vmax=10)
            
            plt.title(titles[i])
            plt.axis("off")

        # 調整子圖邊距，手動在最右邊留出 8% 的空白給 colorbar，防止圖表被自動擠壓
        fig.subplots_adjust(left=0.03, right=0.92, top=0.9, bottom=0.1, wspace=0.15)
        
        # 單獨建立一個獨立的色標座標軸 [左, 下, 寬, 高]
        cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
        fig.colorbar(im, cax=cbar_ax, label='Rain Rate (mm/h)')

    # 存檔路徑確保與你的資料夾結構一致
        plt.savefig(f'training_results/plots/epoch_{epoch+1}_seq.png')
        plt.close()
