高斯噪声、胡椒噪声和盐噪声是图像处理和信号处理中常见的三种噪声类型。它们的主要区别在于噪声的分布和表现形式。以下是对这三种噪声的详细解释：

1. **高斯噪声（Gaussian Noise）**：
   - **定义**：高斯噪声是指像素值服从正态分布（高斯分布）的一种噪声。其概率密度函数呈钟形曲线，因此得名。
   - **特征**：高斯噪声的分布由均值（通常为零）和标准差（决定噪声的强度）参数化。噪声值可以是正数或负数，并且可以叠加在图像的每个像素上。
   - **表现形式**：高斯噪声在图像中表现为整体的细小颗粒状干扰，通常使图像变得模糊。
   - **应用**：模拟传感器噪声、量化误差等。

   ```python
   def add_gaussian_noise(image, mean=0, std=25):
       gauss = np.random.normal(mean, std, image.shape).astype('uint8')
       noisy_image = cv2.add(image, gauss)
       return noisy_image
   ```

2. **胡椒噪声（Pepper Noise）**：
   - **定义**：胡椒噪声是一种随机分布的黑色（最低灰度值）的像素点，通常伴随着盐噪声一起出现。
   - **特征**：胡椒噪声表现为图像中随机分布的黑色点，像是胡椒粒洒在图像上。
   - **表现形式**：胡椒噪声在图像中表现为随机的黑色像素点，通常影响图像的对比度和清晰度。
   - **应用**：模拟传输通道中的丢包现象或图像编码错误等。

   ```python
   def add_pepper_noise(image, prob=0.01):
       output = np.copy(image)
       black = np.zeros(image.shape[:2], dtype=np.bool)
       num_pepper = np.ceil(prob * image.size)
       coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
       output[coords[0], coords[1], :] = 0
       return output
   ```

3. **盐噪声（Salt Noise）**：
   - **定义**：盐噪声是一种随机分布的白色（最高灰度值）的像素点，通常伴随着胡椒噪声一起出现。
   - **特征**：盐噪声表现为图像中随机分布的白色点，像是盐粒洒在图像上。
   - **表现形式**：盐噪声在图像中表现为随机的白色像素点，通常影响图像的对比度和清晰度。
   - **应用**：模拟传输通道中的丢包现象或图像编码错误等。

   ```python
   def add_salt_noise(image, prob=0.01):
       output = np.copy(image)
       white = np.ones(image.shape[:2], dtype=np.bool)
       num_salt = np.ceil(prob * image.size)
       coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
       output[coords[0], coords[1], :] = 255
       return output
   ```

4. **盐与胡椒噪声（Salt-and-Pepper Noise）**：
   - **定义**：盐与胡椒噪声是盐噪声和胡椒噪声的混合形式，包含随机分布的黑点和白点。
   - **特征**：这种噪声表现为图像中随机分布的黑色和白色像素点。
   - **表现形式**：在图像中同时出现黑色和白色的噪声点，通常影响图像的对比度和清晰度。
   - **应用**：模拟传输通道中的丢包现象或图像编码错误等。

   ```python
   def add_salt_and_pepper_noise(image, prob=0.01):
       output = np.copy(image)
       num_salt = np.ceil(prob * image.size * 0.5)
       num_pepper = np.ceil(prob * image.size * 0.5)

       # 添加盐噪声
       coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
       output[coords[0], coords[1], :] = 255

       # 添加胡椒噪声
       coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
       output[coords[0], coords[1], :] = 0

       return output
   ```

总结：
- **高斯噪声**：整体的细小颗粒状干扰，分布符合高斯分布。
- **胡椒噪声**：随机分布的黑色像素点。
- **盐噪声**：随机分布的白色像素点。
- **盐与胡椒噪声**：随机分布的黑色和白色像素点混合。

这些噪声类型各自有不同的应用场景和处理方法，通常用于模拟实际环境中的各种噪声干扰，以测试图像处理算法的鲁棒性。