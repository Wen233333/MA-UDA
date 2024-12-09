import torch.nn as nn


class AttentionDiscriminator(nn.Module):

	def __init__(self, num_classes=69, ndf=64):
		super(AttentionDiscriminator, self).__init__()
		self.conv = nn.Conv2d(num_classes, ndf*32, kernel_size=1, stride=1, padding=0)
		self.gap = nn.AdaptiveAvgPool2d((1,1))
		# self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=1, stride=1, padding=0)
		self.conv1 = nn.Conv2d(ndf*32, ndf, kernel_size=1, stride=1, padding=0)
		self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=1, stride=1, padding=0)
		self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=1, stride=1, padding=0)
		self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=1, stride=1, padding=0)
		self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=1, stride=1, padding=0)

		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
		# self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
		# self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		x = self.conv(x)
		x = self.gap(x)
		x = self.conv1(x)
		x = self.leaky_relu(x)
		x = self.conv2(x)
		x = self.leaky_relu(x)
		x = self.conv3(x)
		x = self.leaky_relu(x)
		x = self.conv4(x)
		x = self.leaky_relu(x)
		x = self.classifier(x)
		# x = self.up_sample(x)
		# x = self.sigmoid(x)

		return x


class Discriminator2D(nn.Module):

	def __init__(self, num_classes=69, ndf=64):
		super(Discriminator2D, self).__init__()

		self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=3, stride=1, padding=1)
		self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=3, stride=1, padding=1)
		self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=3, stride=1, padding=1)
		self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=3, stride=1, padding=1)
		self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=3, stride=1, padding=1)

		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
		# self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
		# self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		x = self.conv1(x)
		x = self.leaky_relu(x)
		x = self.conv2(x)
		x = self.leaky_relu(x)
		x = self.conv3(x)
		x = self.leaky_relu(x)
		x = self.conv4(x)
		x = self.leaky_relu(x)
		x = self.classifier(x)
		# x = self.up_sample(x)
		# x = self.sigmoid(x)

		return x


class FeatureDiscriminator(nn.Module):

	def __init__(self, input_size=2048, ndf=64):
		super(FeatureDiscriminator, self).__init__()

		self.conv1 = nn.Conv2d(input_size, ndf, kernel_size=1, stride=1, padding=0)
		self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=1, stride=1, padding=0)
		self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=1, stride=1, padding=0)
		self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=1, stride=1, padding=0)
		self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=1, stride=1, padding=0)

		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
		# self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
		# self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		x = self.conv1(x)
		x = self.leaky_relu(x)
		x = self.conv2(x)
		x = self.leaky_relu(x)
		x = self.conv3(x)
		x = self.leaky_relu(x)
		x = self.conv4(x)
		x = self.leaky_relu(x)
		x = self.classifier(x)
		# x = self.up_sample(x)
		# x = self.sigmoid(x)

		return x
